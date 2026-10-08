"""Trained-weights path: tiling, class mapping, hybrid GOST refinement, fallbacks.

A fake ultralytics-like model returns (jittered) ground-truth boxes of the
demo schematic - no ultralytics / GPU / real weights needed.
"""

from pathlib import Path

import numpy as np
import pytest

from avers.config import AVERSConfig
from avers.core.types import DETECTION_CLASSES
from avers.core.validators import ProductionPipeline, SlicedDetector
from avers.dataset.demo_schematic import build_demo, evaluate
from avers.stages.stage2_detection import model_detector as md
from avers.stages.stage2_detection.model_detector import (
    ModelClassMismatch, UltralyticsTiledDetector, map_model_classes, nms_per_class, tile_grid,
)


class _Boxes:
    def __init__(self, rows):
        arr = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = arr[:, :4], arr[:, 4], arr[:, 5]

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class FakeModel:
    """Mimics ultralytics YOLO.predict on tiles of one known sheet."""

    def __init__(self, image, objects, names=None, jitter=0.08, drop=(), seed=0):
        self.image_bgr = np.ascontiguousarray(image[:, :, ::-1])
        self.names = names or dict(DETECTION_CLASSES)
        ids = {n: i for i, n in self.names.items()}
        rng = np.random.default_rng(seed)
        self.objects = []
        for k, (name, (x0, y0, x1, y1)) in enumerate(objects):
            if k in drop:
                continue
            w, h = x1 - x0, y1 - y0
            j = rng.uniform(-jitter, jitter, 4) * [w, h, w, h]
            self.objects.append((x0 + j[0], y0 + j[1], x1 + j[2], y1 + j[3], 0.9, ids[name]))
        self.calls = 0

    grid = (2048, 0.2)  # (tile, overlap) the detector under test uses

    def predict(self, tile, **kw):
        self.calls += 1
        th, tw = tile.shape[:2]
        H, W = self.image_bgr.shape[:2]
        for tx, ty, tx1, ty1 in tile_grid(W, H, *self.grid):
            if (ty1 - ty, tx1 - tx) != (th, tw):
                continue
            if np.array_equal(self.image_bgr[ty:ty1, tx:tx1], tile):
                rows = [(x0 - tx, y0 - ty, x1 - tx, y1 - ty, c, k)
                        for x0, y0, x1, y1, c, k in self.objects
                        if x0 >= tx - 2 and y0 >= ty - 2 and x1 <= tx1 + 2 and y1 <= ty1 + 2]
                return [_Result(rows)]
        return [_Result([])]


def _gt_objects(truth):
    return [(c["class_name"], tuple(c["bbox"])) for c in truth["components"]]


def test_tile_grid_covers_sheet_with_overlap():
    tiles = tile_grid(2500, 1300, 1024, 0.2)
    assert tiles[0] == (0, 0, 1024, 1024)
    assert max(t[2] for t in tiles) == 2500 and max(t[3] for t in tiles) == 1300
    assert all(t[2] - t[0] <= 1024 and t[3] - t[1] <= 1024 for t in tiles)
    assert tile_grid(800, 600, 1024, 0.2) == [(0, 0, 800, 600)]


def test_nms_merges_duplicates_and_tile_edge_fragments():
    dets = [
        {"bbox": (100, 100, 200, 140), "confidence": 0.9, "category": "resistor"},
        {"bbox": (102, 101, 199, 141), "confidence": 0.8, "category": "resistor"},   # duplicate
        {"bbox": (150, 100, 200, 140), "confidence": 0.7, "category": "resistor"},   # cut at tile edge
        {"bbox": (100, 100, 200, 140), "confidence": 0.6, "category": "fuse"},       # other class kept
    ]
    out = nms_per_class(dets)
    assert sorted((d["category"], d["confidence"]) for d in out) == [("fuse", 0.6), ("resistor", 0.9)]


def test_coco_weights_are_rejected_not_silently_used():
    coco = {0: "person", 1: "bicycle", 2: "car"}
    assert map_model_classes(coco) == {}
    det = UltralyticsTiledDetector("x.pt", model=type("M", (), {"names": coco})())
    with pytest.raises(ModelClassMismatch):
        det.load()


def test_missing_weights_fall_back_with_explicit_warning():
    image, _ = build_demo()
    config = AVERSConfig()
    config.detection.model_path = "C:/nonexistent/avers_best.pt"
    result = ProductionPipeline(config).run(image, "demo.png")
    assert any("Обученный детектор не используется" in w for w in result.warnings)
    assert any("шаблонный детектор" in w for w in result.warnings)
    assert len(result.manifest.components) > 10


@pytest.mark.parametrize("tile", [2048, 700])
def test_hybrid_recovers_demo_netlist_from_model_boxes(tile):
    image, truth = build_demo()
    fake = FakeModel(image, _gt_objects(truth))
    fake.grid = (tile, 0.25)
    detector = SlicedDetector(model=fake, slice_size=tile, overlap_ratio=0.25,
                              template_fallback=True, merge_templates=False)
    assert detector.load() and detector.backend == "model"
    dets = detector.detect(image)
    assert detector.last_stats["by_source"].get("geometry", 0) == 0
    by_class = {}
    for d in dets:
        by_class.setdefault(d["category"], []).append(d)
    # rotation/pins come from the GOST templates, not from the model
    diode = by_class["diode"][0]
    assert diode["pins_source"] == "template" and diode["rotation"] in (90, 270)
    assert {p["name"] for p in diode["pins"]} == {"A", "K"}
    assert by_class["connector_body"][0]["pins_source"] == "table"


def test_pipeline_with_model_backend_gives_exact_netlist(monkeypatch):
    image, truth = build_demo()
    fake = FakeModel(image, _gt_objects(truth), drop=(3,))  # model misses one symbol
    original = md.UltralyticsTiledDetector.load

    def load(self):
        self.model = fake
        return original(self)

    monkeypatch.setattr(md.UltralyticsTiledDetector, "load", load)
    config = AVERSConfig()
    config.detection.model_path = "fake_avers_weights.pt"
    config.slicing.tile_size = 2048

    result = ProductionPipeline(config).run(image, "demo.png")
    ev = evaluate(result.manifest.model_dump(mode="json"), truth)

    assert not any("шаблонный детектор по библиотеке" in w for w in result.warnings)
    assert not any("не используется" in w for w in result.warnings)
    # the missed symbol is added back by the template detector (merge_templates)
    assert ev["components_found"] == ev["components_expected"]
    assert ev["nets_matched"] == ev["nets_expected"], (ev["missing_nets"], ev["extra_nets"])


def test_registry_current_version_is_used_when_model_path_empty(tmp_path):
    from avers.active_learning.registry import ModelRegistry

    config = AVERSConfig()
    config.active_learning.registry_dir = str(tmp_path / "registry")
    pipeline = ProductionPipeline(config)
    assert pipeline._resolve_model_path() == (None, "none")
    assert not (tmp_path / "registry").exists()  # no side effects

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    registry = ModelRegistry(tmp_path / "registry")
    v1 = registry.register(weights, "yolo11s")          # first -> promoted
    path, source = pipeline._resolve_model_path()
    assert Path(path).name == "best.pt" and source == f"registry {v1.version_id}"

    config.detection.model_path = "explicit.pt"           # explicit config wins
    assert pipeline._resolve_model_path() == ("explicit.pt", "config")
    config.detection.model_path = None
    config.detection.use_registry = False
    assert pipeline._resolve_model_path() == (None, "none")


def test_geometry_fallback_still_gives_pins():
    from avers.stages.stage2_detection.template_detector import GOSTTemplateDetector
    blank = np.full((300, 300, 3), 255, np.uint8)  # nothing to match -> geometry
    dets = GOSTTemplateDetector().refine(blank, [
        {"bbox": (100, 50, 124, 146), "confidence": 0.9, "category": "resistor", "category_id": 8},
    ])
    d, = dets
    assert d["pins_source"] == "geometry" and d["rotation"] == 90
    ys = sorted(p["coord"][1] for p in d["pins"])
    assert ys[0] <= 52 and ys[1] >= 144  # leads at the short ends of a vertical resistor
