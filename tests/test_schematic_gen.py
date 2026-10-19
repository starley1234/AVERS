"""Генератор схем v2 (avers.dataset.schematic_gen): корректность эталона и экспорт YOLO."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from avers.core.types import DETECTION_CLASSES
from avers.dataset.gost_symbols import GOST_SYMBOLS
from avers.dataset.schematic_gen import (
    GenConfig, dataset_sheet, degrade, export_yolo_dataset, generate_sheet, sheet_issues, yolo_boxes,
)


def _dark(rgb, x, y, r=2):
    h, w = rgb.shape[:2]
    win = rgb[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    return win.size and win.min() < 128


def test_deterministic_by_seed():
    a, ta = generate_sheet(3)
    b, tb = generate_sheet(3)
    assert np.array_equal(a, b)
    assert ta == tb
    c, _ = generate_sheet(4)
    assert a.shape != c.shape or not np.array_equal(a, c)


@pytest.mark.parametrize("seed", range(6))
def test_truth_is_consistent(seed):
    rgb, truth = generate_sheet(seed)
    comps = {c["designator"]: c for c in truth["components"]}
    assert len(comps) == len(truth["components"]), "designators must be unique"
    # every net member is an existing pin; every net has >= 2 members
    for net in truth["nets"]:
        assert len(net) >= 2
        for ref in net:
            des, pin = ref.rsplit(":", 1)
            assert des in comps and pin in comps[des]["pins"], ref
    # pins lie on ink, inside the image
    h, w = rgb.shape[:2]
    for c in truth["components"]:
        x0, y0, x1, y1 = c["bbox"]
        assert 0 <= x0 < x1 < w and 0 <= y0 < y1 < h
        for v in c["pins"].values():
            for x, y in (v if isinstance(v[0], list) else [v]):
                assert _dark(rgb, x, y), (c["designator"], x, y)
    for x, y, _ in truth["junctions"]:
        assert _dark(rgb, x, y, 0)
    # no pin is left unconnected (generator draws a wire to every pin, except
    # unused connector cells)
    connected = {ref for net in truth["nets"] for ref in net}
    for c in truth["components"]:
        if c["class_name"] == "connector_body":
            continue
        for p in c["pins"]:
            assert f"{c['designator']}:{p}" in connected, (c["designator"], p)


def test_layout_has_no_conflicts():
    """Нет наложений УГО/подписей, проводов через УГО и Т-примыканий без точки."""
    bad = {s: sheet_issues(s)[:3] for s in range(40) if sheet_issues(s)}
    assert len(bad) <= 1, bad


def test_all_library_classes_are_generated():
    seen = set()
    for s in range(40):
        _, truth = generate_sheet(s)
        seen |= {c["class_name"] for c in truth["components"]}
    drawable = {s.class_name for s in GOST_SYMBOLS.values() if s.draw_fn} - {"pin", "junction_dot", "shield"}
    assert drawable <= seen, drawable - seen


def test_template_only_config():
    from avers.dataset.gost_symbols import get_symbol_by_name
    for s in range(5):
        _, truth = generate_sheet(s, GenConfig(only_template_classes=True))
        for c in truth["components"]:
            sym = get_symbol_by_name(c["class_name"])
            assert sym.template_detect or c["class_name"] == "connector_body"


def test_degrade_keeps_geometry():
    rgb, truth = generate_sheet(1)
    out, t2 = degrade(rgb, truth, seed=5, strength=1.0)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert t2["nets"] == truth["nets"]
    hits = total = 0
    for c in t2["components"]:
        for v in c["pins"].values():
            for x, y in (v if isinstance(v[0], list) else [v]):
                total += 1
                hits += bool(out[max(0, y - 3):y + 4, max(0, x - 3):x + 4].min() < 170)
    assert hits / total > 0.9


def test_yolo_export(tmp_path):
    yaml_path = export_yolo_dataset(tmp_path, num_train=2, num_val=1, seed=0, tile=640, ext="jpg")
    text = yaml_path.read_text(encoding="utf-8")
    assert f"nc: {len(DETECTION_CLASSES)}" in text
    for cid, name in DETECTION_CLASSES.items():
        assert f"  {cid}: {name}" in text
    labels = list((tmp_path / "labels" / "train").glob("*.txt"))
    images = list((tmp_path / "images" / "train").glob("*.jpg"))
    assert labels and len(labels) == len(images)
    rows = [line.split() for p in labels for line in p.read_text().splitlines() if line]
    assert rows
    for r in rows:
        assert 0 <= int(r[0]) < len(DETECTION_CLASSES)
        assert all(0.0 <= float(v) <= 1.0 for v in r[1:])
    assert len(list((tmp_path / "truth" / "val").glob("*.json"))) == 1
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["objects"] == len(rows) + sum(
        len(Path(p).read_text().splitlines()) for p in (tmp_path / "labels" / "val").glob("*.txt"))


def test_dataset_sheet_matches_export(tmp_path):
    """``scripts/eval_model.py`` воспроизводит лист датасета по seed: и картинка,
    и эталон из ``truth/val`` - те же, что у обученной модели на валидации."""
    export_yolo_dataset(tmp_path, num_train=1, num_val=1, seed=0, tile=0, ext="jpg")
    for split, seed in (("train", 0), ("val", 1_000_000)):
        saved = cv2.imread(str(tmp_path / "sheets" / split / f"s{seed:07d}.png"))
        rgb, truth = dataset_sheet(seed, GenConfig(), 0.7)
        assert np.array_equal(saved[:, :, ::-1], rgb)
        stored = json.loads((tmp_path / "truth" / split / f"s{seed:07d}.json").read_text(encoding="utf-8"))
        assert stored["nets"] == truth["nets"]
        assert len(stored["components"]) == len(truth["components"])


def test_yolo_boxes_include_dots():
    _, truth = generate_sheet(2)
    boxes = yolo_boxes(truth)
    assert sum(1 for c, _ in boxes if c == 2) == len(truth["junctions"])
    assert len(boxes) == len(truth["components"]) + len(truth["junctions"])


def test_pipeline_on_generated_sheet(monkeypatch):
    """Сквозной прогон: шаблонный детектор + граф на случайном листе (подставной
    OCR из эталона). Нижняя граница - чтобы ловить регрессии, не цель."""
    from avers.core.validators import ProductionPipeline, SchematicOCR
    from avers.dataset.demo_schematic import evaluate

    rgb, truth = generate_sheet(8, GenConfig(only_template_classes=True))
    labels = [{"text": lb["text"], "bbox": tuple(lb["bbox"]), "confidence": 0.95} for lb in truth["labels"]]
    monkeypatch.setattr(SchematicOCR, "load", lambda self: setattr(self, "_backend", "fake") or True)
    monkeypatch.setattr(SchematicOCR, "recognize", lambda self, img: [dict(t) for t in labels])
    result = ProductionPipeline().run(rgb, "synthetic_8.png")
    ev = evaluate(result.manifest.model_dump(mode="json"), truth)
    assert ev["component_recall"] >= 0.95, ev["component_recall"]
    assert ev["net_recall"] >= 0.9, (ev["missing_nets"], ev["extra_nets"])
