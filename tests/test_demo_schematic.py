"""End-to-end: the demo БКС schematic must yield its reference netlist on CPU."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from avers.core.validators import ProductionPipeline
from avers.dataset.demo_schematic import build_demo, evaluate

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def demo():
    return build_demo()


def _run(image, truth):
    result = ProductionPipeline().run(image, "demo_bks_schematic.png")
    assert result.success, result.errors
    return result, evaluate(result.manifest.model_dump(mode="json"), truth)


def test_demo_netlist_is_recovered_exactly(demo):
    image, truth = demo
    result, ev = _run(image, truth)

    assert ev["components_found"] == ev["components_expected"] == 19
    assert ev["component_precision"] == 1.0
    assert ev["nets_matched"] == ev["nets_expected"] == 11, (ev["missing_nets"], ev["extra_nets"])
    assert ev["net_precision"] == 1.0
    assert any("шаблонный детектор" in w for w in result.warnings)

    # Polarity is preserved: diode cathode on the +12 V side, LED anode to R1.
    by_id = {c.id: c for c in result.manifest.components}
    des = ev["designator_map"]
    nets = [{(des.get(c.component_id), c.pin) for c in n.connections} for n in result.manifest.nets]
    assert any(("VD1", "K") in n and ("X1", "1") in n for n in nets)
    assert any(("VD1", "A") in n and ("_GND1", "1") in n for n in nets)
    assert any(("HL1", "A") in n for n in nets)
    assert {by_id[i].designator[:2] for i in des} >= {"GB", "FU", "SA", "VD", "EL", "HL"}


def test_demo_survives_scan_degradations(demo):
    image, truth = demo
    rng = np.random.default_rng(0)
    noisy = np.clip(image.astype(int) + rng.normal(0, 18, image.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60])
    jpeg = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    for variant in (noisy, jpeg):
        _, ev = _run(variant, truth)
        assert ev["components_found"] == 19
        assert ev["nets_matched"] == 11, (ev["missing_nets"], ev["extra_nets"])


def test_demo_works_at_another_scale(demo):
    image, truth = demo
    s = 0.75
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    scaled = json.loads(json.dumps(truth))
    for comp in scaled["components"]:
        comp["bbox"] = [int(round(v * s)) for v in comp["bbox"]]
    _, ev = _run(small, scaled)
    assert ev["components_found"] == 19
    assert ev["nets_matched"] == 11, (ev["missing_nets"], ev["extra_nets"])


def test_large_sheet_with_thick_lines(demo, monkeypatch):
    """x3 sheet (4800x2700, ~8 px lines): pixel tolerances must scale with the
    stroke; crossings of thick lines must not become T-joints."""
    from avers.core.validators import SchematicOCR

    image, truth = demo
    s = 3
    big = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    scaled = json.loads(json.dumps(truth))
    labels = []
    for comp in scaled["components"]:
        comp["bbox"] = [v * s for v in comp["bbox"]]
        if not comp["designator"].startswith("_"):
            x0, y0, x1, _ = comp["bbox"]
            labels.append({"text": comp["designator"], "bbox": (x0, max(0, y0 - 80), x0 + 100, y0 - 30),
                           "confidence": 0.9})
    monkeypatch.setattr(SchematicOCR, "load", lambda self: setattr(self, "_backend", "fake") or True)
    monkeypatch.setattr(SchematicOCR, "recognize", lambda self, img: [dict(t) for t in labels])
    _, ev = _run(big, scaled)
    assert ev["components_found"] == 19
    assert ev["nets_matched"] == 11, (ev["missing_nets"], ev["extra_nets"])
    assert ev["designators_from_ocr"] == ev["designators_expected"]


def test_committed_demo_files_match_generator(demo):
    image, truth = demo
    png = DATA / "demo_bks_schematic.png"
    if not png.exists():
        pytest.skip("demo PNG not generated")
    stored = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    assert stored.shape == image.shape
    assert np.abs(stored.astype(int) - image.astype(int)).max() == 0
    stored_truth = json.loads(png.with_suffix(".netlist.json").read_text(encoding="utf-8"))
    assert stored_truth["nets"] == truth["nets"]
