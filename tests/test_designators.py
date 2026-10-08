"""ГОСТ 2.710 designators from OCR: normalisation, assignment, pipeline wiring."""

import cv2
import pytest

from avers.core.designators import (
    assign_designators, connector_pin_labels, normalize_designator,
)
from avers.core.validators import ProductionPipeline, SchematicOCR
from avers.dataset.demo_schematic import build_demo, evaluate


@pytest.mark.parametrize("raw,expected", [
    ("R1", "R1"), ("VD1", "VD1"), ("K1.1", "K1.1"), ("K1,1", "K1.1"),
    ("К1.1", "K1.1"),          # Cyrillic К
    ("Х1", "X1"), ("С1", "C1"), ("ЕL2", "EL2"),
    ("R l", "R1"), ("RO", None), ("HL1", "HL1"), ("FUl", "FU1"),
    ("SA 1", "SA1"), ("GB1", "GB1"), ("XP12", "XP12"),
    ("1000", None), ("AVERS", None), ("", None), ("12V", None), ("QQQ1", None),
])
def test_normalize_designator(raw, expected):
    assert normalize_designator(raw) == expected


def _t(text, x, y, w=30, h=14, conf=0.9):
    return {"text": text, "bbox": (x, y, x + w, y + h), "confidence": conf}


def test_assign_respects_class_letter_codes_and_distance():
    comps = [
        {"bbox": (100, 100, 180, 136), "gost_class": "switch_no"},   # relay contact
        {"bbox": (300, 100, 348, 184), "gost_class": "lamp"},
        {"bbox": (500, 100, 596, 124), "gost_class": "resistor"},
    ]
    texts = [
        _t("K1.1", 125, 80),   # above the contact
        _t("EL1", 255, 135),   # left of the lamp
        _t("R1", 900, 400),    # far from everything -> not assigned
        _t("C5", 520, 80),     # capacitor code next to a resistor -> rejected
    ]
    got = assign_designators(comps, texts)
    assert {i: v["designator"] for i, v in got.items()} == {0: "K1.1", 1: "EL1"}


def test_assign_is_one_to_one():
    comps = [{"bbox": (0, 0, 60, 20), "gost_class": "resistor"},
             {"bbox": (0, 100, 60, 120), "gost_class": "resistor"}]
    texts = [_t("R1", 10, -18), _t("R1", 10, 82)]
    got = assign_designators(comps, texts)
    assert sorted(v["designator"] for v in got.values()) == ["R1"]


def test_connector_pin_numbers_from_cells():
    bbox = (100, 100, 148, 260)
    coords = {str(i + 1): [(100, 120 + 40 * i), (148, 120 + 40 * i)] for i in range(4)}
    texts = [_t("3", 118, 113, 12, 14), _t("7", 118, 153, 12, 14),
             _t("8", 118, 193, 12, 14), _t("9", 118, 233, 12, 14)]
    assert connector_pin_labels(bbox, coords, texts) == {"1": "3", "2": "7", "3": "8", "4": "9"}
    # partial read that would duplicate a name ("3" read in cell 1, cell 3 unread) -> ignored
    partial = [_t("3", 118, 113, 12, 14), _t("2", 118, 153, 12, 14)]
    assert connector_pin_labels(bbox, coords, partial) == {}


def _demo_labels(truth):
    """What an ideal OCR would return for the demo: every printed designator."""
    texts = []
    for comp in truth["components"]:
        des = comp["designator"]
        if des.startswith("_"):
            continue
        x0, y0, x1, y1 = comp["bbox"]
        (tw, th), _ = cv2.getTextSize(des, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        texts.append({"text": des, "bbox": (x0, max(0, y0 - 10 - th), x0 + tw, y0 - 10),
                      "confidence": 0.6})
    return texts


def test_pipeline_uses_ocr_designators(monkeypatch):
    image, truth = build_demo()
    labels = _demo_labels(truth)
    monkeypatch.setattr(SchematicOCR, "load", lambda self: setattr(self, "_backend", "fake") or True)
    monkeypatch.setattr(SchematicOCR, "recognize", lambda self, img: labels)

    result = ProductionPipeline().run(image, "demo.png")
    ev = evaluate(result.manifest.model_dump(mode="json"), truth)

    assert ev["designators_from_ocr"] == ev["designators_expected"], ev["designators_wrong"]
    assert ev["nets_matched"] == ev["nets_expected"]
    by_des = {c.designator: c for c in result.manifest.components}
    assert by_des["K1.1"].text_associations["gost_class"] == "switch_no"
    assert by_des["K1.1"].text_associations["designator_source"] == "ocr"
    assert by_des["K1.1"].text_associations["auto_designator"].startswith("SA")
    # no duplicated designators after renumbering the unlabelled ones
    names = [c.designator for c in result.manifest.components]
    assert len(names) == len(set(names))


def test_decimal_point_is_restored_from_pixels():
    import numpy as np
    from avers.core.designators import restore_decimal_point

    img = np.full((30, 70), 255, np.uint8)
    cv2.putText(img, "K1.1", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1, cv2.LINE_AA)
    ink = img < 128
    assert restore_decimal_point("K11", ink) == "K1.1"
    plain = np.full((30, 70), 255, np.uint8)
    cv2.putText(plain, "K11", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1, cv2.LINE_AA)
    assert restore_decimal_point("K11", plain < 128) == "K11"


def _easyocr_ready():
    """EasyOCR installed *and* its models already downloaded (no network in tests)."""
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return False
    from pathlib import Path
    models = Path.home() / ".EasyOCR" / "model"
    return models.is_dir() and any(models.glob("*.pth"))


@pytest.mark.skipif(not _easyocr_ready(), reason="EasyOCR or its models not installed")
def test_demo_with_real_easyocr():
    image, truth = build_demo()
    result = ProductionPipeline().run(image, "demo.png")
    ev = evaluate(result.manifest.model_dump(mode="json"), truth)

    assert not any("OCR недоступен" in w for w in result.warnings)
    assert ev["nets_matched"] == ev["nets_expected"], (ev["missing_nets"], ev["extra_nets"])
    assert ev["designators_from_ocr"] >= ev["designators_expected"] - 1, ev["designators_wrong"]
    assert any(c.designator == "K1.1" for c in result.manifest.components)
