"""ГОСТ УГО library and the CPU template detector built on it."""

import cv2
import numpy as np
import pytest

from avers.core.types import CLASS_TO_COMPONENT_TYPE, DETECTION_CLASSES
from avers.dataset.gost_symbols import (
    DRAW_FUNCTIONS, GOST_SYMBOLS, designator_prefix, draw_symbol, library_table,
    render_library_sheet, symbol_bbox,
)
from avers.dataset.demo_schematic import _Sheet
from avers.stages.stage2_detection.template_detector import GOSTTemplateDetector


def test_class_ids_are_stable_and_mirrored_in_core_types():
    assert [GOST_SYMBOLS[i].class_name for i in range(10)] == [
        "connector_body", "pin", "junction_dot", "ground", "shield",
        "offpage_connector", "diode", "relay", "resistor", "capacitor",
    ]
    assert {cid: s.class_name for cid, s in GOST_SYMBOLS.items()} == DETECTION_CLASSES
    assert set(CLASS_TO_COMPONENT_TYPE) == set(DETECTION_CLASSES.values())


def test_symbols_are_well_formed():
    for sym in GOST_SYMBOLS.values():
        assert sym.draw_fn in DRAW_FUNCTIONS, sym.class_name
        assert sym.gost_standard.startswith("ГОСТ"), sym.class_name
        w, h = sym.size_mm
        for name, x, y in sym.pins:
            assert 0 <= x <= w and 0 <= y <= h, (sym.class_name, name)
        names = [p[0] for p in sym.pins]
        assert len(names) == len(set(names)), sym.class_name


@pytest.mark.parametrize("cls,prefix", [
    ("resistor", "R"), ("capacitor", "C"), ("diode", "VD"), ("relay", "K"),
    ("fuse", "FU"), ("switch_no", "SA"), ("pushbutton_no", "SB"), ("lamp", "EL"),
    ("led", "HL"), ("battery", "GB"), ("motor", "M"), ("transistor_npn", "VT"),
    ("connector_body", "X"), ("plug_contact", "XP"), ("socket_contact", "XS"),
])
def test_designators_follow_gost_2710(cls, prefix):
    assert designator_prefix(cls) == prefix


def test_resistor_is_gost_rectangle_not_ansi_zigzag():
    sym = GOST_SYMBOLS[8]
    img = np.full((60, 140, 3), 255, np.uint8)
    bbox = symbol_bbox(sym, (10, 20), 7.0)  # 16x4 mm -> 112x28 px
    draw_symbol(img, sym, bbox, thickness=2)
    ink = img[:, :, 0] < 128
    x0, y0, x1, y1 = bbox
    body = ink[y0:y1 + 1, x0 + 3 * 7:x0 + 13 * 7]
    # top and bottom edges of the 10x4 mm body are straight, solid lines
    assert body[:3].any(axis=0).mean() > 0.95
    assert body[-3:].any(axis=0).mean() > 0.95
    # the inside of the rectangle is empty (no zigzag)
    assert not body[6:-6, 6:-6].any()


def test_library_table_and_sheet():
    table = library_table()
    assert len(table) == len(GOST_SYMBOLS) >= 30
    sheet = render_library_sheet()
    assert sheet.ndim == 3 and (sheet < 128).any()


def _isolated_symbol(sym, rotation):
    """A single УГО with wires leaving every pin, as on a real schematic."""
    sheet = _Sheet(size=(420, 420))
    canvas, pins = sheet._render(sym, rotation)
    ch, cw = canvas.shape[:2]
    x0, y0 = 210 - cw // 2, 210 - ch // 2
    region = sheet.img[y0:y0 + ch, x0:x0 + cw]
    np.minimum(region, canvas, out=region)
    for _, (px, py) in pins:
        _, sx, sy = min([(px, -1, 0), (cw - 1 - px, 1, 0), (py, 0, -1), (ch - 1 - py, 0, 1)])
        cv2.line(sheet.img, (x0 + px, y0 + py), (x0 + px + sx * 80, y0 + py + sy * 80), (0, 0, 0), 2)
    return cv2.cvtColor(sheet.img, cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("cls", sorted(s.class_name for s in GOST_SYMBOLS.values() if s.template_detect))
def test_every_detectable_symbol_is_recognised(cls):
    sym = next(s for s in GOST_SYMBOLS.values() if s.class_name == cls)
    detector = GOSTTemplateDetector(px_per_mm=6.0)
    for rotation in (0, 1):
        found = [d for d in detector.detect(_isolated_symbol(sym, rotation))
                 if d["category"] != "junction_dot"]
        assert [d["category"] for d in found] == [cls], (cls, rotation)
        assert len(found[0]["pins"]) == len(sym.pins)


def test_template_detector_ignores_noise_photos_and_plain_wires():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    assert GOSTTemplateDetector().detect(noise) == []

    wires = np.full((300, 400, 3), 255, np.uint8)
    cv2.line(wires, (20, 100), (380, 100), (0, 0, 0), 2)
    cv2.line(wires, (200, 100), (200, 280), (0, 0, 0), 2)
    cv2.circle(wires, (200, 100), 5, (0, 0, 0), -1)
    found = GOSTTemplateDetector().detect(wires)
    assert [d["category"] for d in found] == ["junction_dot"]
