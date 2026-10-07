"""
Демонстрационная схема БКС по ЕСКД с эталонным netlist.

Схема «Цепи освещения и сигнализации» (упрощённая бортовая сеть 12 В):

* GB1 - аккумуляторная батарея, C1 - помехоподавляющий конденсатор;
* FU1 - предохранитель, SA1 - выключатель цепи;
* R1 + HL1 - индикатор включения (светодиод с токоограничивающим резистором);
* X1 - разъём жгута (4 контакта, таблица по ГОСТ 2.755-87);
* K1 - реле (обмотка), VD1 - защитный диод обмотки, K1.1 - контакт реле;
* EL1, EL2 - лампы фар;
* SB1 - кнопка, M1 - электродвигатель (стеклоочиститель);
* заземления (ГОСТ 2.721-74) и соединения с корпусом.

Все УГО рисуются функциями ``avers.dataset.gost_symbols`` в натуральных
пропорциях ГОСТ (масштаб ``PX_PER_MM``), провода - ортогональные линии,
точки электрического соединения - по ГОСТ 2.721-74, пересечение без точки -
отсутствие соединения (провод X1:3 пересекает провод к R1).

Генерация::

    python -m avers.dataset.demo_schematic data/demo_bks_schematic.png

создаёт PNG и рядом ``*.netlist.json`` с эталонными компонентами и цепями,
по которым тест ``tests/test_demo_schematic.py`` проверяет весь пайплайн.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from avers.dataset.gost_symbols import GOSTSymbol, draw_symbol, get_symbol_by_name

PX_PER_MM = 6.0
LINE = 2
SIZE = (1600, 900)  # (W, H)

Pt = Tuple[int, int]


def _transform(pt: Pt, w: int, h: int, rotation: int) -> Pt:
    x, y = pt
    for _ in range(rotation % 4):
        x, y = y, w - 1 - x
        w, h = h, w
    return (x, y)


class _Sheet:
    def __init__(self, size=SIZE, ppm=PX_PER_MM):
        self.w, self.h = size
        self.ppm = ppm
        self.img = np.full((self.h, self.w, 3), 255, np.uint8)
        self.components: List[Dict] = []
        self._pins: Dict[str, Pt] = {}

    # --- symbols -----------------------------------------------------------
    def _render(self, sym: GOSTSymbol, rotation: int):
        w = int(round(sym.size_mm[0] * self.ppm)) + 1
        h = int(round(sym.size_mm[1] * self.ppm)) + 1
        canvas = np.full((h, w, 3), 255, np.uint8)
        draw_symbol(canvas, sym, (0, 0, w - 1, h - 1), thickness=LINE)
        pins = [(n, _transform(p, w, h, rotation)) for n, p in sym.pins_in_bbox((0, 0, w - 1, h - 1))]
        if rotation:
            canvas = np.ascontiguousarray(np.rot90(canvas, rotation))
        return canvas, pins

    def place(self, designator: str, class_name: str, pin: str, at: Pt,
              rotation: int = 0, label_side: str = "top") -> Dict:
        """Поставить УГО так, чтобы вывод ``pin`` оказался в точке ``at``."""
        sym = get_symbol_by_name(class_name)
        canvas, pins = self._render(sym, rotation)
        local = dict(pins)[pin]
        x0, y0 = at[0] - local[0], at[1] - local[1]
        ch, cw = canvas.shape[:2]
        region = self.img[y0:y0 + ch, x0:x0 + cw]
        np.minimum(region, canvas, out=region)
        bbox = (x0, y0, x0 + cw - 1, y0 + ch - 1)
        comp = {
            "designator": designator,
            "class_name": class_name,
            "bbox": list(bbox),
            "pins": {n: [x0 + px, y0 + py] for n, (px, py) in pins},
        }
        self.components.append(comp)
        for n, (px, py) in pins:
            self._pins[f"{designator}:{n}"] = (x0 + px, y0 + py)
        if designator and not designator.startswith("_"):
            self._label(designator, bbox, label_side)
        return comp

    def connector_table(self, designator: str, x0: int, y0: int, cells: int,
                        cell_w: int = 48, cell_h: int = 40) -> Dict:
        bbox = (x0, y0, x0 + cell_w, y0 + cells * cell_h)
        draw_symbol(self.img, get_symbol_by_name("connector_body"), bbox, thickness=LINE, pin_count=cells)
        pins = {}
        for i in range(cells):
            cy = y0 + i * cell_h + cell_h // 2
            pins[str(i + 1)] = [[x0, cy], [x0 + cell_w, cy]]
            self._pins[f"{designator}:{i + 1}:L"] = (x0, cy)
            self._pins[f"{designator}:{i + 1}:R"] = (x0 + cell_w, cy)
        comp = {"designator": designator, "class_name": "connector_body", "bbox": list(bbox), "pins": pins}
        self.components.append(comp)
        self._label(designator, bbox, "top")
        return comp

    def _label(self, text: str, bbox, side: str) -> None:
        x0, y0, x1, y1 = bbox
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if side == "top":
            org = ((x0 + x1) // 2 - tw // 2, y0 - 10)
        elif side == "right":
            org = (x1 + 10, (y0 + y1) // 2 + th // 2)
        elif side == "left":
            org = (x0 - tw - 10, (y0 + y1) // 2 + th // 2)
        else:
            org = ((x0 + x1) // 2 - tw // 2, y1 + th + 10)
        cv2.putText(self.img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # --- wires -------------------------------------------------------------
    def pin(self, ref: str) -> Pt:
        return self._pins[ref]

    def wire(self, *points: Pt) -> None:
        for a, b in zip(points, points[1:]):
            assert a[0] == b[0] or a[1] == b[1], f"non-orthogonal wire {a}->{b}"
            cv2.line(self.img, a, b, (0, 0, 0), LINE)

    def dot(self, p: Pt, r: int = 5) -> None:
        cv2.circle(self.img, p, r, (0, 0, 0), -1, cv2.LINE_AA)

    def text(self, s: str, org: Pt, scale: float = 0.7) -> None:
        cv2.putText(self.img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def build_demo() -> Tuple[np.ndarray, Dict]:
    """Нарисовать демо-схему. Возвращает (RGB изображение, эталон netlist)."""
    s = _Sheet()

    # --- Питание: GB1 (+ сверху), C1, заземление ----------------------------
    s.place("GB1", "battery", "+", (140, 294), rotation=3, label_side="left")
    s.place("C1", "capacitor", "1", (220, 260), rotation=3, label_side="right")
    s.place("_GND0", "ground", "1", (180, 560))
    s.wire(s.pin("GB1:-"), (140, 460), (220, 460))
    s.wire(s.pin("C1:2"), (220, 460))
    s.wire((180, 460), s.pin("_GND0:1"))
    s.dot((180, 460))

    # --- +12 В: FU1 -> SA1 -> X1:1 -----------------------------------------
    s.place("FU1", "fuse", "1", (282, 150))
    s.place("SA1", "switch_no", "1", (480, 150))
    s.wire(s.pin("GB1:+"), (140, 150), s.pin("FU1:1"))
    s.wire((220, 150), s.pin("C1:1"))
    s.dot((220, 150))
    s.wire(s.pin("FU1:2"), s.pin("SA1:1"))

    s.connector_table("X1", 760, 130, cells=4)
    s.wire(s.pin("SA1:2"), s.pin("X1:1:L"))

    # --- Индикатор: R1 -> HL1 -> земля --------------------------------------
    s.place("R1", "resistor", "1", (640, 282), rotation=3, label_side="right")
    s.wire((640, 150), s.pin("R1:1"))
    s.dot((640, 150))
    s.place("HL1", "led", "A", (640, 470), rotation=3, label_side="right")
    s.wire(s.pin("R1:2"), s.pin("HL1:A"))
    k = s.pin("HL1:K")
    s.place("_GND3", "ground", "1", (k[0], k[1] + 90))
    s.wire(k, s.pin("_GND3:1"))

    # --- Ветвь кнопки: FU1:2 -> X1:3 (пересекает провод R1 без соединения) ---
    s.wire((420, 150), (420, 230), s.pin("X1:3:L"))
    s.dot((420, 150))

    # --- Реле K1 с защитным диодом VD1 и контактом K1.1 ----------------------
    s.wire(s.pin("X1:1:R"), (1100, 150))
    s.dot((1000, 150))
    s.place("K1", "relay", "1", (1000, 300), label_side="left")
    s.wire((1000, 150), s.pin("K1:1"))
    s.place("VD1", "diode", "K", (1090, 310), rotation=1, label_side="right")
    s.wire((1000, 230), (1090, 230), s.pin("VD1:K"))
    s.dot((1000, 230))
    s.wire(s.pin("K1:2"), (1000, 480), (1090, 480), s.pin("VD1:A"))
    s.place("_GND1", "ground", "1", (1000, 580))
    s.wire((1000, 480), s.pin("_GND1:1"))
    s.dot((1000, 480))

    s.place("K1.1", "switch_no", "1", (1100, 150))
    s.wire(s.pin("K1.1:2"), (1450, 150))
    s.dot((1300, 150))

    # --- Лампы EL1, EL2 -> корпус -------------------------------------------
    s.place("EL1", "lamp", "1", (1300, 230), rotation=3, label_side="left")
    s.place("EL2", "lamp", "1", (1450, 230), rotation=3, label_side="right")
    s.wire((1300, 150), s.pin("EL1:1"))
    s.wire((1450, 150), s.pin("EL2:1"))
    s.wire(s.pin("EL1:2"), (1300, 420), (1450, 420), s.pin("EL2:2"))
    s.place("_CH1", "chassis", "1", (1375, 520))
    s.wire((1375, 420), s.pin("_CH1:1"))
    s.dot((1375, 420))

    # --- Стеклоочиститель: X1:3 -> SB1 -> M1 -> корпус -----------------------
    s.wire(s.pin("X1:3:R"), (900, 230), (900, 720), (960, 720))
    s.place("SB1", "pushbutton_no", "1", (960, 720))
    s.place("M1", "motor", "1", (1150, 720))
    s.wire(s.pin("SB1:2"), s.pin("M1:1"))
    s.wire(s.pin("M1:2"), (1350, 720), (1350, 800))
    s.place("_CH2", "chassis", "1", (1350, 800))

    s.text("AVERS demo: BKS lighting & signalling, 12 V (GOST 2.702 / 2.721 / 2.728 / 2.730 / 2.755)",
           (40, 870), 0.55)

    # Контакт разъёма X1 соединяет провода по обе стороны ячейки (жгут A ↔
    # жгут B), поэтому X1:1 и X1:3 объединяют левую и правую части цепей.
    nets = [
        ["C1:1", "FU1:1", "GB1:+"],
        ["C1:2", "GB1:-", "_GND0:1"],
        ["FU1:2", "SA1:1", "SB1:1", "X1:3"],
        ["K1.1:1", "K1:1", "R1:1", "SA1:2", "VD1:K", "X1:1"],
        ["HL1:A", "R1:2"],
        ["HL1:K", "_GND3:1"],
        ["K1:2", "VD1:A", "_GND1:1"],
        ["EL1:1", "EL2:1", "K1.1:2"],
        ["EL1:2", "EL2:2", "_CH1:1"],
        ["M1:1", "SB1:2"],
        ["M1:2", "_CH2:1"],
    ]
    truth = {
        "image": {"width": s.w, "height": s.h, "px_per_mm": s.ppm, "line_px": LINE},
        "components": s.components,
        "nets": [sorted(n) for n in nets],
        "notes": "Designators starting with '_' are unnamed symbols (ground/chassis). "
                 "X1:1 and X1:3 have wires on both sides of the connector cell.",
    }
    return cv2.cvtColor(s.img, cv2.COLOR_BGR2RGB), truth


# Классы, у которых имена выводов взаимозаменяемы (неполярные двухполюсники).
_SYMMETRIC = {"resistor", "capacitor", "fuse", "lamp", "relay", "switch_no", "switch_nc",
              "pushbutton_no", "motor", "generator", "inductor"}


def _bbox_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def evaluate(manifest: Dict, truth: Dict, iou: float = 0.5) -> Dict:
    """Сравнить результат пайплайна (manifest dict) с эталоном демо-схемы.

    Компонент считается найденным, если класс совпадает и IoU bbox ≥ ``iou``.
    Цепь считается восстановленной, если множество её подключений
    «обозначение:вывод» совпадает с эталонным точно (для неполярных
    двухполюсников имя вывода не учитывается)."""
    gt_comps = truth["components"]
    gt_class = {c["designator"]: c["class_name"] for c in gt_comps}
    det_to_gt: Dict[str, str] = {}
    used = set()
    for comp in manifest.get("components", []):
        cls = comp.get("text_associations", {}).get("gost_class")
        best, best_iou = None, iou
        for g in gt_comps:
            if g["designator"] in used or g["class_name"] != cls:
                continue
            v = _bbox_iou(comp["bbox"], g["bbox"])
            if v >= best_iou:
                best, best_iou = g["designator"], v
        if best:
            det_to_gt[comp["id"]] = best
            used.add(best)

    def norm(des: str, pin: str) -> str:
        return f"{des}:*" if gt_class.get(des) in _SYMMETRIC else f"{des}:{pin}"

    gt_nets = {frozenset(norm(*ref.rsplit(":", 1)) for ref in net) for net in truth["nets"]}
    det_nets = set()
    for net in manifest.get("nets", []):
        refs = frozenset(
            norm(det_to_gt[c["component_id"]], c["pin"]) if c["component_id"] in det_to_gt
            else f"?{c['component_id']}:{c['pin']}"
            for c in net.get("connections", [])
        )
        if len(refs) >= 2:
            det_nets.add(refs)
    expected_symbols = [c for c in gt_comps]
    matched_nets = gt_nets & det_nets
    detected = [c for c in manifest.get("components", []) if c.get("type") != "junction_dot"]
    return {
        "components_expected": len(expected_symbols),
        "components_found": len(used),
        "components_detected": len(detected),
        "component_recall": len(used) / max(1, len(expected_symbols)),
        "component_precision": len(det_to_gt) / max(1, len(detected)),
        "nets_expected": len(gt_nets),
        "nets_matched": len(matched_nets),
        "net_recall": len(matched_nets) / max(1, len(gt_nets)),
        "net_precision": len(matched_nets) / max(1, len(det_nets)),
        "missing_nets": sorted(sorted(n) for n in gt_nets - det_nets),
        "extra_nets": sorted(sorted(n) for n in det_nets - gt_nets),
        "designator_map": det_to_gt,
    }


def save_demo(path: str | Path) -> Tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, truth = build_demo()
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    truth_path = path.with_suffix(".netlist.json")
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, truth_path


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "data/demo_bks_schematic.png"
    img_path, truth_path = save_demo(out)
    print(f"Saved {img_path} and {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
