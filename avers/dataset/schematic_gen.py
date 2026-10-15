"""
Генератор синтетических схем БКС v2: случайные, но правильные схемы по ЕСКД
с эталонным netlist.

Заменяет устаревший ``synthetic.py``/``generator.py`` (6 классов из 31,
искажённые пропорции, провода через центры символов). Здесь:

* все УГО библиотеки ``gost_symbols`` в натуральных пропорциях ГОСТ, повороты
  на 90° и зеркальные варианты несимметричных символов;
* провода ортогональные и идут **к выводам**; Т-соединения с точкой (ГОСТ
  2.721), углы без точки, пересечения без точки (не соединение);
* топология «шина + ветви»: источник (батарея, генератор, разъём, обрыв
  линии) → шина с последовательными элементами → ответвления вниз из 1-3
  элементов → заземление / корпус / общий обратный провод; трёх- и
  четырёхвыводные УГО (транзисторы, потенциометр, переключатель,
  трансформатор) с отводами; таблицы разъёмов ГОСТ 2.755; перемычки между
  группами, пересекающие чужие шины без соединения;
* позиционные обозначения по ГОСТ 2.710 (R1, VD2, K1.1 для контактов реле);
* отвлекающие элементы: рамка и основная надпись ГОСТ 2.104, текст
  примечаний, таблица «перечень элементов»;
* эффекты скана (``degrade``): шум, размытие, JPEG, контраст, неровный фон,
  крапины, толщина линий, наклон ±1°.

Эталон в том же формате, что у демо-схемы (``demo_schematic.evaluate``
работает без изменений), плюс ``junctions`` (точки соединения) и ``labels``
(рамки подписей - для подставного OCR и обучения OCR).

    from avers.dataset.schematic_gen import generate_sheet
    rgb, truth = generate_sheet(seed=0)

Экспорт датасета YOLO: ``export_yolo_dataset`` /
``python -m avers.dataset.cli generate-v2 ...``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from avers.dataset.gost_symbols import GOST_SYMBOLS, GOSTSymbol, draw_symbol, get_symbol_by_name

Pt = Tuple[int, int]
Box = Tuple[int, int, int, int]

# --- наборы классов ----------------------------------------------------------
# Двухвыводные, которые ставят в разрыв шины (горизонтально).
INLINE_2PIN = ("fuse", "switch_no", "switch_nc", "pushbutton_no", "resistor", "inductor",
               "diode", "zener", "capacitor", "led", "relay", "lamp")
# Двухвыводные в ветвях (вертикально).
BRANCH_2PIN = ("resistor", "capacitor", "capacitor_polar", "lamp", "led", "motor", "generator",
               "relay", "diode", "zener", "inductor", "switch_no", "switch_nc", "pushbutton_no",
               "fuse", "battery")
THREE_PIN = ("transistor_npn", "transistor_pnp", "potentiometer", "switch_changeover")
# Однополюсные «концы» цепей.
TERMINALS = ("ground", "chassis", "offpage_connector", "plug_contact", "socket_contact", "terminal")
SOURCES = ("battery", "generator")
# Пары выводов «вход/выход» у многовыводных УГО (остальные - отводы).
MAIN_PINS = {
    "transistor_npn": ("C", "E"), "transistor_pnp": ("E", "C"),
    "potentiometer": ("1", "3"), "switch_changeover": ("C", "NO"),
}
# Безымянные УГО: обозначение начинается с "_" (как в демо).
UNNAMED_PREFIX = {"ground": "_GND", "chassis": "_CH", "offpage_connector": "_OP"}
SHEET_FORMATS_MM = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420)}
FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_COMPLEX)


@dataclass
class GenConfig:
    """Параметры случайной схемы."""
    formats: Tuple[str, ...] = ("A4", "A3", "A3", "A2")
    ppm_range: Tuple[float, float] = (4.0, 8.0)          # px/мм (≈ 100-200 dpi)
    line_mm_range: Tuple[float, float] = (0.25, 0.45)    # толщина линии в мм
    max_side_px: int = 4200                              # ограничение размера листа
    frame_p: float = 0.7                                 # рамка ГОСТ 2.104
    notes_p: float = 0.5                                 # текст примечаний
    parts_table_p: float = 0.3                           # перечень элементов
    return_rail_p: float = 0.3                           # общий обратный провод у группы
    jumpers: Tuple[int, int] = (0, 2)                    # перемычки с пересечениями
    italic_p: float = 0.3
    value_label_p: float = 0.35                          # номиналы под обозначением
    connector_numbers_p: float = 0.8
    only_template_classes: bool = False                  # только классы шаблонного детектора


# =============================================================================
# Геометрия символа
# =============================================================================

def _rot_pt(pt: Pt, w: int, h: int, rotation: int) -> Pt:
    """Точка после np.rot90(canvas, rotation) (против часовой)."""
    x, y = pt
    for _ in range(rotation % 4):
        x, y = y, w - 1 - x
        w, h = h, w
    return (x, y)


@dataclass
class SymGeom:
    cls: str
    rotation: int
    mirror: bool
    w: int
    h: int
    pins: Dict[str, Pt]
    canvas: np.ndarray  # BGR, белый фон

    def edge(self, name: str) -> str:
        x, y = self.pins[name]
        d = {"left": x, "right": self.w - 1 - x, "top": y, "bottom": self.h - 1 - y}
        return min(d, key=d.get)


_GEOM_CACHE: Dict[Tuple, SymGeom] = {}


def sym_geom(cls: str, rotation: int, mirror: bool, ppm: float, line: int) -> SymGeom:
    key = (cls, rotation % 4, mirror, round(ppm, 3), line)
    g = _GEOM_CACHE.get(key)
    if g is not None:
        return g
    sym = get_symbol_by_name(cls)
    w = int(round(sym.size_mm[0] * ppm)) + 1
    h = int(round(sym.size_mm[1] * ppm)) + 1
    canvas = np.full((h, w, 3), 255, np.uint8)
    draw_symbol(canvas, sym, (0, 0, w - 1, h - 1), thickness=line)
    pins = list(sym.pins_in_bbox((0, 0, w - 1, h - 1)))
    if mirror:
        canvas = canvas[:, ::-1]
        pins = [(n, (w - 1 - x, y)) for n, (x, y) in pins]
    pins = [(n, _rot_pt(p, w, h, rotation)) for n, p in pins]
    if rotation % 4:
        canvas = np.rot90(canvas, rotation % 4)
    canvas = np.ascontiguousarray(canvas)
    g = SymGeom(cls, rotation % 4, mirror, canvas.shape[1], canvas.shape[0], dict(pins), canvas)
    if len(_GEOM_CACHE) > 4000:
        _GEOM_CACHE.clear()
    _GEOM_CACHE[key] = g
    return g


def _variants(cls: str) -> List[Tuple[int, bool]]:
    sym = get_symbol_by_name(cls)
    return [(r, m) for m in ((False, True) if sym.chiral else (False,)) for r in range(4)]


OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
DIRS = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}


def orientations(cls: str, axis: str, ppm: float, line: int,
                 pair: Optional[Tuple[str, str]] = None,
                 aligned: bool = True) -> List[Tuple[int, bool, str, str]]:
    """Ориентации, в которых пара выводов лежит на противоположных сторонах вдоль
    оси ('h': слева/справа, 'v': сверху/снизу). -> (rotation, mirror, вход, выход),
    вход - левый/верхний вывод."""
    out = []
    first = "left" if axis == "h" else "top"
    for rot, mir in _variants(cls):
        g = sym_geom(cls, rot, mir, ppm, line)
        names = list(pair) if pair else list(g.pins)
        if len(names) != 2:
            continue
        a, b = names
        ea, eb = g.edge(a), g.edge(b)
        if OPPOSITE.get(ea) != eb or ea not in (first, OPPOSITE[first]):
            continue
        pa, pb = g.pins[a], g.pins[b]
        if aligned and ((axis == "h" and pa[1] != pb[1]) or (axis == "v" and pa[0] != pb[0])):
            continue  # выводы не на одной линии: провод ломался бы у символа
        out.append((rot, mir, a, b) if ea == first else (rot, mir, b, a))
    return out


def terminal_variants(cls: str, facing: str, ppm: float, line: int) -> List[Tuple[int, bool]]:
    """Ориентации однополюсного УГО, у которых вывод смотрит на сторону ``facing``."""
    out = []
    for rot, mir in _variants(cls):
        if cls in ("ground", "chassis") and rot != 0:
            continue  # заземление рисуют выводом вверх
        g = sym_geom(cls, rot, mir, ppm, line)
        if g.edge(next(iter(g.pins))) == facing:
            out.append((rot, mir))
    return out


# =============================================================================
# Примитивы плана (рисуются только после компоновки)
# =============================================================================

@dataclass
class PSym:
    geom: SymGeom
    x0: int
    y0: int
    designator: str

    @property
    def box(self) -> Box:
        return (self.x0, self.y0, self.x0 + self.geom.w - 1, self.y0 + self.geom.h - 1)

    def pin(self, name: str) -> Pt:
        px, py = self.geom.pins[name]
        return (self.x0 + px, self.y0 + py)


@dataclass
class PTable:
    designator: str
    x0: int
    y0: int
    cell_w: int
    cell_h: int
    cells: int
    numbers: bool

    @property
    def box(self) -> Box:
        return (self.x0, self.y0, self.x0 + self.cell_w, self.y0 + self.cells * self.cell_h)

    def pin(self, k: int, side: str) -> Pt:
        cy = self.y0 + (k - 1) * self.cell_h + self.cell_h // 2
        return (self.x0 if side == "L" else self.x0 + self.cell_w, cy)


@dataclass
class PLabel:
    text: str
    org: Pt
    font: int
    scale: float
    thick: int
    box: Box
    designator: bool = True


@dataclass
class Plan:
    syms: List[PSym] = field(default_factory=list)
    tables: List[PTable] = field(default_factory=list)
    wires: List[List[Pt]] = field(default_factory=list)
    dots: List[Pt] = field(default_factory=list)
    labels: List[PLabel] = field(default_factory=list)
    links: List[Tuple[str, str]] = field(default_factory=list)   # электрические связи узлов

    def extend(self, other: "Plan") -> None:
        self.syms += other.syms
        self.tables += other.tables
        self.wires += other.wires
        self.dots += other.dots
        self.labels += other.labels
        self.links += other.links

    def boxes(self, wires: bool = True) -> List[Box]:
        out = [s.box for s in self.syms] + [t.box for t in self.tables] + [lb.box for lb in self.labels]
        if wires:
            for w in self.wires:
                for a, b in zip(w, w[1:]):
                    out.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
        out += [(x - 1, y - 1, x + 1, y + 1) for x, y in self.dots]
        return out

    def bounds(self, wires: bool = True) -> Optional[Box]:
        bs = self.boxes(wires)
        if not bs:
            return None
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))

    def shift(self, dx: int, dy: int) -> "Plan":
        for s in self.syms:
            s.x0 += dx
            s.y0 += dy
        for t in self.tables:
            t.x0 += dx
            t.y0 += dy
        self.wires = [[(x + dx, y + dy) for x, y in w] for w in self.wires]
        self.dots = [(x + dx, y + dy) for x, y in self.dots]
        for lb in self.labels:
            lb.org = (lb.org[0] + dx, lb.org[1] + dy)
            lb.box = (lb.box[0] + dx, lb.box[1] + dy, lb.box[2] + dx, lb.box[3] + dy)
        return self


def _boxes_hit(a: Box, b: Box, pad: int = 0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


# =============================================================================
# Генератор
# =============================================================================

class SchematicGenerator:
    """Одна случайная схема. Используйте ``generate_sheet``."""

    def __init__(self, seed: int = 0, config: Optional[GenConfig] = None):
        self.cfg = config or GenConfig()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        r = self.rng
        fmt = str(r.choice(self.cfg.formats))
        self.sheet_mm = SHEET_FORMATS_MM[fmt]
        ppm = float(r.uniform(*self.cfg.ppm_range))
        ppm = min(ppm, self.cfg.max_side_px / self.sheet_mm[0])
        self.ppm = ppm
        self.u = ppm  # 1 мм
        self.line = max(1, int(round(ppm * r.uniform(*self.cfg.line_mm_range))))
        self.W = int(round(self.sheet_mm[0] * ppm))
        self.H = int(round(self.sheet_mm[1] * ppm))
        self.dot_r = max(self.line + 1, int(round(0.75 * ppm)))
        self.font = int(r.choice(FONTS))
        if r.random() < self.cfg.italic_p:
            self.font |= cv2.FONT_ITALIC
        self.text_mm = float(r.uniform(2.5, 3.5))
        self.counters: Dict[str, int] = {}
        self.class_counts: Dict[str, int] = {}
        self.relays: List[str] = []
        self.plan = Plan()
        self.frame: Optional[Dict] = None
        self._node = 0
        self.rail_segments: List[Tuple[int, int, int, str]] = []  # (x0, x1, y, узел)

    # ------------------------------------------------------------- helpers
    def mm(self, a: float, b: Optional[float] = None) -> int:
        v = a if b is None else self.rng.uniform(a, b)
        return int(round(v * self.u))

    def node(self, tag: str = "n") -> str:
        self._node += 1
        return f"#{tag}{self._node}"

    def allowed(self, classes: Sequence[str]) -> List[str]:
        out = []
        for c in classes:
            sym = get_symbol_by_name(c)
            if sym is None or not sym.draw_fn:
                continue
            if self.cfg.only_template_classes and not sym.template_detect:
                continue
            out.append(c)
        return out

    def pick(self, classes: Sequence[str]) -> str:
        """Случайный класс с поправкой на баланс: редкие на листе - вероятнее."""
        classes = self.allowed(classes)
        w = np.array([1.0 / (1.0 + self.class_counts.get(c, 0)) for c in classes])
        return str(self.rng.choice(classes, p=w / w.sum()))

    def designator(self, cls: str) -> str:
        if cls in UNNAMED_PREFIX:
            p = UNNAMED_PREFIX[cls]
            n = self.counters.get(p, 0)
            self.counters[p] = n + 1
            return f"{p}{n}"
        sym = get_symbol_by_name(cls)
        prefix = sym.designator or "X"
        if cls in ("switch_no", "switch_nc", "switch_changeover") and self.relays and self.rng.random() < 0.5:
            relay = str(self.rng.choice(self.relays))
            k = f"{relay}."
            n = self.counters.get(k, 0) + 1
            self.counters[k] = n
            return f"{relay}.{n}"
        n = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = n
        des = f"{prefix}{n}"
        if cls == "relay":
            self.relays.append(des)
        return des

    def label(self, text: str, box: Box, side: str, designator: bool = True,
              size_mm: Optional[float] = None) -> PLabel:
        target = (size_mm or self.text_mm) * self.u
        (tw1, th1), _ = cv2.getTextSize("R1", self.font, 1.0, 1)
        scale = max(0.3, target / max(1, th1))
        thick = max(1, int(round(scale * 1.3))) if self.line >= 2 else 1
        (tw, th), base = cv2.getTextSize(text, self.font, scale, thick)
        pad = self.mm(1.2)
        x0, y0, x1, y1 = box
        if side == "top":
            org = ((x0 + x1) // 2 - tw // 2, y0 - pad - base)
        elif side == "bottom":
            org = ((x0 + x1) // 2 - tw // 2, y1 + pad + th)
        elif side == "right":
            org = (x1 + pad, (y0 + y1) // 2 + th // 2)
        else:
            org = (x0 - pad - tw, (y0 + y1) // 2 + th // 2)
        lb = PLabel(text, org, self.font, scale, thick,
                    (org[0], org[1] - th, org[0] + tw, org[1] + base), designator)
        return lb

    def count(self, cls: str) -> None:
        self.class_counts[cls] = self.class_counts.get(cls, 0) + 1

    def put(self, plan: Plan, cls: str, rot: int, mir: bool, pin: str, at: Pt,
            label_side: Optional[str] = None, value: bool = True) -> PSym:
        g = sym_geom(cls, rot, mir, self.ppm, self.line)
        px, py = g.pins[pin]
        des = self.designator(cls)
        s = PSym(g, at[0] - px, at[1] - py, des)
        plan.syms.append(s)
        self.count(cls)
        if label_side and not des.startswith("_"):
            lb = self.label(des, s.box, label_side)
            plan.labels.append(lb)
            if value and self.rng.random() < self.cfg.value_label_p and label_side in ("left", "right"):
                text = self.value_text(cls)
                if text:
                    vb = self.label(text, (lb.box[0], lb.box[3], lb.box[2], lb.box[3]), "bottom",
                                    designator=False, size_mm=self.text_mm * 0.85)
                    if label_side == "left":  # выровнять по правому краю обозначения
                        dx = lb.box[2] - vb.box[2]
                        vb.org = (vb.org[0] + dx, vb.org[1])
                        vb.box = (vb.box[0] + dx, vb.box[1], vb.box[2] + dx, vb.box[3])
                    else:
                        dx = lb.box[0] - vb.box[0]
                        vb.org = (vb.org[0] + dx, vb.org[1])
                        vb.box = (vb.box[0] + dx, vb.box[1], vb.box[2] + dx, vb.box[3])
                    plan.labels.append(vb)
        return s

    def value_text(self, cls: str) -> str:
        r = self.rng
        table = {
            "resistor": [f"{v}k" for v in (1, 2.2, 4.7, 10, 47)] + ["330", "1.5k"],
            "capacitor": ["0.1u", "10n", "1u", "470p"], "capacitor_polar": ["100u", "47u", "1000u"],
            "fuse": ["5A", "10A", "15A", "2A"], "lamp": ["12V 21W", "27V 5W", "A12-5"],
            "relay": ["12V", "27V"], "motor": ["12V", "27V 40W"], "inductor": ["10mH", "100uH"],
            "battery": ["12V", "24V"], "generator": ["14V", "28V"],
        }
        opts = table.get(cls)
        return str(r.choice(opts)) if opts else ""

    # ------------------------------------------------------------- terminals
    def terminal(self, plan: Plan, end: Pt, travel: str, node: str,
                 classes: Sequence[str] = TERMINALS) -> Optional[PSym]:
        """Однополюсный УГО на конце провода, приходящего в ``end`` по ``travel``."""
        facing = OPPOSITE[travel]
        cands = [c for c in self.allowed(classes) if terminal_variants(c, facing, self.ppm, self.line)]
        if not cands:
            return None
        cls = self.pick(cands)
        rot, mir = terminal_variants(cls, facing, self.ppm, self.line)[
            int(self.rng.integers(len(terminal_variants(cls, facing, self.ppm, self.line))))]
        g = sym_geom(cls, rot, mir, self.ppm, self.line)
        pin = next(iter(g.pins))
        # подпись снаружи, по ходу провода (не на соседние отводы)
        side = travel if travel in ("left", "right") else "right"
        s = self.put(plan, cls, rot, mir, pin, end, label_side=side, value=False)
        plan.links.append((node, f"{s.designator}:{pin}"))
        if cls == "offpage_connector" and self.rng.random() < 0.7:
            addr = str(self.rng.choice(["A2:5", "X3:1", "Sh.2", "L3", "A1-X2:4", "to 12"]))
            plan.labels.append(self.label(addr, s.box, side, designator=False, size_mm=self.text_mm * 0.85))
        return s

    def stub(self, plan: Plan, sym: PSym, pin: str, avoid_dir: Optional[str] = None,
             min_len: int = 0) -> None:
        """Отвод от дополнительного вывода к однополюсному УГО."""
        start = sym.pin(pin)
        edge = sym.geom.edge(pin)
        node = f"{sym.designator}:{pin}"
        dx, dy = DIRS[edge]
        s1 = max(self.mm(4, 8), min_len)
        p1 = (start[0] + dx * s1, start[1] + dy * s1)
        pts = [start, p1]
        travel = edge
        if avoid_dir and edge in (avoid_dir, OPPOSITE[avoid_dir]):
            # вывод смотрит вдоль основной оси - уходим перпендикулярно
            if edge in ("left", "right"):
                travel = "top" if start[1] < sym.y0 + sym.geom.h / 2 else "bottom"
            else:
                travel = "left" if start[0] < sym.x0 + sym.geom.w / 2 else "right"
            ex, ey = DIRS[travel]
            s2 = self.mm(5, 9)
            pts.append((p1[0] + ex * s2, p1[1] + ey * s2))
        plan.wires.append(pts)
        self.terminal(plan, pts[-1], travel, node)

    # ------------------------------------------------------------- slots
    def slot_inline(self, rail_node: str) -> Tuple[Plan, Pt, Pt, str]:
        """Элемент в разрыве шины. -> (план, вход, выход, узел после)."""
        plan = Plan()
        r = self.rng
        roll = r.random()
        if roll < 0.12 and self.allowed(["transformer"]):
            return self._inline_transformer(rail_node)
        if roll < 0.25:
            cls = self.pick([c for c in THREE_PIN if c in ("potentiometer", "switch_changeover")])
            ors = orientations(cls, "h", self.ppm, self.line, MAIN_PINS[cls], aligned=False)
        else:
            cls = self.pick(INLINE_2PIN)
            ors = orientations(cls, "h", self.ppm, self.line)
        rot, mir, a, b = ors[int(r.integers(len(ors)))]
        g = sym_geom(cls, rot, mir, self.ppm, self.line)
        extra = [n for n in g.pins if n not in (a, b)]
        # подпись с той стороны, куда не уходит отвод
        up = any(g.edge(n) == "top" or (g.edge(n) in ("left", "right") and g.pins[n][1] < g.h / 2)
                 for n in extra)
        side = "bottom" if up or (not extra and r.random() < 0.2) else "top"
        s = self.put(plan, cls, rot, mir, a, (0, 0), label_side=side)
        plan.links.append((rail_node, f"{s.designator}:{a}"))
        exit_pt = s.pin(b)
        jog = 0
        if exit_pt[1] != 0:
            # выход не на линии шины (переключатель): уступ обратно на шину
            jog = self.mm(3, 6)
            corner = (exit_pt[0] + jog, exit_pt[1])
            plan.wires.append([exit_pt, corner, (corner[0], 0)])
            exit_pt = (corner[0], 0)
        for n in extra:
            self.stub(plan, s, n, avoid_dir="right", min_len=jog + self.mm(3))
        return plan, s.pin(a), exit_pt, f"{s.designator}:{b}"

    def _inline_transformer(self, rail_node: str) -> Tuple[Plan, Pt, Pt, str]:
        plan = Plan()
        s = self.put(plan, "transformer", 0, False, "1", (0, 0), label_side="top")
        plan.links.append((rail_node, f"{s.designator}:1"))
        for n, d in (("2", -1), ("4", 1)):
            # нижние выводы обмоток: наружу по стороне вывода, затем вниз к
            # заземлению / корпусу (провод всегда отходит от вывода наружу)
            p = s.pin(n)
            corner = (p[0] + d * self.mm(6, 9), p[1])
            end = (corner[0], p[1] + self.mm(5, 9))
            plan.wires.append([p, corner, end])
            self.terminal(plan, end, "bottom", f"{s.designator}:{n}", ("ground", "chassis"))
        return plan, s.pin("1"), s.pin("3"), f"{s.designator}:3"

    def slot_table(self, rail_node: str) -> Tuple[Plan, Pt, Pt, str]:
        """Таблица разъёма в разрыве шины: шина проходит через контакт k."""
        plan = Plan()
        r = self.rng
        cells = int(r.integers(2, 9))
        cell_h = self.mm(6, 8.5)
        cell_w = self.mm(7, 10)
        k = int(r.integers(1, cells + 1))
        des = self.designator("connector_body")
        t = PTable(des, 0, -((k - 1) * cell_h + cell_h // 2), cell_w, cell_h, cells,
                   r.random() < self.cfg.connector_numbers_p)
        plan.tables.append(t)
        self.count("connector_body")
        plan.labels.append(self.label(des, t.box, "top"))
        plan.links.append((rail_node, f"{des}:{k}"))
        # прочие контакты: отводы влево/вправо к однополюсным УГО (не к соседним ячейкам)
        last = {"L": -9, "R": -9}
        for i in range(1, cells + 1):
            if i == k:
                continue
            for side in ("L", "R"):
                if side == "R" and abs(i - k) == 1:
                    continue  # рядом с шиной справа тесно
                if i - last[side] == 1:
                    continue  # соседние отводы с одной стороны: УГО налезут друг на друга
                if r.random() < 0.3:
                    last[side] = i
                    p = t.pin(i, side)
                    d = -1 if side == "L" else 1
                    end = (p[0] + d * self.mm(5, 9), p[1])
                    plan.wires.append([p, end])
                    self.terminal(plan, end, "left" if side == "L" else "right", f"{des}:{i}",
                                  ("offpage_connector", "plug_contact", "socket_contact", "terminal"))
        return plan, t.pin(k, "L"), t.pin(k, "R"), f"{des}:{k}"

    def slot_branch(self, rail_node: str, open_end: bool) -> Tuple[Plan, Pt, Optional[Tuple[Pt, str]]]:
        """Ответвление вниз от точки шины (0, 0). -> (план, точка на шине,
        открытый конец (точка, узел) для общего обратного провода)."""
        plan = Plan()
        r = self.rng
        cur, node = (0, 0), rail_node
        n = int(r.integers(1, 4))
        label_side = "left" if r.random() < 0.3 else "right"
        for i in range(n):
            gap = self.mm(5, 11) if i else self.mm(8, 14)
            if r.random() < 0.15:
                cls = self.pick([c for c in THREE_PIN if c != "switch_changeover"])
                ors = orientations(cls, "v", self.ppm, self.line, MAIN_PINS[cls])
            else:
                cls = self.pick(BRANCH_2PIN)
                ors = orientations(cls, "v", self.ppm, self.line)
            rot, mir, a, b = ors[int(r.integers(len(ors)))]
            g = sym_geom(cls, rot, mir, self.ppm, self.line)
            extra = [x for x in g.pins if x not in (a, b)]
            side = label_side
            if extra:
                side = "left" if g.edge(extra[0]) == "right" else "right"
            entry = (cur[0], cur[1] + gap)
            s = self.put(plan, cls, rot, mir, a, entry, label_side=side)
            plan.wires.append([cur, entry])
            plan.links.append((node, f"{s.designator}:{a}"))
            for x in extra:
                self.stub(plan, s, x, avoid_dir="bottom")
            cur, node = s.pin(b), f"{s.designator}:{b}"
        if open_end:
            return plan, (0, 0), (cur, node)
        end = (cur[0], cur[1] + self.mm(4, 9))
        plan.wires.append([cur, end])
        classes = ("ground", "chassis") if r.random() < 0.8 else ("offpage_connector", "plug_contact",
                                                                 "socket_contact", "terminal")
        self.terminal(plan, end, "bottom", node, classes)
        return plan, (0, 0), None

    def source(self, rail_node: str) -> Tuple[Plan, Pt]:
        """Начало группы: точка начала шины (0, 0) и то, что её питает."""
        plan = Plan()
        r = self.rng
        roll = r.random()
        if roll < 0.4:
            cls = self.pick(SOURCES)
            ors = orientations(cls, "v", self.ppm, self.line)
            rot, mir, a, b = ors[int(r.integers(len(ors)))]
            entry = (0, self.mm(8, 14))
            s = self.put(plan, cls, rot, mir, a, entry, label_side="left")
            plan.wires.append([(0, 0), entry])
            plan.links.append((rail_node, f"{s.designator}:{a}"))
            p = s.pin(b)
            end = (p[0], p[1] + self.mm(4, 9))
            plan.wires.append([p, end])
            self.terminal(plan, end, "bottom", f"{s.designator}:{b}", ("ground", "chassis"))
        elif roll < 0.7:
            self.terminal(plan, (0, 0), "left", rail_node,
                          ("offpage_connector", "plug_contact", "socket_contact", "terminal"))
        else:
            tp, _, exit_pt, node = self.slot_table(rail_node)
            tp.shift(-exit_pt[0], -exit_pt[1])
            plan.extend(tp)
            plan.links.append((node, rail_node))
        return plan, (0, 0)

    # ------------------------------------------------------------- group
    def group(self, x_limit: int) -> Optional[Plan]:
        """Одна группа «источник → шина → ветви». Строится в координатах с
        шиной на y=0, началом шины в x=0."""
        r = self.rng
        plan = Plan()
        rail = self.node("rail")
        src, start = self.source(rail)
        plan.extend(src)
        sb = src.bounds()
        prev_max = sb[2] if sb else 0
        cursor = start
        node = rail
        use_return = r.random() < self.cfg.return_rail_p
        open_ends: List[Tuple[int, Pt, str]] = []
        n_slots = 0
        last_branch: Optional[Dict] = None
        table_used = False
        spacing = self.mm(4, 8)
        while True:
            roll = r.random()
            kind = "branch" if roll < 0.5 else ("table" if roll < 0.58 and not table_used else "inline")
            if kind == "branch":
                sp, attach, open_end = self.slot_branch(node, use_return)
            elif kind == "table":
                sp, entry, exit_pt, new_node = self.slot_table(node)
            else:
                sp, entry, exit_pt, new_node = self.slot_inline(node)
            b = sp.bounds()
            dx = prev_max + spacing - b[0]
            if kind == "branch":
                dx = max(dx, cursor[0] + self.mm(6) - attach[0])
            else:
                dx = max(dx, cursor[0] + self.mm(4) - entry[0])
            sp.shift(dx, 0)
            nb = sp.bounds()
            if nb[2] > x_limit:
                self._rollback(sp)
                break
            n_slots += 1
            if kind == "branch":
                ax = attach[0] + dx
                plan.wires.append([cursor, (ax, 0)])
                self.rail_segments.append((cursor[0], ax, 0, node))
                last_branch = {"x": ax, "dot_index": len(plan.dots)}
                plan.dots.append((ax, 0))
                cursor = (ax, 0)
                if open_end:
                    open_ends.append((ax, (open_end[0][0] + dx, open_end[0][1]), open_end[1]))
            else:
                ex, ey = entry[0] + dx, entry[1]
                plan.wires.append([cursor, (ex, ey)])
                self.rail_segments.append((cursor[0], ex, 0, node))
                cursor = (exit_pt[0] + dx, exit_pt[1])
                node = new_node
                last_branch = None
                table_used = table_used or kind == "table"
            plan.extend(sp)
            prev_max = nb[2]
            if n_slots >= 3 and r.random() < 0.12:
                break
        if n_slots == 0:
            return None
        # Конец шины
        if last_branch is not None:
            # шина кончается на последнем ответвлении: угол без точки
            plan.dots.pop(last_branch["dot_index"])
        else:
            t = Plan()
            end = (cursor[0] + self.mm(5, 10), 0)
            if end[0] + self.mm(14) < x_limit:
                t.wires.append([cursor, end])
                self.rail_segments.append((cursor[0], end[0], 0, node))
                self.terminal(t, end, "right", node,
                              ("offpage_connector", "plug_contact", "socket_contact", "terminal"))
                plan.extend(t)
        if open_ends:
            self._return_rail(plan, open_ends)
        return plan

    def _rollback(self, sp: Plan) -> None:
        """Отменить счётчики обозначений/классов отброшенного слота (не критично:
        нумерация может пропустить номер, как бывает и на реальных схемах)."""
        for s in sp.syms:
            c = s.geom.cls
            self.class_counts[c] = max(0, self.class_counts.get(c, 1) - 1)
            if s.designator in self.relays:
                self.relays.remove(s.designator)
        for t in sp.tables:
            self.class_counts["connector_body"] = max(0, self.class_counts.get("connector_body", 1) - 1)

    def _return_rail(self, plan: Plan, open_ends: List[Tuple[int, Pt, str]]) -> None:
        bottoms = []
        for s in plan.syms:
            bottoms.append(s.box[3])
        y_ret = max(p[1] for _, p, _ in open_ends) + self.mm(6, 10)
        # ниже всех элементов ветвей этой группы, пересекающих диапазон x
        xs = [x for x, _, _ in open_ends]
        x0, x1 = min(xs), max(xs)
        for s in plan.syms + []:
            b = s.box
            if b[2] >= x0 and b[0] <= x1:
                y_ret = max(y_ret, b[3] + self.mm(5))
        for lb in plan.labels:
            b = lb.box
            if b[2] >= x0 and b[0] <= x1:
                y_ret = max(y_ret, b[3] + self.mm(4))
        ret = self.node("ret")
        for x, p, node in open_ends:
            plan.wires.append([p, (x, y_ret)])
            plan.links.append((node, ret))
        if len(open_ends) > 1:
            plan.wires.append([(x0, y_ret), (x1, y_ret)])
            for x in xs[1:-1]:
                plan.dots.append((x, y_ret))
            # заземление посередине между двумя соседними ветвями
            i = int(self.rng.integers(len(xs) - 1))
            gx = (sorted(xs)[i] + sorted(xs)[i + 1]) // 2
            plan.dots.append((gx, y_ret))
            start = (gx, y_ret)
        else:
            start = (x0, y_ret)
        end = (start[0], start[1] + self.mm(4, 8))
        plan.wires.append([start, end])
        self.terminal(plan, end, "bottom", ret, ("ground", "chassis"))

    # ------------------------------------------------------------- sheet
    def work_area(self) -> Tuple[Box, Optional[Box]]:
        """Рабочее поле листа и основная надпись (её не занимаем)."""
        if self.frame:
            fx0, fy0, fx1, fy1 = self.frame["inner"]
            tb = self.frame["title"]
            m = self.mm(6)
            return (fx0 + m, fy0 + m, fx1 - m, fy1 - m), tb
        m = self.mm(10)
        return (m, m, self.W - m, self.H - m), None

    def layout(self) -> None:
        r = self.rng
        if r.random() < self.cfg.frame_p:
            self.frame = self._frame_geometry()
        (ax0, ay0, ax1, ay1), tb = self.work_area()
        y = ay0
        tries = 0
        right = ax1
        while y < ay1 - self.mm(35) and tries < 16:
            tries += 1
            x0 = ax0 + self.mm(4, 20)
            g = self.group(right - x0)
            if g is None:
                break
            b = g.bounds()
            # левые отводы источника не должны вылезать за поле листа
            dx, dy = max(x0, ax0 - b[0]), y - b[1]
            g.shift(dx, dy)
            gb = g.bounds()
            hits_tb = bool(tb) and _boxes_hit(gb, tb, self.mm(3))
            if gb[3] > ay1 or gb[2] > ax1 or hits_tb:
                self._discard_group(g)
                if gb[3] > ay1 and not hits_tb:
                    break
                # над основной надписью - строим группу короче
                if tb:
                    right = min(right, tb[0] - self.mm(5))
                continue
            self._commit_rail_segments(dx, dy)
            self.plan.extend(g)
            y = gb[3] + self.mm(8, 16)
        self._jumpers()
        self._distractors()

    def _discard_group(self, g: Plan) -> None:
        self._rollback(g)
        self._pending_segments_clear()

    def _pending_segments_clear(self) -> None:
        self.rail_segments = [s for s in self.rail_segments if s[3].startswith("@")]

    def _commit_rail_segments(self, dx: int, dy: int) -> None:
        out = []
        for x0, x1, y, node in self.rail_segments:
            if node.startswith("@"):
                out.append((x0, x1, y, node))
            else:
                out.append((x0 + dx, x1 + dx, y + dy, "@" + node))
        self.rail_segments = out

    # ------------------------------------------------------------- crossings
    def _free(self, box: Box, pad: int, ignore: Iterable[Box] = ()) -> bool:
        W, H = self.W, self.H
        if box[0] < 0 or box[1] < 0 or box[2] >= W or box[3] >= H:
            return False
        ign = list(ignore)
        for b in self.plan.boxes(wires=True):
            if b in ign:
                continue
            if _boxes_hit(box, b, pad):
                return False
        if self.frame:
            if _boxes_hit(box, self.frame["title"], pad):
                return False
            fx0, fy0, fx1, fy1 = self.frame["inner"]
            if box[0] < fx0 + pad or box[1] < fy0 + pad or box[2] > fx1 - pad or box[3] > fy1 - pad:
                return False
        return True

    def _vertical_ok(self, x: int, y0: int, y1: int) -> Tuple[bool, int]:
        """Вертикальный провод x, y0..y1 не задевает УГО/подписи/параллельные
        провода и пересекает горизонтальные провода только вдали от их концов
        и точек. -> (можно, число пересечений)."""
        pad = self.mm(2.5)
        seg = (x, y0, x, y1)
        for b in [s.box for s in self.plan.syms] + [t.box for t in self.plan.tables] + \
                 [lb.box for lb in self.plan.labels]:
            if _boxes_hit(seg, b, pad):
                return False, 0
        crossings = 0
        far = self.mm(4)
        for w in self.plan.wires:
            for a, b in zip(w, w[1:]):
                if a[0] == b[0]:  # вертикальный
                    if abs(a[0] - x) <= pad and min(a[1], b[1]) <= y1 and max(a[1], b[1]) >= y0:
                        return False, 0
                    continue
                yy = a[1]
                lo, hi = min(a[0], b[0]), max(a[0], b[0])
                if not (y0 - pad <= yy <= y1 + pad) or not (lo - pad <= x <= hi + pad):
                    continue
                if not (y0 + far <= yy <= y1 - far) or not (lo + far <= x <= hi - far):
                    return False, 0
                crossings += 1
        for d in self.plan.dots:
            if abs(d[0] - x) <= far and y0 + far < d[1] < y1:
                return False, 0
        return True, crossings

    def _jumpers(self) -> None:
        r = self.rng
        n = int(r.integers(self.cfg.jumpers[0], self.cfg.jumpers[1] + 1))
        made = 0
        segs = [s for s in self.rail_segments if s[1] - s[0] > self.mm(16)]
        for _ in range(60):
            if made >= n or not segs:
                break
            x0, x1, y, node = segs[int(r.integers(len(segs)))]
            x = int(r.integers(x0 + self.mm(6), x1 - self.mm(6)))
            if any(abs(d[0] - x) < self.mm(5) and abs(d[1] - y) < self.mm(5) for d in self.plan.dots):
                continue
            length = self.mm(30, 110)
            y_end = y + length
            ok, crossings = self._vertical_ok(x, y + self.mm(3), y_end)
            if not ok or crossings == 0:
                continue
            t = Plan()
            sym = self.terminal(t, (x, y_end), "bottom", node[1:],
                                ("offpage_connector", "plug_contact", "socket_contact"))
            if sym is None or not all(self._free(b, self.mm(2)) for b in [s.box for s in t.syms] +
                                      [lb.box for lb in t.labels]):
                self._rollback(t)
                continue
            t.wires.append([(x, y), (x, y_end)])
            t.dots.append((x, y))
            self.plan.extend(t)
            made += 1

    # ------------------------------------------------------------- distractors
    def _frame_geometry(self) -> Dict:
        m5, m20 = self.mm(5), self.mm(20)
        inner = (m20, m5, self.W - m5, self.H - m5)
        tw, th = self.mm(185), self.mm(55)
        title = (inner[2] - tw, inner[3] - th, inner[2], inner[3])
        return {"inner": inner, "title": title}

    def _distractors(self) -> None:
        r = self.rng
        texts: List[Tuple[str, Pt, float]] = []
        if r.random() < self.cfg.notes_p:
            lines = ["1. Wires BPVL-0.35 unless noted.", "2. * - select on assembly.",
                     "3. Marking per GOST 2.710-81.", "4. Contacts X1:2, X1:4 are spare."]
            k = int(r.integers(2, len(lines) + 1))
            size = self.text_mm * 0.9
            (tw, th), _ = cv2.getTextSize(max(lines[:k], key=len), self.font,
                                          size * self.u / 22.0, 1)
            box_h = int(k * th * 1.8)
            for _ in range(30):
                x = int(r.integers(self.mm(25), max(self.mm(26), self.W - tw - self.mm(10))))
                y = int(r.integers(self.mm(10), max(self.mm(11), self.H - box_h - self.mm(10))))
                if self._free((x, y, x + tw, y + box_h), self.mm(4)):
                    for i, s in enumerate(lines[:k]):
                        texts.append((s, (x, y + int((i + 1) * th * 1.8)), size))
                    self.plan.labels.append(PLabel("", (x, y), self.font, 0, 1, (x, y, x + tw, y + box_h), False))
                    break
        self.notes = texts
        self.parts_table = None
        if r.random() < self.cfg.parts_table_p:
            des = [s.designator for s in self.plan.syms if not s.designator.startswith("_")][:12]
            if des:
                row_h, cw = self.mm(8), (self.mm(20), self.mm(70), self.mm(10))
                tw, th = sum(cw), row_h * (len(des) + 1)
                for _ in range(30):
                    x = int(r.integers(self.mm(25), max(self.mm(26), self.W - tw - self.mm(10))))
                    y = int(r.integers(self.mm(10), max(self.mm(11), self.H - th - self.mm(10))))
                    if self._free((x, y, x + tw, y + th), self.mm(4)):
                        self.parts_table = {"x": x, "y": y, "row_h": row_h, "cols": cw, "rows": des}
                        self.plan.labels.append(PLabel("", (x, y), self.font, 0, 1, (x, y, x + tw, y + th), False))
                        break

    # ------------------------------------------------------------- render
    def _text(self, img: np.ndarray, s: str, org: Pt, scale: float, thick: int = 1) -> None:
        """Отвлекающий текст: рисуется и попадает в эталон подписей (designator=False)."""
        cv2.putText(img, s, org, self.font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(s, self.font, scale, thick)
        self._extra_labels.append({"text": s, "bbox": [org[0], org[1] - th, org[0] + tw, org[1] + base],
                                   "designator": False})

    def render(self) -> Tuple[np.ndarray, Dict]:
        img = np.full((self.H, self.W, 3), 255, np.uint8)
        L = self.line
        self._extra_labels: List[Dict] = []
        if self.frame:
            fx0, fy0, fx1, fy1 = self.frame["inner"]
            cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (0, 0, 0), max(L, int(round(L * 1.5))))
            tx0, ty0, tx1, ty1 = self.frame["title"]
            cv2.rectangle(img, (tx0, ty0), (tx1, ty1), (0, 0, 0), max(L, int(round(L * 1.5))))
            for frac in (0.18, 0.36, 0.55):
                yy = int(ty0 + frac * (ty1 - ty0))
                cv2.line(img, (tx0, yy), (tx0 + self.mm(65), yy), (0, 0, 0), max(1, L - 1))
            cv2.line(img, (tx0 + self.mm(65), ty0), (tx0 + self.mm(65), ty1), (0, 0, 0), L)
            cv2.line(img, (tx0 + self.mm(135), ty0 + self.mm(15)), (tx1, ty0 + self.mm(15)), (0, 0, 0), L)
            cv2.line(img, (tx0 + self.mm(135), ty0), (tx0 + self.mm(135), ty1), (0, 0, 0), L)
            sc = self.text_mm * 1.6 * self.u / 22.0
            self._text(img, f"BKS.{self.seed % 1000:03d}.E3", (tx0 + self.mm(70), ty0 + self.mm(12)),
                       sc, max(1, L - 1))
            self._text(img, "Wiring diagram", (tx0 + self.mm(70), ty0 + self.mm(32)), sc * 0.8)
            self._text(img, "Sheet 1", (tx0 + self.mm(140), ty0 + self.mm(10)), sc * 0.6)
        for s in self.plan.syms:
            x0, y0 = s.x0, s.y0
            h, w = s.geom.canvas.shape[:2]
            region = img[y0:y0 + h, x0:x0 + w]
            np.minimum(region, s.geom.canvas[:region.shape[0], :region.shape[1]], out=region)
        for t in self.plan.tables:
            draw_symbol(img, get_symbol_by_name("connector_body"), t.box, thickness=L,
                        pin_count=t.cells, numbers=t.numbers)
        for w in self.plan.wires:
            for a, b in zip(w, w[1:]):
                cv2.line(img, a, b, (0, 0, 0), L)
        for d in self.plan.dots:
            cv2.circle(img, d, self.dot_r, (0, 0, 0), -1, cv2.LINE_AA)
        for lb in self.plan.labels:
            if lb.text:
                cv2.putText(img, lb.text, lb.org, lb.font, lb.scale, (0, 0, 0), lb.thick, cv2.LINE_AA)
        for s, org, size in getattr(self, "notes", []):
            self._text(img, s, org, size * self.u / 22.0)
        pt = getattr(self, "parts_table", None)
        if pt:
            x, y, rh, cols = pt["x"], pt["y"], pt["row_h"], pt["cols"]
            rows = ["Ref"] + pt["rows"]
            tw = sum(cols)
            for i in range(len(rows) + 1):
                cv2.line(img, (x, y + i * rh), (x + tw, y + i * rh), (0, 0, 0), max(1, L - 1))
            cx = x
            for c in (0,) + tuple(np.cumsum(cols)):
                cv2.line(img, (x + int(c), y), (x + int(c), y + len(rows) * rh), (0, 0, 0), max(1, L - 1))
            sc = self.text_mm * 0.85 * self.u / 22.0
            for i, text in enumerate(rows):
                self._text(img, text, (cx + self.mm(2), y + i * rh + int(rh * 0.7)), sc)
                if i:
                    self._text(img, "1", (cx + cols[0] + cols[1] + self.mm(3), y + i * rh + int(rh * 0.7)), sc)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.truth()

    def truth(self) -> Dict:
        comps = []
        for s in self.plan.syms:
            comps.append({"designator": s.designator, "class_name": s.geom.cls,
                          "class_id": get_symbol_by_name(s.geom.cls).class_id,
                          "bbox": list(s.box), "rotation": 90 * s.geom.rotation, "mirrored": s.geom.mirror,
                          "pins": {n: list(s.pin(n)) for n in s.geom.pins}})
        for t in self.plan.tables:
            comps.append({"designator": t.designator, "class_name": "connector_body", "class_id": 0,
                          "bbox": list(t.box), "rotation": 0, "mirrored": False,
                          "pins": {str(k): [list(t.pin(k, "L")), list(t.pin(k, "R"))]
                                   for k in range(1, t.cells + 1)}})
        # электрические связи -> цепи (union-find по именам узлов)
        parent: Dict[str, str] = {}

        def find(a: str) -> str:
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in self.plan.links:
            a, b = a.lstrip("@"), b.lstrip("@")
            parent[find(a)] = find(b)
        groups: Dict[str, set] = {}
        for a in list(parent):
            if a.startswith("#"):
                continue
            groups.setdefault(find(a), set()).add(a)
        nets = sorted(sorted(g) for g in groups.values() if len(g) >= 2)
        labels = [{"text": lb.text, "bbox": list(lb.box), "designator": lb.designator}
                  for lb in self.plan.labels if lb.text] + list(getattr(self, "_extra_labels", []))
        return {
            "image": {"width": self.W, "height": self.H, "px_per_mm": round(self.ppm, 4),
                      "line_px": self.line, "sheet_mm": list(self.sheet_mm), "seed": self.seed},
            "components": comps,
            "nets": nets,
            "junctions": [[x, y, self.dot_r] for x, y in self.plan.dots],
            "labels": labels,
            "notes": "Generated by avers.dataset.schematic_gen. Designators starting with '_' are "
                     "unnamed symbols (ground/chassis/off-page).",
        }


def plan_issues(plan: Plan, pad: int = 1) -> List[str]:
    """Нарушения компоновки, при которых картинка не соответствует эталону:
    наложения УГО/подписей, провод через тело УГО или подпись, провода внахлёст,
    Т-примыкание без точки (по ГОСТ это соединение, а в эталоне его нет)."""
    issues: List[str] = []
    items = [(f"sym {s.designator}", s.box) for s in plan.syms] + \
            [(f"table {t.designator}", t.box) for t in plan.tables] + \
            [(f"label {lb.text!r}", lb.box) for lb in plan.labels]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _boxes_hit(items[i][1], items[j][1], pad):
                issues.append(f"overlap {items[i][0]} / {items[j][0]}")
    segs = [(a, b) for w in plan.wires for a, b in zip(w, w[1:]) if a != b]
    bodies = [(f"sym {s.designator}", s.box) for s in plan.syms] + [(f"table {t.designator}", t.box)
                                                                    for t in plan.tables]
    for a, b in segs:
        sb = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        for name, (x0, y0, x1, y1) in bodies:
            inner = (x0 + 2, y0 + 2, x1 - 2, y1 - 2)
            if inner[0] < inner[2] and inner[1] < inner[3] and _boxes_hit(sb, inner):
                issues.append(f"wire {a}-{b} through {name}")
        for lb in plan.labels:
            if _boxes_hit(sb, lb.box, 1):
                issues.append(f"wire {a}-{b} through label {lb.text!r}")
    # внахлёст
    for i in range(len(segs)):
        a, b = segs[i]
        for j in range(i + 1, len(segs)):
            c, d = segs[j]
            if a[1] == b[1] == c[1] == d[1]:
                lo, hi = max(min(a[0], b[0]), min(c[0], d[0])), min(max(a[0], b[0]), max(c[0], d[0]))
                if hi - lo > 1:
                    issues.append(f"overlapping wires {a}-{b} / {c}-{d}")
            elif a[0] == b[0] == c[0] == d[0]:
                lo, hi = max(min(a[1], b[1]), min(c[1], d[1])), min(max(a[1], b[1]), max(c[1], d[1]))
                if hi - lo > 1:
                    issues.append(f"overlapping wires {a}-{b} / {c}-{d}")
    # Т-примыкание без точки
    dots = set(plan.dots)
    for w in plan.wires:
        for p in (w[0], w[-1]):
            if p in dots:
                continue
            for a, b in segs:
                if p in (a, b):
                    continue
                if a[1] == b[1] == p[1] and min(a[0], b[0]) < p[0] < max(a[0], b[0]) or \
                        a[0] == b[0] == p[0] and min(a[1], b[1]) < p[1] < max(a[1], b[1]):
                    issues.append(f"T without dot at {p}")
    return issues


def sheet_issues(seed: int = 0, config: Optional[GenConfig] = None) -> List[str]:
    gen = SchematicGenerator(seed, config)
    gen.layout()
    return plan_issues(gen.plan)


def generate_sheet(seed: int = 0, config: Optional[GenConfig] = None) -> Tuple[np.ndarray, Dict]:
    """Случайная схема (RGB) и её эталон. Детерминирована по ``seed``."""
    gen = SchematicGenerator(seed, config)
    gen.layout()
    return gen.render()


# =============================================================================
# Эффекты скана
# =============================================================================

def _transform_truth(truth: Dict, M: np.ndarray) -> Dict:
    t = json.loads(json.dumps(truth))

    def tp(p):
        x, y = p
        return [int(round(M[0, 0] * x + M[0, 1] * y + M[0, 2])), int(round(M[1, 0] * x + M[1, 1] * y + M[1, 2]))]

    def tb(b):
        pts = [tp((b[0], b[1])), tp((b[2], b[1])), tp((b[0], b[3])), tp((b[2], b[3]))]
        return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]

    for c in t["components"]:
        c["bbox"] = tb(c["bbox"])
        for n, v in c["pins"].items():
            c["pins"][n] = [tp(q) for q in v] if isinstance(v[0], list) else tp(v)
    t["junctions"] = [tp(j[:2]) + [j[2]] for j in t.get("junctions", [])]
    for lb in t.get("labels", []):
        lb["bbox"] = tb(lb["bbox"])
    return t


def degrade(rgb: np.ndarray, truth: Dict, seed: int = 0, strength: float = 1.0,
            rotate: bool = True) -> Tuple[np.ndarray, Dict]:
    """Эффекты скана со случайными параметрами. Геометрия эталона пересчитывается
    при наклоне листа."""
    r = np.random.default_rng(seed)
    img = rgb.astype(np.float32)
    line = int(truth["image"].get("line_px", 2))
    gray_ink = r.random() < 0.3
    if gray_ink:  # выцветшая печать / карандаш
        img = 255 - (255 - img) * r.uniform(0.55, 0.85)
    if r.random() < 0.35 * strength and line >= 2:
        k = np.ones((2, 2), np.uint8)
        u8 = img.astype(np.uint8)
        img = (cv2.dilate(u8, k) if r.random() < 0.6 else cv2.erode(u8, k)).astype(np.float32)
    if rotate and r.random() < 0.5 * strength:
        h, w = img.shape[:2]
        angle = float(r.uniform(-1.0, 1.0))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        truth = _transform_truth(truth, M)
    if r.random() < 0.4 * strength:  # неровное освещение
        h, w = img.shape[:2]
        gx = np.linspace(-1, 1, w)[None, :] * r.uniform(-1, 1)
        gy = np.linspace(-1, 1, h)[:, None] * r.uniform(-1, 1)
        img = img * (1 - 0.12 * strength * (gx + gy + 2) / 4)[..., None]
    if r.random() < 0.3 * strength:  # желтоватая бумага
        img = img * np.array([1.0, r.uniform(0.95, 1.0), r.uniform(0.85, 0.97)], np.float32)
    if r.random() < 0.5 * strength:
        img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.4, 1.0))
    if r.random() < 0.7 * strength:
        img = img + r.normal(0, r.uniform(3, 16) * strength, img.shape)
    if r.random() < 0.3 * strength:  # крапины
        h, w = img.shape[:2]
        n = int(h * w * r.uniform(2e-5, 1.5e-4))
        ys, xs = r.integers(0, h, n), r.integers(0, w, n)
        for y, x in zip(ys, xs):
            cv2.circle(img, (int(x), int(y)), int(r.integers(1, max(2, line + 1))), (60, 60, 60), -1)
    if r.random() < 0.4 * strength:
        a = r.uniform(0.75, 1.0)
        img = a * img + (1 - a) * 255 * r.uniform(0.5, 1.0)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.5 * strength:
        q = int(r.integers(45, 90))
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    return out, truth


# =============================================================================
# Экспорт YOLO
# =============================================================================

def yolo_boxes(truth: Dict, include_dots: bool = True) -> List[Tuple[int, Box]]:
    """(class_id, bbox) всех размеченных объектов листа."""
    out = [(int(c["class_id"]), tuple(c["bbox"])) for c in truth["components"]]
    if include_dots:
        for x, y, r in truth.get("junctions", []):
            out.append((2, (x - r - 1, y - r - 1, x + r + 1, y + r + 1)))
    return out


def tile_sheet(rgb: np.ndarray, boxes: List[Tuple[int, Box]], tile: int, overlap: float = 0.2,
               min_visible: float = 0.6) -> List[Tuple[np.ndarray, List[Tuple[int, Box]]]]:
    """Нарезка листа на тайлы (как при инференсе ``UltralyticsTiledDetector``).
    Объект попадает в тайл, если видна доля площади ≥ ``min_visible``."""
    from avers.stages.stage2_detection.model_detector import tile_grid

    h, w = rgb.shape[:2]
    out = []
    for x0, y0, x1, y1 in tile_grid(w, h, tile, overlap):
        crop = rgb[y0:y1, x0:x1]
        kept = []
        for cid, (bx0, by0, bx1, by1) in boxes:
            ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1 - 1), min(by1, y1 - 1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            area = (bx1 - bx0) * (by1 - by0)
            if area <= 0 or (ix1 - ix0) * (iy1 - iy0) / area < min_visible:
                continue
            kept.append((cid, (ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0)))
        out.append((crop, kept))
    return out


def _yolo_line(cid: int, box: Box, w: int, h: int) -> str:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _export_one(args) -> Dict[str, int]:
    (seed, split, out, tile, overlap, ext, degrade_p, cfg_dict, keep_empty) = args
    cfg = GenConfig(**cfg_dict) if cfg_dict else GenConfig()
    rgb, truth = generate_sheet(seed, cfg)
    r = np.random.default_rng(seed + 7_000_000)
    if r.random() < degrade_p:
        rgb, truth = degrade(rgb, truth, seed=seed, strength=float(r.uniform(0.5, 1.2)))
    out = Path(out)
    stem = f"s{seed:07d}"
    (out / "truth" / split).mkdir(parents=True, exist_ok=True)
    (out / "truth" / split / f"{stem}.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
    boxes = yolo_boxes(truth)
    pieces = tile_sheet(rgb, boxes, tile, overlap) if tile else [(rgb, boxes)]
    stats = {"images": 0, "objects": 0}
    for i, (img, objs) in enumerate(pieces):
        if not objs and r.random() > keep_empty:
            continue
        name = f"{stem}_{i:02d}" if tile else stem
        img_path = out / "images" / split / f"{name}.{ext}"
        lbl_path = out / "labels" / split / f"{name}.txt"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else []
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params)
        h, w = img.shape[:2]
        lbl_path.write_text("\n".join(_yolo_line(c, b, w, h) for c, b in objs) + ("\n" if objs else ""),
                            encoding="utf-8")
        stats["images"] += 1
        stats["objects"] += len(objs)
        for c, _ in objs:
            stats[f"c{c}"] = stats.get(f"c{c}", 0) + 1
    return stats


def export_yolo_dataset(output_dir: str | Path, num_train: int = 1000, num_val: int = 100,
                        seed: int = 0, tile: int = 1024, overlap: float = 0.2, ext: str = "png",
                        degrade_p: float = 0.7, workers: int = 1, config: Optional[GenConfig] = None,
                        keep_empty: float = 0.1, progress: bool = False) -> Path:
    """Датасет YOLO: ``images/{train,val}``, ``labels/...``, ``truth/...`` (эталоны
    листов для оценки пайплайна) и ``dataset.yaml`` со всеми 31 классом
    ``DETECTION_CLASSES`` (имена совпадают - веса подхватываются
    ``map_model_classes``)."""
    from avers.core.types import DETECTION_CLASSES

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = config.__dict__.copy() if config else None
    jobs = [(seed + i, "train", str(out), tile, overlap, ext, degrade_p, cfg_dict, keep_empty)
            for i in range(num_train)]
    jobs += [(seed + 1_000_000 + i, "val", str(out), tile, overlap, ext, degrade_p, cfg_dict, keep_empty)
             for i in range(num_val)]
    total: Dict[str, int] = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_export_one, jobs, chunksize=4)
            for i, st in enumerate(results):
                for k, v in st.items():
                    total[k] = total.get(k, 0) + v
                if progress and (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(jobs)} листов", flush=True)
    else:
        for i, job in enumerate(jobs):
            for k, v in _export_one(job).items():
                total[k] = total.get(k, 0) + v
            if progress and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} листов", flush=True)
    names = "\n".join(f"  {i}: {DETECTION_CLASSES[i]}" for i in sorted(DETECTION_CLASSES))
    (out / "dataset.yaml").write_text(
        f"# AVERS synthetic v2 (avers.dataset.schematic_gen), seed={seed}, tile={tile}\n"
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(DETECTION_CLASSES)}\nnames:\n{names}\n", encoding="utf-8")
    per_class = {DETECTION_CLASSES[int(k[1:])]: v for k, v in total.items() if k.startswith("c")}
    (out / "stats.json").write_text(json.dumps({
        "sheets": {"train": num_train, "val": num_val}, "images": total.get("images", 0),
        "objects": total.get("objects", 0), "per_class": dict(sorted(per_class.items())),
        "tile": tile, "seed": seed, "degrade_p": degrade_p,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / "dataset.yaml"


def main(argv: Optional[List[str]] = None) -> int:
    """Сохранить несколько листов для просмотра: ``python -m avers.dataset.schematic_gen out_dir [n] [seed]``."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "schematic_gen_preview")
    n = int(argv[1]) if len(argv) > 1 else 5
    seed = int(argv[2]) if len(argv) > 2 else 0
    out.mkdir(parents=True, exist_ok=True)
    for s in range(seed, seed + n):
        rgb, truth = generate_sheet(s)
        cv2.imwrite(str(out / f"sheet_{s:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (out / f"sheet_{s:05d}.netlist.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        print(f"{out / f'sheet_{s:05d}.png'}: {rgb.shape[1]}x{rgb.shape[0]}, "
              f"{len(truth['components'])} УГО, {len(truth['nets'])} цепей")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
