"""
CPU-детектор УГО по шаблонам библиотеки ГОСТ (без обученных весов).

Зачем: обученные RT-DETR/YOLO веса есть не всегда (их обучение требует GPU),
но схемы, выполненные по ЕСКД, рисуются стандартными символами со
стандартными пропорциями (ГОСТ 2.721/2.728/2.730/2.755 ...). Поэтому чистые
(CAD / хорошо отсканированные) схемы можно разобрать классическим CV:

1. бинаризация, оценка толщины линии;
2. оценка масштаба чертежа (px/мм) по нескольким «опорным» символам;
3. нормированная кросс-корреляция размытых карт штрихов с шаблонами,
   отрисованными той же функцией, что и библиотека ``gost_symbols``
   (4 поворота и зеркало для несимметричных УГО);
4. проверка каждого кандидата по пикселям: шаблон должен быть покрыт
   штрихами изображения (coverage), а внутри окна не должно быть много
   «лишних» штрихов (extra) - так резистор не путается с предохранителем;
5. подавление пересечений (NMS), где при близких оценках побеждает символ,
   объясняющий больше штрихов (светодиод vs диод, кнопка vs контакт);
6. отдельно: разъёмы-таблицы (ГОСТ 2.755-87, ячейки контактов) и точки
   соединения (ГОСТ 2.721-74).

Каждая детекция содержит координаты выводов (pins) - именно они позволяют
графовому синтезу превратить провода в netlist «компонент:вывод».

Ограничения (честно): это не замена обученной модели. Рукописные и сильно
искажённые сканы, нестандартные УГО и плотный текст поверх символов
распознаются плохо; распознанные классы стоит проверять в валидаторе.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from avers.core.logger import get_logger
from avers.dataset.gost_symbols import GOST_SYMBOLS, GOSTSymbol, draw_symbol

logger = get_logger("avers.detection.templates")

# Символы, по которым оценивается масштаб чертежа (часто встречаются в БКС).
# УГО с зачернёнными областями (ГОСТ 2.755-87: штырь разъёмного соединения).
SOLID_CLASSES = frozenset({"plug_contact"})

PROBE_CLASSES = ("resistor", "fuse", "lamp", "diode", "relay", "switch_no", "capacitor", "battery")


@dataclass
class _Template:
    symbol: GOSTSymbol
    rotation: int          # число поворотов на 90° против часовой (np.rot90)
    mirror: bool
    ppm: float             # px/мм в рабочем (уменьшенном) изображении
    feat: np.ndarray       # размытая карта штрихов (float32)
    on: np.ndarray         # штрихи шаблона (bool)
    near: np.ndarray       # окрестность штрихов шаблона (bool)
    pins: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)
    core: Optional[np.ndarray] = None   # «глубокие» пиксели зачернённых областей
    core_sum: int = 0
    body: Optional[np.ndarray] = None   # штрихи без отрезков выводов
    body_sum: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.on.shape


@dataclass
class _Candidate:
    tmpl: _Template
    x: int
    y: int
    ncc: float
    coverage: float
    extra: float
    explained: int
    free_pins: int = 0

    @property
    def score(self) -> float:
        raw = 0.4 * self.ncc + 0.6 * self.coverage - 0.6 * self.extra - 0.12 * self.free_pins
        return float(np.clip(raw, 0.0, 1.0))

    @property
    def box(self) -> Tuple[int, int, int, int]:
        h, w = self.tmpl.shape
        return (self.x, self.y, self.x + w, self.y + h)


@dataclass
class _Img:
    ink: np.ndarray        # bool, штрихи
    feat: np.ndarray       # float32, размытые штрихи
    near: np.ndarray       # bool, окрестность штрихов
    integral: np.ndarray   # интегральное изображение штрихов
    stroke: float          # толщина линии, px


def _transform_point(pt: Tuple[int, int], w: int, h: int, rotation: int, mirror: bool) -> Tuple[int, int]:
    x, y = pt
    if mirror:
        x = w - 1 - x
    for _ in range(rotation % 4):
        # np.rot90 (против часовой): (x, y) -> (y, w-1-x); ширина и высота меняются
        x, y = y, w - 1 - x
        w, h = h, w
    return (x, y)


def _overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Пересечение, делённое на площадь меньшего прямоугольника."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


class GOSTTemplateDetector:
    """Шаблонный детектор УГО по библиотеке ``GOST_SYMBOLS``."""

    def __init__(
        self,
        symbols: Optional[Sequence[GOSTSymbol]] = None,
        min_score: float = 0.62,
        px_per_mm: Optional[float] = None,
        max_side: int = 1400,
        min_ncc: float = 0.45,
        min_coverage: float = 0.85,
        max_extra: float = 0.25,
        max_ring: float = 0.06,
    ):
        self.symbols = list(symbols) if symbols is not None else [
            s for s in GOST_SYMBOLS.values() if s.template_detect and s.draw_fn
        ]
        self.min_score = min_score
        self.px_per_mm = px_per_mm
        self.max_side = max_side
        self.min_ncc = min_ncc
        self.min_coverage = min_coverage
        self.max_extra = max_extra
        self.max_ring = max_ring
        self._stroke = 2.0
        self._integral = None
        self.last_info: Dict[str, object] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _binarize(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Очень светлые изображения: Otsu может выбрать порог по шуму бумаги.
        ink[gray > 200] = 0
        return ink > 0

    @staticmethod
    def _stroke_width(ink: np.ndarray) -> float:
        if not ink.any():
            return 2.0
        from skimage.morphology import skeletonize
        skel_len = int(skeletonize(ink).sum())
        if skel_len == 0:
            return 2.0
        # Площадь штрихов / длина скелета = средняя толщина линии.
        return float(np.clip(ink.sum() / skel_len, 1.0, 12.0))

    def _sigma(self, stroke: float) -> float:
        return max(1.0, 0.6 * stroke)

    def _render(self, symbol: GOSTSymbol, ppm: float, stroke: float,
                rotation: int, mirror: bool) -> Optional[_Template]:
        w_mm, h_mm = symbol.size_mm
        w = int(round(w_mm * ppm)) + 1
        h = int(round(h_mm * ppm)) + 1
        if min(w, h) < 5:
            return None
        canvas = np.full((h, w, 3), 255, np.uint8)
        draw_symbol(canvas, symbol, (0, 0, w - 1, h - 1), thickness=max(1, int(round(stroke))))
        on = canvas[:, :, 0] < 128
        pins = symbol.pins_in_bbox((0, 0, w - 1, h - 1))
        if mirror:
            on = on[:, ::-1]
        if rotation:
            on = np.rot90(on, rotation)
        on = np.ascontiguousarray(on)
        pins = [(name, _transform_point(pt, w, h, rotation, mirror)) for name, pt in pins]
        if on.sum() < 10:
            return None
        sigma = self._sigma(stroke)
        feat = cv2.GaussianBlur(on.astype(np.float32), (0, 0), sigma)
        r = max(1, int(round(stroke)) + 1)
        near = cv2.dilate(on.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        # Тело символа = штрихи без отрезков выводов: выводы совпадают с любым
        # проводом и не должны давать «уверенность» в классе.
        hh, ww = on.shape
        yy, xx = np.mgrid[0:hh, 0:ww]
        lead = np.zeros_like(on)
        rad = 2.5 * ppm
        for _, (px, py) in pins:
            lead |= (xx - px) ** 2 + (yy - py) ** 2 <= rad * rad
        body = on & ~lead
        if body.sum() < 0.25 * on.sum():
            body = on
        k = 2 * max(1, int(round(stroke))) + 3
        core = cv2.erode(on.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        core_sum = int(core.sum())
        if symbol.class_name not in SOLID_CLASSES or core_sum < 2 * stroke * stroke:
            core_sum = 0   # толстые черты/уголки линий - не зачернённая область
        return _Template(symbol, rotation, mirror, ppm, feat, on, near, pins,
                         core=core, core_sum=core_sum, body=body, body_sum=int(body.sum()))

    def _orientations(self, symbol: GOSTSymbol, ppm: float, stroke: float) -> List[_Template]:
        """Все различимые ориентации символа (дубликаты по симметрии отброшены)."""
        out: List[_Template] = []
        for mirror in ((False, True) if symbol.chiral else (False,)):
            for rot in range(4):
                t = self._render(symbol, ppm, stroke, rot, mirror)
                if t is None:
                    continue
                dup = False
                for o in out:
                    if o.on.shape == t.on.shape:
                        inter = np.logical_and(o.on, t.on).sum()
                        union = np.logical_or(o.on, t.on).sum()
                        if union and inter / union > 0.8:
                            dup = True
                            break
                if not dup:
                    out.append(t)
        return out

    def _prepare(self, ink: np.ndarray) -> "_Img":
        stroke = self._stroke_width(ink)
        feat = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), self._sigma(stroke))
        r = max(1, int(round(stroke * 0.75)))
        near = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        integral = cv2.integral(ink.astype(np.uint8)).astype(np.float64)
        return _Img(ink=ink, feat=feat, near=near, integral=integral, stroke=stroke)

    def _match_all(self, img: "_Img", templates: List[_Template], max_peaks: int = 300) -> List[_Candidate]:
        """Сопоставить много шаблонов параллельно (OpenCV отпускает GIL)."""
        if not templates:
            return []
        from concurrent.futures import ThreadPoolExecutor
        import os
        workers = max(1, min(8, (os.cpu_count() or 2)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(lambda t: self._match(img, t, max_peaks), templates))
        return [c for part in parts for c in part]

    def _match(self, img: "_Img", tmpl: _Template, max_peaks: int = 300) -> List[_Candidate]:
        feat, ink, ink_near = img.feat, img.ink, img.near
        th, tw = tmpl.shape
        H, W = feat.shape
        if th >= H or tw >= W:
            return []
        res = cv2.matchTemplate(feat, tmpl.feat, cv2.TM_CCOEFF_NORMED)
        res[~np.isfinite(res)] = 0.0
        # На пустых (почти постоянных) окнах нормированная корреляция
        # вырождается в ±1 - требуем, чтобы в окне было достаточно штрихов.
        on_sum = float(tmpl.on.sum())
        integral = img.integral
        rh, rw = res.shape
        win = (integral[th:th + rh, tw:tw + rw] - integral[0:rh, tw:tw + rw]
               - integral[th:th + rh, 0:rw] + integral[0:rh, 0:rw])
        res[win < 0.3 * on_sum] = 0.0
        k = max(3, (min(th, tw) // 2) | 1)
        local_max = cv2.dilate(res, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        ys, xs = np.where((res >= self.min_ncc) & (res >= local_max - 1e-6))
        if len(xs) == 0:
            return []
        order = np.argsort(-res[ys, xs])[:max_peaks]
        cands = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            win_near = ink_near[y:y + th, x:x + tw]
            win_ink = ink[y:y + th, x:x + tw]
            matched = int(np.logical_and(tmpl.on, win_near).sum())
            body_sum = float(tmpl.body_sum or on_sum)
            coverage = float(np.logical_and(tmpl.body, win_near).sum()) / body_sum
            if coverage < self.min_coverage:
                continue
            extra = float(np.logical_and(win_ink, ~tmpl.near).sum()) / body_sum
            if extra > self.max_extra:
                continue
            if tmpl.core_sum:
                # зачернённые области (штырь, точка) должны быть сплошными
                solid = np.logical_and(tmpl.core, win_ink).sum() / tmpl.core_sum
                if solid < 0.85:
                    continue
            if self._ring_ink(img, tmpl, x, y) > self.max_ring:
                continue
            if self._is_plain_wire(img, tmpl, win_ink):
                continue
            connected = self._pins_connected(img, tmpl, x, y)
            n_free = connected.count(False)
            if connected and (n_free == len(connected) or (len(connected) == 1 and n_free)):
                continue  # к символу не подходит ни один провод - вероятно, текст
            cands.append(_Candidate(tmpl, x, y, float(res[y, x]), coverage, extra, matched, n_free))
        return cands

    @staticmethod
    def _is_plain_wire(img: "_Img", tmpl: _Template, win_ink: np.ndarray) -> bool:
        """True, если штрихи в окне - это просто провод вдоль оси выводов.

        На мелком масштабе тонкие детали (дуги катушки, клин штыря) лежат в
        паре пикселей от оси и «покрываются» обычной линией; реальный символ
        всегда имеет заметную долю штрихов вне оси."""
        if len(tmpl.pins) not in (1, 2):
            return False
        th, tw = tmpl.shape
        (_, a), = tmpl.pins[:1]
        if len(tmpl.pins) == 2:
            b = tmpl.pins[1][1]
        else:
            b = (tw - 1 - a[0], th - 1 - a[1])
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = float(np.hypot(dx, dy))
        if norm < 1:
            return False
        ys, xs = np.nonzero(win_ink)
        if len(xs) == 0:
            return True
        d = np.abs((xs - a[0]) * dy - (ys - a[1]) * dx) / norm
        return float((d <= 0.75 * img.stroke + 1.0).mean()) > 0.85

    def _pins_connected(self, img: "_Img", tmpl: _Template, x: int, y: int) -> List[bool]:
        """Для каждого вывода: отходит ли от него наружу непрерывная линия."""
        th, tw = tmpl.shape
        H, W = img.ink.shape
        m = max(4, int(round(1.2 * tmpl.ppm)))
        out = []
        for _, (px, py) in tmpl.pins:
            gx, gy = x + px, y + py
            if px <= 1:
                xs, ys, horiz = np.arange(x - m, x), None, True
            elif px >= tw - 2:
                xs, ys, horiz = np.arange(x + tw, x + tw + m), None, True
            elif py <= 1:
                xs, ys, horiz = None, np.arange(y - m, y), False
            elif py >= th - 2:
                xs, ys, horiz = None, np.arange(y + th, y + th + m), False
            else:
                out.append(True)
                continue
            hits = 0
            total = 0
            if horiz:
                for xx in xs:
                    if 0 <= xx < W:
                        total += 1
                        hits += bool(img.ink[max(0, gy - 2):min(H, gy + 3), xx].any())
            else:
                for yy in ys:
                    if 0 <= yy < H:
                        total += 1
                        hits += bool(img.ink[yy, max(0, gx - 2):min(W, gx + 3)].any())
            out.append(total > 0 and hits / total >= 0.8)
        return out

    def _ring_ink(self, img: "_Img", tmpl: _Template, x: int, y: int) -> float:
        """Доля штрихов в кольце вокруг окна, не объяснимых проводами от выводов.

        Настоящий УГО изолирован: снаружи к нему подходят только провода в
        точках выводов. Если вокруг окна есть другие штрихи, найденный
        «символ» - фрагмент более крупной фигуры (угол провода, часть другого
        УГО, буква), и кандидат отбрасывается."""
        th, tw = tmpl.shape
        ink = img.ink
        H, W = ink.shape
        m = max(3, int(round(1.0 * tmpl.ppm)))
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(W, x + tw + m), min(H, y + th + m)
        region = ink[y0:y1, x0:x1].copy()
        # само окно шаблона не учитываем
        region[y - y0:y - y0 + th, x - x0:x - x0 + tw] = False
        # провода, выходящие из выводов, разрешены
        a = int(round(img.stroke)) + 2
        for _, (px, py) in tmpl.pins:
            gx, gy = x + px - x0, y + py - y0
            if px <= 1 or px >= tw - 2:      # вывод на левой/правой стороне
                region &= ~self._side_mask(region.shape, gx, gy, a, horizontal=True, left=(px <= 1))
            if py <= 1 or py >= th - 2:      # вывод на верхней/нижней стороне
                region &= ~self._side_mask(region.shape, gx, gy, a, horizontal=False, left=(py <= 1))
        perimeter = 2.0 * (tw + th)
        return float(region.sum()) / max(1.0, perimeter * img.stroke)

    @staticmethod
    def _side_mask(shape, gx, gy, a, horizontal: bool, left: bool):
        mask = np.zeros(shape, bool)
        h, w = shape
        if horizontal:
            ys = slice(max(0, gy - a), min(h, gy + a + 1))
            xs = slice(0, max(0, gx + 1)) if left else slice(max(0, gx), w)
        else:
            xs = slice(max(0, gx - a), min(w, gx + a + 1))
            ys = slice(0, max(0, gy + 1)) if left else slice(max(0, gy), h)
        mask[ys, xs] = True
        return mask

    # ------------------------------------------------------------- pipeline
    @staticmethod
    def _ppm_range(stroke: float) -> Tuple[float, float]:
        """Допустимый масштаб px/мм по толщине линии: по ГОСТ 2.303/2.721 линии
        УГО и связи имеют толщину ~0,2-0,6 мм. Берём с запасом 0,15-0,8 мм,
        иначе крошечные «символы» размером в пару толщин линии находятся везде."""
        return max(1.5, stroke / 0.8), min(40.0, max(3.0, stroke / 0.15))

    def _estimate_scale(self, img: "_Img") -> Optional[float]:
        """Масштаб (px/мм в координатах img): тот, при котором больше всего
        опорных символов находится уверенно."""
        probes = [s for s in self.symbols if s.class_name in PROBE_CLASSES] or self.symbols
        lo, hi = self._ppm_range(img.stroke)
        scales = []
        ppm = lo
        while ppm <= hi:
            scales.append(ppm)
            ppm *= 1.12
        templates = [t for ppm in scales for sym in probes for rot in (0, 1)
                     for t in [self._render(sym, ppm, img.stroke, rot, False)] if t is not None]
        cands = self._match_all(img, templates, max_peaks=30)
        per_scale: Dict[float, Dict[str, float]] = {}
        for c in cands:
            if c.score < self.min_score:
                continue
            d = per_scale.setdefault(c.tmpl.ppm, {})
            d[c.tmpl.symbol.class_name] = max(d.get(c.tmpl.symbol.class_name, 0.0), c.score)
        best_ppm, best_val = None, 0.0
        for ppm, d in per_scale.items():
            val = sum(d.values())
            if val > best_val + 1e-9:
                best_ppm, best_val = ppm, val
        return best_ppm

    def _nms(self, cands: List[_Candidate]) -> List[_Candidate]:
        cands = sorted(cands, key=lambda c: -c.score)
        kept: List[_Candidate] = []
        for c in cands:
            drop = False
            for i, k in enumerate(kept):
                if _overlap_ratio(c.box, k.box) < 0.45:
                    continue
                # Близкие оценки: побеждает символ, объясняющий больше штрихов
                # (светодиод > диод, кнопка > контакт, батарея > конденсатор).
                if c.explained >= 1.3 * k.explained and c.score >= k.score - 0.1:
                    kept[i] = c
                drop = True
                break
            if not drop:
                kept.append(c)
        # После замен могли появиться пересечения - повторяем до сходимости.
        changed = True
        while changed:
            changed = False
            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    if _overlap_ratio(kept[i].box, kept[j].box) >= 0.45:
                        a, b = kept[i], kept[j]
                        loser = j if (a.explained, a.score) >= (b.explained, b.score) else i
                        kept.pop(loser)
                        changed = True
                        break
                if changed:
                    break
        return kept

    def _detect_connector_tables(self, ink: np.ndarray, stroke: float,
                                 occupied: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Разъём ГОСТ 2.755-87 в виде таблицы: стопка ≥2 одинаковых ячеек."""
        free = (~ink).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        H, W = ink.shape
        min_side = max(8, int(3 * stroke))
        cells = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                continue  # фон
            if w < min_side or h < min_side or w > W * 0.5 or h > H * 0.5:
                continue
            if area < 0.7 * w * h:
                continue  # не прямоугольник
            box = (int(x), int(y), int(x + w), int(y + h))
            if any(_overlap_ratio(box, o) > 0.5 for o in occupied):
                continue
            cells.append(box)
        tol = max(3, int(round(stroke)) + 2)
        gap = int(round(2 * stroke)) + 4
        used = set()
        tables = []
        for vertical in (True, False):
            key = (lambda b: (b[0], b[1])) if vertical else (lambda b: (b[1], b[0]))
            for i, c in sorted(enumerate(cells), key=lambda t: key(t[1])):
                if i in used:
                    continue
                stack = [i]
                cur = c
                while True:
                    nxt = None
                    for j, d in enumerate(cells):
                        if j in used or j in stack:
                            continue
                        if vertical:
                            ok = (abs(d[0] - cur[0]) <= tol and abs(d[2] - cur[2]) <= tol
                                  and 0 <= d[1] - cur[3] <= gap)
                        else:
                            ok = (abs(d[1] - cur[1]) <= tol and abs(d[3] - cur[3]) <= tol
                                  and 0 <= d[0] - cur[2] <= gap)
                        if ok:
                            nxt = j
                            break
                    if nxt is None:
                        break
                    stack.append(nxt)
                    cur = cells[nxt]
                if len(stack) < 2:
                    continue
                boxes = [cells[j] for j in stack]
                sizes = [(b[3] - b[1]) if vertical else (b[2] - b[0]) for b in boxes]
                if max(sizes) > 1.6 * min(sizes):
                    continue  # ячейки разъёма одинаковые
                if self._divider_continues(ink, boxes, vertical, stroke):
                    continue  # «разделитель» - провод, проходящий через прямоугольник
                used.update(stack)
                pad = int(np.ceil(stroke))
                x0 = max(0, min(b[0] for b in boxes) - pad)
                y0 = max(0, min(b[1] for b in boxes) - pad)
                x1 = min(W - 1, max(b[2] for b in boxes) + pad)
                y1 = min(H - 1, max(b[3] for b in boxes) + pad)
                pins = []
                for num, b in enumerate(boxes, start=1):
                    if vertical:
                        cy = (b[1] + b[3]) // 2
                        pins += [(str(num), (x0, cy)), (str(num), (x1, cy))]
                    else:
                        cx = (b[0] + b[2]) // 2
                        pins += [(str(num), (cx, y0)), (str(num), (cx, y1))]
                tables.append({"box": (x0, y0, x1, y1), "pins": pins, "cells": len(boxes)})
        return tables

    @staticmethod
    def _divider_continues(ink: np.ndarray, boxes, vertical: bool, stroke: float) -> bool:
        """True, если линия между ячейками продолжается за контур таблицы."""
        H, W = ink.shape
        off = int(np.ceil(stroke)) + 2
        span = off + int(np.ceil(2 * stroke)) + 2
        for a, b in zip(boxes, boxes[1:]):
            if vertical:
                yb = (a[3] + b[1]) // 2
                x_left, x_right = min(c[0] for c in boxes), max(c[2] for c in boxes)
                rows = slice(max(0, yb - 1), min(H, yb + 2))
                left = ink[rows, max(0, x_left - span):max(0, x_left - off)]
                right = ink[rows, min(W, x_right + off):min(W, x_right + span)]
            else:
                xb = (a[2] + b[0]) // 2
                y_top, y_bot = min(c[1] for c in boxes), max(c[3] for c in boxes)
                cols = slice(max(0, xb - 1), min(W, xb + 2))
                left = ink[max(0, y_top - span):max(0, y_top - off), cols]
                right = ink[min(H, y_bot + off):min(H, y_bot + span), cols]
            for side in (left, right):
                if side.size and side.any(axis=0 if vertical else 1).mean() > 0.6:
                    return True
        return False

    def _detect_junction_dots(self, ink: np.ndarray, stroke: float,
                              occupied: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        # Точка соединения - зачернённый круг заметно толще линии: ищем
        # «ядра», где расстояние до фона больше, чем у любого штриха/угла.
        dist = cv2.distanceTransform(np.pad(ink, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        half = stroke / 2.0
        # Пересечение «+» двух линий толщины w даёт ~w/√2 ≈ 1.41·half, угол/Т
        # ещё меньше; точка по ГОСТ - диаметр ≥ ~3w.
        thr = max(2.2 * half, half + 1.6)
        core = (dist >= thr).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        dots = []
        max_core = 4 * stroke + 2
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w > max_core or h > max_core or max(w, h) > 2.0 * max(1, min(w, h)):
                continue
            pad = int(np.ceil(thr))
            box = (int(x - pad), int(y - pad), int(x + w + pad), int(y + h + pad))
            if any(_overlap_ratio(box, o) > 0.3 for o in occupied):
                continue
            # точка соединения лежит на проводах: ≥2 направлений с линией
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            r0 = (box[2] - box[0]) // 2 + 1
            ln = max(4, r0)
            H, W = ink.shape
            arms = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                hits = 0
                for s in range(r0, r0 + ln):
                    px, py = cx + dx * s, cy + dy * s
                    if 0 <= px < W and 0 <= py < H:
                        hits += bool(ink[max(0, py - 1):py + 2, max(0, px - 1):px + 2].any())
                arms += hits >= 0.8 * ln
            if arms < 2:
                continue
            dots.append(box)
        return dots

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Найти УГО. Возвращает словари в формате SlicedDetector.detect()."""
        self.last_info = {}
        if image is None or image.size == 0 or min(image.shape[:2]) < 16:
            return []
        ink_full = self._binarize(image)
        density = float(ink_full.mean())
        self.last_info["ink_density"] = density
        if density == 0.0 or density > 0.3:
            # Не похоже на штриховой чертёж (фото/шум) - не выдумываем символы.
            self.last_info["skipped"] = "not a line drawing"
            return []

        H0, W0 = ink_full.shape
        f = min(1.0, self.max_side / float(max(H0, W0)))
        if f < 1.0:
            ink = cv2.resize(ink_full.astype(np.uint8) * 255, (int(W0 * f), int(H0 * f)),
                             interpolation=cv2.INTER_AREA) >= 64
        else:
            ink = ink_full
        img = self._prepare(ink)
        stroke = img.stroke

        if self.px_per_mm:
            ppm = self.px_per_mm * f
        else:
            # Грубая оценка масштаба на вдвое уменьшенной копии (в 4 раза быстрее).
            h, w = ink.shape
            if max(h, w) > 700:
                small = cv2.resize(ink.astype(np.uint8) * 255, (w // 2, h // 2),
                                   interpolation=cv2.INTER_AREA) >= 64
                coarse = self._estimate_scale(self._prepare(small))
                ppm = coarse * 2 if coarse else self._estimate_scale(img)
            else:
                ppm = self._estimate_scale(img)
        self.last_info.update({"stroke_px": stroke / f, "scale_factor": f})
        cands: List[_Candidate] = []
        if ppm:
            self.last_info["px_per_mm"] = ppm / f
            templates = [t for mult in (0.9, 0.97, 1.04, 1.11) for sym in self.symbols
                         for t in self._orientations(sym, ppm * mult, stroke)]
            cands = [c for c in self._match_all(img, templates) if c.score >= self.min_score]
        kept = self._nms(cands)

        def up(v: float) -> int:
            return int(round(v / f))

        detections: List[Dict] = []
        occupied_small = []
        for c in kept:
            x0, y0, x1, y1 = c.box
            occupied_small.append(c.box)
            box = (up(x0), up(y0), min(W0 - 1, up(x1)), min(H0 - 1, up(y1)))
            pins = [{"name": name, "coord": (min(W0 - 1, max(0, up(c.x + px))),
                                             min(H0 - 1, max(0, up(c.y + py))))}
                    for name, (px, py) in c.tmpl.pins]
            detections.append({
                "bbox": box,
                "confidence": round(c.score, 3),
                "category": c.tmpl.symbol.class_name,
                "category_id": c.tmpl.symbol.class_id,
                "pins": pins,
                "rotation": 90 * c.tmpl.rotation,
                "mirrored": c.tmpl.mirror,
                "source": "gost_templates",
            })

        for table in self._detect_connector_tables(ink, stroke, occupied_small):
            x0, y0, x1, y1 = table["box"]
            occupied_small.append(table["box"])
            detections.append({
                "bbox": (up(x0), up(y0), min(W0 - 1, up(x1)), min(H0 - 1, up(y1))),
                "confidence": 0.8,
                "category": "connector_body",
                "category_id": 0,
                "pins": [{"name": n, "coord": (min(W0 - 1, up(px)), min(H0 - 1, up(py)))}
                         for n, (px, py) in table["pins"]],
                "rotation": 0,
                "mirrored": False,
                "source": "gost_tables",
            })

        for x0, y0, x1, y1 in self._detect_junction_dots(ink, stroke, occupied_small):
            detections.append({
                "bbox": (up(x0), up(y0), min(W0 - 1, up(x1)), min(H0 - 1, up(y1))),
                "confidence": 0.75,
                "category": "junction_dot",
                "category_id": 2,
                "pins": [],
                "rotation": 0,
                "mirrored": False,
                "source": "gost_dots",
                "exclude_from_wires": False,
            })

        self.last_info["detections"] = len(detections)
        logger.info(
            f"GOST template detector: {len(detections)} objects "
            f"(scale={self.last_info.get('px_per_mm', 'n/a')} px/mm, stroke={stroke / f:.1f}px)"
        )
        return detections

    # ------------------------------------------------------------ hybrid mode
    def refine(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Гибридный режим: уточнить детекции обученной модели по геометрии ГОСТ.

        Модель отвечает за «где и что» (класс + рамка), шаблон того же класса
        ищется только в окрестности рамки и даёт поворот, зеркальность, точную
        рамку и координаты выводов. Пороги мягче, чем в detect(): класс уже
        подтверждён моделью. Если шаблон не нашёлся, выводы раскладываются по
        геометрии УГО из рамки (поворот по соотношению сторон).
        """
        if not detections:
            return []
        ink_full = self._binarize(image)
        H, W = ink_full.shape
        stroke = self._stroke_width(ink_full) if ink_full.any() else 2.0
        relaxed = GOSTTemplateDetector(
            symbols=self.symbols, min_ncc=0.3, min_coverage=0.7, max_extra=0.6, max_ring=0.5,
        )
        by_name = {s.class_name: s for s in GOST_SYMBOLS.values()}
        out: List[Dict] = []
        for det in detections:
            det = dict(det)
            name = det["category"]
            sym = by_name.get(name)
            x0, y0, x1, y1 = (int(v) for v in det["bbox"])
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            pad = int(max(10, 0.35 * max(bw, bh)))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            crop = ink_full[cy0:cy1, cx0:cx1]

            if name == "junction_dot":
                det.setdefault("pins", [])
                det.setdefault("exclude_from_wires", False)
                det["pins_source"] = "dot"
                out.append(det)
                continue
            if name == "connector_body":
                tables = self._detect_connector_tables(crop, stroke, []) if crop.size else []
                if tables:
                    t = max(tables, key=lambda t: t["cells"])
                    tx0, ty0, tx1, ty1 = t["box"]
                    det["bbox"] = (tx0 + cx0, ty0 + cy0, tx1 + cx0, ty1 + cy0)
                    det["pins"] = [{"name": n, "coord": (px + cx0, py + cy0)} for n, (px, py) in t["pins"]]
                    det["pins_source"] = "table"
                else:
                    det.setdefault("pins", [])
                    det["pins_source"] = "none"
                out.append(det)
                continue
            if sym is None or not sym.draw_fn or not sym.pins:
                det.setdefault("pins", [])
                det["pins_source"] = "none"
                out.append(det)
                continue

            best = None
            if crop.size and crop.any():
                img = relaxed._prepare(crop)
                img.stroke = stroke
                w_mm, h_mm = sym.size_mm
                templates = []
                for mirror in ((False, True) if sym.chiral else (False,)):
                    for rot in range(4):
                        tw_mm, th_mm = (w_mm, h_mm) if rot % 2 == 0 else (h_mm, w_mm)
                        base = float(np.sqrt((bw / tw_mm) * (bh / th_mm)))
                        for mult in (0.85, 0.93, 1.0, 1.08, 1.17):
                            t = relaxed._render(sym, base * mult, stroke, rot, mirror)
                            if t is not None and t.shape[0] < crop.shape[0] and t.shape[1] < crop.shape[1]:
                                templates.append(t)
                cands = relaxed._match_all(img, templates, max_peaks=20)
                if cands:
                    best = max(cands, key=lambda c: c.score)
            if best is not None:
                bx0, by0 = best.x + cx0, best.y + cy0
                th, tw = best.tmpl.shape
                det["bbox"] = (bx0, by0, min(W - 1, bx0 + tw), min(H - 1, by0 + th))
                det["pins"] = [{"name": n, "coord": (min(W - 1, bx0 + px), min(H - 1, by0 + py))}
                               for n, (px, py) in best.tmpl.pins]
                det["rotation"] = 90 * best.tmpl.rotation
                det["mirrored"] = best.tmpl.mirror
                det["template_score"] = round(best.score, 3)
                det["pins_source"] = "template"
            else:
                # Геометрия ГОСТ без проверки по пикселям: поворот 0/90 по рамке.
                w_mm, h_mm = sym.size_mm
                vertical = (bh > bw) != (h_mm > w_mm)
                pins = []
                for n, px_mm, py_mm in sym.pins:
                    fx, fy = px_mm / w_mm, py_mm / h_mm
                    if vertical:  # поворот на 90° против часовой: (x, y) -> (y, 1 - x)
                        fx, fy = fy, 1.0 - fx
                    pins.append({"name": n, "coord": (int(round(x0 + fx * bw)), int(round(y0 + fy * bh)))})
                det["pins"] = pins
                det["rotation"] = 90 if vertical else 0
                det["pins_source"] = "geometry"
            out.append(det)
        return out
