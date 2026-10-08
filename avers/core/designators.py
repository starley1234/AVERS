"""
Позиционные обозначения (ГОСТ 2.710-81) из OCR и их привязка к УГО.

* ``normalize_designator`` - чистит типичные ошибки OCR: кириллические буквы,
  похожие на латинские (К→K, Х→X, С→C ...), O/0 и I/1 в номере, пробелы,
  запятая вместо точки (K1,1 → K1.1). Возвращает ``None``, если строка не
  похожа на обозначение.
* ``assign_designators`` - сопоставляет обозначения компонентам: подпись
  ставится ближайшему УГО, для которого её буквенный код допустим (K1.1 можно
  присвоить контакту реле, но не лампе), один к одному, жадно по расстоянию.
* ``assign_connector_pins`` - номера контактов разъёма из цифр в его ячейках.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Кириллица, внешне совпадающая с латиницей (буквенные коды ГОСТ 2.710 - латиница).
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "Г": "G", "Л": "L",
    "а": "A", "в": "B", "с": "C", "е": "E", "н": "H", "к": "K", "м": "M",
    "о": "O", "р": "P", "т": "T", "х": "X", "у": "Y",
})

# Буквенные коды по ГОСТ 2.710-81 (табл. 1 и 2), встречающиеся на схемах БКС.
_KNOWN_PREFIXES = {
    "A", "B", "BA", "BF", "BK", "BL", "BP", "BQ", "BR", "BV", "C", "D", "DA", "DD", "E", "EK",
    "EL", "F", "FA", "FP", "FU", "FV", "G", "GB", "GC", "GE", "H", "HA", "HG", "HL", "K", "KA",
    "KH", "KK", "KM", "KT", "KV", "L", "LL", "M", "P", "PA", "PC", "PF", "PI", "PK", "PR", "PS",
    "PT", "PV", "PW", "Q", "QF", "QK", "QS", "R", "RK", "RP", "RS", "RU", "S", "SA", "SB", "SF",
    "SL", "SP", "SQ", "SR", "ST", "T", "TA", "TS", "TV", "U", "V", "VD", "VL", "VS", "VT",
    "W", "WA", "WE", "WS", "WT", "X", "XA", "XP", "XS", "XT", "XW", "Y", "YA", "YB", "YC",
    "YH", "Z",
}

# Какие коды допустимы для класса УГО (первый - основной по ГОСТ 2.710).
CLASS_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "resistor": ("R", "RK", "RP", "RS", "RU"),
    "potentiometer": ("R", "RP"),
    "capacitor": ("C",),
    "capacitor_polar": ("C",),
    "fuse": ("FU", "F"),
    "diode": ("VD", "V", "D"),
    "zener": ("VD", "V"),
    "led": ("HL", "VD", "H"),
    "lamp": ("EL", "HL", "H", "E"),
    "relay": ("K", "KA", "KM", "KT", "KV", "KK", "KH"),
    "switch_no": ("SA", "S", "SB", "SF", "SQ", "SP", "SL", "SR", "K", "KA", "KM", "KT", "KV", "Q", "QF", "QS"),
    "switch_nc": ("SA", "S", "SB", "SF", "SQ", "SP", "SL", "SR", "K", "KA", "KM", "KT", "KV", "Q", "QF", "QS"),
    "switch_changeover": ("SA", "S", "K", "KA", "KM", "KT", "KV", "Q"),
    "pushbutton_no": ("SB", "S", "SA"),
    "inductor": ("L", "LL"),
    "transformer": ("T", "TV", "TA", "TS"),
    "transistor_npn": ("VT", "V"),
    "transistor_pnp": ("VT", "V"),
    "battery": ("GB", "G"),
    "generator": ("G", "GE"),
    "motor": ("M",),
    "connector_body": ("X", "XS", "XP", "XT", "XA", "XW"),
    "plug_contact": ("XP", "X"),
    "socket_contact": ("XS", "X"),
    "terminal": ("XT", "X"),
}

# Номера по ГОСТ 2.710 начинаются с 1 (R0, K01 - ошибки OCR, а не обозначения).
_DESIGNATOR_RE = re.compile(r"^([A-Z]{1,3})([1-9]\d{0,2})(?:\.([1-9]\d?))?$")


def normalize_designator(text: str) -> Optional[str]:
    """'К1,1' -> 'K1.1', 'R l' -> 'R1', 'V D2' -> 'VD2'; иначе None."""
    if not text:
        return None
    s = text.translate(_HOMOGLYPHS).upper()
    s = re.sub(r"[\s_\-]+", "", s).replace(",", ".")
    s = s.strip(".:;'\"`()[]{}|")
    m = re.match(r"^([A-Z]{1,3})([0-9OIL|.]{1,6})$", s)
    if not m:
        return None
    letters, tail = m.groups()
    # В номере O/I/L/| - почти всегда ошибки распознавания 0/1.
    tail = tail.replace("O", "0").replace("I", "1").replace("L", "1").replace("|", "1")
    # Буква L/I, прилипшая к коду (KL -> K1): если код неизвестен, а без
    # последней буквы известен - это была цифра 1.
    if letters not in _KNOWN_PREFIXES and letters[:-1] in _KNOWN_PREFIXES and letters[-1] in "IL":
        letters, tail = letters[:-1], "1" + tail
    candidate = letters + tail
    m = _DESIGNATOR_RE.match(candidate)
    if not m or m.group(1) not in _KNOWN_PREFIXES:
        return None
    return candidate


def restore_decimal_point(text: str, ink) -> str:
    """'K11' -> 'K1.1', если на изображении подписи между цифрами есть точка.

    OCR часто теряет точку в обозначениях элементов изделия (K1.1 - контакт 1
    реле K1). ``ink`` - бинарная (bool) вырезка изображения подписи. Глифы
    ищутся как связные компоненты: высокие - буквы/цифры, маленькая у нижнего
    края - точка. Если число высоких глифов совпадает с длиной строки,
    точка вставляется на своё место.
    """
    import cv2
    import numpy as np

    s = text.strip()
    norm = normalize_designator(s)
    if not norm or "." in norm or ink is None or ink.size == 0:
        return text
    m = re.match(r"^([A-Z]{1,3})(\d{2,4})$", norm)
    if not m:
        return text
    h = ink.shape[0]
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    tall, dots = [], []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if hh >= 0.45 * h:
            tall.append(x + w / 2.0)
        elif hh <= 0.3 * h and w <= 0.3 * h and y + hh >= 0.6 * h and area >= 2:
            dots.append(x + w / 2.0)
    if len(tall) != len(norm) or len(dots) != 1:
        return text
    tall.sort()
    before = sum(1 for cx in tall if cx < dots[0])
    letters = len(m.group(1))
    if letters < before < len(norm):
        return norm[:before] + "." + norm[before:]
    return text


def designator_prefix(designator: str) -> str:
    m = _DESIGNATOR_RE.match(designator)
    return m.group(1) if m else ""


def _rect_distance(point: Tuple[float, float], bbox: Sequence[int]) -> float:
    x, y = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    return (dx * dx + dy * dy) ** 0.5


def _center(bbox: Sequence[int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def assign_designators(
    components: Sequence[Dict],
    texts: Iterable[Dict],
    max_distance: float = 60.0,
) -> Dict[int, Dict]:
    """Сопоставить OCR-обозначения компонентам.

    components: dict с ключами ``bbox`` и ``gost_class`` (имя класса УГО).
    texts: результаты OCR (``text``, ``bbox``, ``confidence``).
    Возвращает {индекс компонента: {"designator", "confidence", "text", "distance"}}.
    """
    labels = []
    for t in texts:
        des = normalize_designator(t.get("text", ""))
        if des:
            labels.append((des, t))

    pairs = []
    for li, (des, t) in enumerate(labels):
        prefix = designator_prefix(des)
        tc = _center(t["bbox"])
        th = max(1, t["bbox"][3] - t["bbox"][1])
        for ci, comp in enumerate(components):
            allowed = CLASS_PREFIXES.get(comp.get("gost_class", ""), ())
            if prefix not in allowed:
                continue
            bbox = comp["bbox"]
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            limit = max(max_distance, 0.8 * size, 3 * th)
            d = _rect_distance(tc, bbox)
            if d <= limit:
                # основной код класса чуть предпочтительнее альтернативного
                rank = allowed.index(prefix)
                pairs.append((d + 4.0 * rank, d, li, ci))

    result: Dict[int, Dict] = {}
    used_labels, used_comps, used_names = set(), set(), set()
    for _, d, li, ci in sorted(pairs):
        des, t = labels[li]
        if li in used_labels or ci in used_comps or des in used_names:
            continue
        used_labels.add(li)
        used_comps.add(ci)
        used_names.add(des)
        result[ci] = {
            "designator": des,
            "confidence": float(t.get("confidence", 0.0)),
            "text": t.get("text", ""),
            "distance": round(d, 1),
        }
    return result


def connector_pin_labels(
    comp_bbox: Sequence[int],
    pin_coords: Dict[str, List[Tuple[int, int]]],
    texts: Iterable[Dict],
) -> Dict[str, str]:
    """Номера контактов разъёма: цифры, напечатанные внутри его ячеек.

    pin_coords: {текущее имя вывода: [координаты выводов этой ячейки]}.
    Возвращает {текущее имя: распознанный номер}.
    """
    x0, y0, x1, y1 = comp_bbox
    inside = []
    for t in texts:
        raw = t.get("text", "").strip().translate(_HOMOGLYPHS).upper()
        raw = raw.replace("O", "0").replace("I", "1").replace("L", "1")
        if not re.fullmatch(r"\d{1,3}", raw):
            continue
        cx, cy = _center(t["bbox"])
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append((raw, cx, cy))
    mapping: Dict[str, str] = {}
    taken = set()
    for name, coords in pin_coords.items():
        if not coords:
            continue
        px = sum(c[0] for c in coords) / len(coords)
        py = sum(c[1] for c in coords) / len(coords)
        vertical = all(abs(c[1] - py) <= 2 for c in coords)  # выводы слева/справа ячейки
        best = None
        for num, cx, cy in inside:
            d = abs(cy - py) if vertical else abs(cx - px)
            if best is None or d < best[0]:
                best = (d, num)
        if best is not None and best[1] not in taken:
            cell = (y1 - y0) / max(1, len(pin_coords)) if vertical else (x1 - x0) / max(1, len(pin_coords))
            if best[0] <= 0.5 * cell:
                mapping[name] = best[1]
                taken.add(best[1])
    # Partial reads must not create duplicate contact names: two cells with
    # the same name would be merged into one node of the netlist graph.
    final = [mapping.get(name, name) for name in pin_coords]
    if len(set(final)) != len(final):
        return {}
    return mapping
