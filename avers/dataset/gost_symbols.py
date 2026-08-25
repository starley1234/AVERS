"""
ГОСТ УГО - Определения условных графических обозначений по ГОСТ 2.721-74, 2.728-74, 2.730-73 и др.

Используется для синтетической генерации датасета для RT-DETR / YOLO.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2


@dataclass
class GOSTSymbol:
    """Определение ГОСТ символа."""
    class_id: int
    class_name: str
    gost_standard: str
    description: str
    width_range: Tuple[int, int] = (20, 100)
    height_range: Tuple[int, int] = (20, 100)
    has_pins: bool = False
    pin_count_range: Tuple[int, int] = (1, 1)
    color: Tuple[int, int, int] = (0, 0, 0)  # BGR for drawing
    draw_fn: Optional[str] = None  # name of draw method


def _draw_connector_body(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Рисует корпус разъема - прямоугольник с утолщением."""
    x1, y1, x2, y2 = bbox
    thickness = kwargs.get('thickness', 2)
    # Основной корпус
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)
    # Внутренняя штриховка для обозначения разъема
    if (x2-x1) > 30:
        mid_x = (x1+x2)//2
        cv2.line(img, (mid_x, y1), (mid_x, y2), (0,0,0), 1)
    return img


def _draw_pin(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Рисует пин - маленький квадрат или точку контакта."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    size = min(x2-x1, y2-y1)//2
    # Круг-контакт
    cv2.circle(img, (cx, cy), max(2, size), (0,0,0), -1)
    # Линия отвода
    cv2.line(img, (cx, cy), (cx+size*2, cy), (0,0,0), 1)
    return img


def _draw_junction_dot(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Точка соединения - закрашенный круг 3-6px."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    r = kwargs.get('radius', np.random.randint(3, 7))
    cv2.circle(img, (cx, cy), r, (0,0,0), -1)
    return img


def _draw_ground(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Земля - три горизонтальные линии убывающей длины."""
    x1, y1, x2, y2 = bbox
    cx = (x1+x2)//2
    cy = y1 + (y2-y1)//3
    # Вертикальная линия
    cv2.line(img, (cx, y1), (cx, cy), (0,0,0), 2)
    # Три горизонтальные
    w = (x2-x1)
    cv2.line(img, (cx - w//2, cy), (cx + w//2, cy), (0,0,0), 2)
    cv2.line(img, (cx - w//3, cy+6), (cx + w//3, cy+6), (0,0,0), 2)
    cv2.line(img, (cx - w//4, cy+12), (cx + w//4, cy+12), (0,0,0), 2)
    return img


def _draw_shield(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Экран - пунктирная окружность или прямоугольник."""
    x1, y1, x2, y2 = bbox
    # Пунктирный прямоугольник
    for x in range(x1, x2, 6):
        cv2.line(img, (x, y1), (min(x+3, x2), y1), (0,0,0), 1)
        cv2.line(img, (x, y2), (min(x+3, x2), y2), (0,0,0), 1)
    for y in range(y1, y2, 6):
        cv2.line(img, (x1, y), (x1, min(y+3, y2)), (0,0,0), 1)
        cv2.line(img, (x2, y), (x2, min(y+3, y2)), (0,0,0), 1)
    return img


def _draw_offpage(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Стрелка перехода на другой лист - треугольник со стрелкой."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    # Треугольник-стрелка
    pts = np.array([[x1, y1], [x2, cy], [x1, y2]], np.int32)
    cv2.polylines(img, [pts], True, (0,0,0), 2)
    cv2.fillPoly(img, [pts], (200,200,200))
    return img


def _draw_diode(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Диод - треугольник + черта (ГОСТ 2.730)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    w, h = (x2-x1), (y2-y1)
    # Треугольник
    pts = np.array([[x1, y1], [x1, y2], [cx, cy]], np.int32)
    cv2.polylines(img, [pts], True, (0,0,0), 2)
    # Черта
    cv2.line(img, (cx, y1), (cx, y2), (0,0,0), 2)
    # Выводы
    cv2.line(img, (x1-5, cy), (x1, cy), (0,0,0), 1)
    cv2.line(img, (cx, cy), (x2+5, cy), (0,0,0), 1)
    return img


def _draw_resistor(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Резистор - зигзаг (ГОСТ 2.728)."""
    x1, y1, x2, y2 = bbox
    cy = (y1+y2)//2
    w = x2-x1
    # Зигзаг
    points = []
    steps = 6
    for i in range(steps+1):
        x = x1 + (w * i)//steps
        y = cy + (10 if i%2==0 else -10)
        if i==0 or i==steps:
            y = cy
        points.append((x, y))
    for i in range(len(points)-1):
        cv2.line(img, points[i], points[i+1], (0,0,0), 2)
    # Выводы
    cv2.line(img, (x1-5, cy), (x1, cy), (0,0,0), 1)
    cv2.line(img, (x2, cy), (x2+5, cy), (0,0,0), 1)
    return img


def _draw_relay(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Реле - прямоугольник с катушкой."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 2)
    # Катушка внутри - несколько полукругов
    mid_y = (y1+y2)//2
    for i in range(3):
        x = x1 + 10 + i*12
        if x < x2-10:
            cv2.ellipse(img, (x, mid_y), (6, 8), 0, 0, 180, (0,0,0), 1)
    return img


def _draw_capacitor(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Конденсатор - две параллельные линии."""
    x1, y1, x2, y2 = bbox
    cx = (x1+x2)//2
    cv2.line(img, (cx-3, y1), (cx-3, y2), (0,0,0), 2)
    cv2.line(img, (cx+3, y1), (cx+3, y2), (0,0,0), 2)
    cy = (y1+y2)//2
    cv2.line(img, (x1, cy), (cx-3, cy), (0,0,0), 1)
    cv2.line(img, (cx+3, cy), (x2, cy), (0,0,0), 1)
    return img


# Реестр ГОСТ символов
GOST_SYMBOLS: Dict[int, GOSTSymbol] = {
    0: GOSTSymbol(0, "connector_body", "ГОСТ 2.755-87", "Корпус разъема", (40, 120), (60, 200), has_pins=True, pin_count_range=(2, 20), draw_fn="_draw_connector_body"),
    1: GOSTSymbol(1, "pin", "ГОСТ 2.755-87", "Контакт разъема", (6, 16), (6, 16), draw_fn="_draw_pin"),
    2: GOSTSymbol(2, "junction_dot", "ГОСТ 2.721-74", "Точка соединения", (4, 12), (4, 12), draw_fn="_draw_junction_dot"),
    3: GOSTSymbol(3, "ground", "ГОСТ 2.721-74", "Заземление / корпус", (20, 40), (20, 40), draw_fn="_draw_ground"),
    4: GOSTSymbol(4, "shield", "ГОСТ 2.721-74", "Экран кабеля", (30, 80), (20, 50), draw_fn="_draw_shield"),
    5: GOSTSymbol(5, "offpage_connector", "ГОСТ 2.721-74", "Переход на другой лист", (20, 50), (15, 30), draw_fn="_draw_offpage"),
    6: GOSTSymbol(6, "diode", "ГОСТ 2.730-73", "Диод", (20, 40), (12, 24), draw_fn="_draw_diode"),
    7: GOSTSymbol(7, "relay", "ГОСТ 2.756-76", "Реле / катушка", (40, 80), (20, 40), draw_fn="_draw_relay"),
    8: GOSTSymbol(8, "resistor", "ГОСТ 2.728-74", "Резистор", (30, 60), (10, 20), draw_fn="_draw_resistor"),
    9: GOSTSymbol(9, "capacitor", "ГОСТ 2.728-74", "Конденсатор", (20, 30), (10, 20), draw_fn="_draw_capacitor"),
}

# Маппинг имени функции к реальному callable
DRAW_FUNCTIONS = {
    "_draw_connector_body": _draw_connector_body,
    "_draw_pin": _draw_pin,
    "_draw_junction_dot": _draw_junction_dot,
    "_draw_ground": _draw_ground,
    "_draw_shield": _draw_shield,
    "_draw_offpage": _draw_offpage,
    "_draw_diode": _draw_diode,
    "_draw_resistor": _draw_resistor,
    "_draw_relay": _draw_relay,
    "_draw_capacitor": _draw_capacitor,
}

# Дополнительные символы для расширенного датасета
EXTENDED_SYMBOLS = {
    10: GOSTSymbol(10, "wire_crossing", "ГОСТ 2.721-74", "Пересечение без соединения", (10, 20), (10, 20)),
    11: GOSTSymbol(11, "wire_junction_T", "ГОСТ 2.721-74", "Т-образное соединение", (10, 20), (10, 20)),
    12: GOSTSymbol(12, "text_label", "ГОСТ 2.304-81", "Текстовая метка", (20, 100), (10, 20)),
}


def draw_symbol(img: np.ndarray, symbol: GOSTSymbol, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Отрисовать символ в изображении."""
    fn_name = symbol.draw_fn
    if fn_name and fn_name in DRAW_FUNCTIONS:
        return DRAW_FUNCTIONS[fn_name](img, bbox, **kwargs)
    else:
        # Fallback: простой прямоугольник
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 1)
        cv2.putText(img, symbol.class_name[:3], (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        return img


def get_symbol_by_name(name: str) -> Optional[GOSTSymbol]:
    """Найти символ по имени класса."""
    for sym in GOST_SYMBOLS.values():
        if sym.class_name == name:
            return sym
    for sym in EXTENDED_SYMBOLS.values():
        if sym.class_name == name:
            return sym
    return None
