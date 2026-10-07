"""
ГОСТ УГО - библиотека условных графических обозначений для электрических схем.

Источники (ЕСКД):
  * ГОСТ 2.721-74  - обозначения общего применения (заземление, корпус, экран,
                     точка соединения линий);
  * ГОСТ 2.722-68  - машины электрические (двигатель M, генератор G);
  * ГОСТ 2.723-68  - катушки индуктивности, трансформаторы;
  * ГОСТ 2.727-68  - разрядники, предохранители;
  * ГОСТ 2.728-74  - резисторы, конденсаторы;
  * ГОСТ 2.730-73  - приборы полупроводниковые (диод, стабилитрон,
                     светодиод, транзисторы);
  * ГОСТ 2.732-68  - источники света (лампа накаливания);
  * ГОСТ 2.755-87  - устройства коммутационные и контактные соединения
                     (контакты замыкающий/размыкающий/переключающий,
                     кнопка, штырь/гнездо разъёмного соединения, разъём);
  * ГОСТ 2.756-76  - воспринимающая часть электромеханических устройств
                     (обмотка реле);
  * ГОСТ 2.768-90  - источники электрохимические (батарея GB);
  * ГОСТ 2.710-81  - буквенно-цифровые обозначения (R, C, VD, K, FU, SA ...).

Каждый символ описан в миллиметрах (как в стандарте, вместе с короткими
отрезками выводов), имеет координаты выводов (pins) и буквенный код.
Отрисовка масштабируется под любой bbox, поэтому одна и та же геометрия
используется:
  * синтетическим генератором датасета (YOLO / RT-DETR);
  * CPU-детектором по шаблонам (``avers.stages.stage2_detection.template_detector``),
    который работает без обученных весов;
  * генератором демонстрационной схемы (``avers.dataset.demo_schematic``).

Идентификаторы классов 0-9 сохранены с v0.1 (совместимость обученных весов и
разметки), новые классы добавлены с id 10 и выше.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
PinSpec = Tuple[str, float, float]  # (имя вывода, x_mm, y_mm)


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
    # --- v0.4: реальная геометрия по ГОСТ ---
    size_mm: Tuple[float, float] = (10.0, 4.0)  # габарит УГО вместе с выводами
    pins: Tuple[PinSpec, ...] = ()               # выводы в мм от левого верхнего угла
    designator: str = ""                         # буквенный код по ГОСТ 2.710-81
    component_type: str = "unknown"              # значение avers.core.types.ComponentType
    template_detect: bool = True                 # использовать в шаблонном детекторе
    chiral: bool = False                         # нужно ли зеркальное отражение при поиске

    def pins_in_bbox(self, bbox: Tuple[int, int, int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """Координаты выводов в пикселях для символа, нарисованного в bbox."""
        x1, y1, x2, y2 = bbox
        w_mm, h_mm = self.size_mm
        return [
            (name, (int(round(x1 + x / w_mm * (x2 - x1))), int(round(y1 + y / h_mm * (y2 - y1)))))
            for name, x, y in self.pins
        ]


# =============================================================================
# Перо: рисование в миллиметровых координатах символа
# =============================================================================

class _Pen:
    """Переводит координаты символа (мм) в пиксели bbox и рисует примитивы."""

    def __init__(self, img: np.ndarray, bbox: Tuple[int, int, int, int],
                 size_mm: Tuple[float, float], thickness: int = 2,
                 color: Tuple[int, int, int] = (0, 0, 0)):
        self.img = img
        self.x1, self.y1, x2, y2 = bbox
        self.sx = (x2 - self.x1) / float(size_mm[0])
        self.sy = (y2 - self.y1) / float(size_mm[1])
        self.t = max(1, int(thickness))
        self.color = color

    def p(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(self.x1 + x * self.sx)), int(round(self.y1 + y * self.sy)))

    def line(self, a: Point, b: Point, thick: float = 1.0) -> None:
        cv2.line(self.img, self.p(*a), self.p(*b), self.color,
                 max(1, int(round(self.t * thick))), cv2.LINE_AA)

    def polyline(self, pts, closed: bool = False, fill: bool = False) -> None:
        arr = np.array([self.p(*q) for q in pts], np.int32)
        if fill:
            cv2.fillPoly(self.img, [arr], self.color, cv2.LINE_AA)
        cv2.polylines(self.img, [arr], closed, self.color, self.t, cv2.LINE_AA)

    def rect(self, a: Point, b: Point, fill: bool = False) -> None:
        pa, pb = self.p(*a), self.p(*b)
        cv2.rectangle(self.img, pa, pb, self.color, -1 if fill else self.t, cv2.LINE_AA)

    def circle(self, c: Point, r: float, fill: bool = False) -> None:
        axes = (max(1, int(round(r * self.sx))), max(1, int(round(r * self.sy))))
        cv2.ellipse(self.img, self.p(*c), axes, 0, 0, 360, self.color,
                    -1 if fill else self.t, cv2.LINE_AA)

    def arc(self, c: Point, r: float, start: float, end: float) -> None:
        """Дуга; углы в градусах как в OpenCV (0 = вправо, по часовой, ось Y вниз)."""
        axes = (max(1, int(round(r * self.sx))), max(1, int(round(r * self.sy))))
        cv2.ellipse(self.img, self.p(*c), axes, 0, start, end, self.color, self.t, cv2.LINE_AA)

    def dashed(self, a: Point, b: Point, dash: float = 1.0, gap: float = 0.8) -> None:
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length == 0:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        s = 0.0
        while s < length:
            e = min(length, s + dash)
            self.line((a[0] + ux * s, a[1] + uy * s), (a[0] + ux * e, a[1] + uy * e))
            s = e + gap

    def arrow_head(self, tip: Point, tail: Point, size: float = 1.2, fill: bool = True) -> None:
        dx, dy = tip[0] - tail[0], tip[1] - tail[1]
        n = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / n, dy / n
        bx, by = tip[0] - ux * size, tip[1] - uy * size
        w = size * 0.4
        pts = [tip, (bx - uy * w, by + ux * w), (bx + uy * w, by - ux * w)]
        self.polyline(pts, closed=True, fill=fill)

    def letter(self, text: str, c: Point, height_mm: float) -> None:
        """Буква внутри УГО (M, G, +) шрифтом Hershey, по центру точки c."""
        h_px = max(6.0, height_mm * self.sy)
        scale = h_px / 22.0
        thick = max(1, self.t)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cx, cy = self.p(*c)
        cv2.putText(self.img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, self.color, thick, cv2.LINE_AA)


def _pen(img, bbox, size_mm, kwargs) -> _Pen:
    return _Pen(img, bbox, size_mm, kwargs.get("thickness", 2), kwargs.get("color", (0, 0, 0)))


# =============================================================================
# Функции отрисовки. Все - в горизонтальной ориентации, координаты в мм.
# =============================================================================

def _draw_connector_body(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Разъём многоконтактный (ГОСТ 2.755-87, табл. 6 п.2-4): прямоугольник,
    разделённый на ячейки контактов; номера контактов пишутся в ячейках."""
    x1, y1, x2, y2 = bbox
    t = kwargs.get("thickness", 2)
    n = int(kwargs.get("pin_count", max(2, (y2 - y1) // 24)))
    n = max(1, n)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), t)
    for i in range(1, n):
        y = int(round(y1 + i * (y2 - y1) / n))
        cv2.line(img, (x1, y), (x2, y), (0, 0, 0), max(1, t - 1))
    if kwargs.get("numbers", True):
        cell_h = (y2 - y1) / n
        scale = max(0.3, min(0.9, cell_h / 40.0))
        for i in range(n):
            label = str(i + 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cy = int(y1 + (i + 0.5) * cell_h)
            cv2.putText(img, label, ((x1 + x2) // 2 - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _draw_pin(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Контакт (ячейка) разъёма - маленький квадрат с отводом."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    size = max(2, min(x2 - x1, y2 - y1) // 2)
    cv2.circle(img, (cx, cy), size, (0, 0, 0), -1)
    cv2.line(img, (cx, cy), (cx + size * 2, cy), (0, 0, 0), 1)
    return img


def _draw_junction_dot(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Точка электрического соединения линий (ГОСТ 2.721-74): зачернённый круг."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = kwargs.get("radius", max(2, min(x2 - x1, y2 - y1) // 2))
    cv2.circle(img, (cx, cy), int(r), (0, 0, 0), -1, cv2.LINE_AA)
    return img


def _draw_ground(img, bbox, **kwargs):
    """Заземление (ГОСТ 2.721-74): вывод и три черты убывающей длины."""
    p = _pen(img, bbox, (10, 8), kwargs)
    p.line((5, 0), (5, 3.5))
    p.line((0, 3.5), (10, 3.5))
    p.line((1.8, 5.5), (8.2, 5.5))
    p.line((3.5, 7.5), (6.5, 7.5))
    return img


def _draw_chassis(img, bbox, **kwargs):
    """Соединение с корпусом (ГОСТ 2.721-74): черта со штрихами."""
    p = _pen(img, bbox, (10, 6), kwargs)
    p.line((5, 0), (5, 3))
    p.line((0, 3), (10, 3))
    for x in (2.2, 5.8, 9.4):
        p.line((x, 3), (x - 1.8, 5.8))
    return img


def _draw_shield(img: np.ndarray, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Экранирование (ГОСТ 2.721-74): штриховой контур."""
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2, 6):
        cv2.line(img, (x, y1), (min(x + 3, x2), y1), (0, 0, 0), 1)
        cv2.line(img, (x, y2), (min(x + 3, x2), y2), (0, 0, 0), 1)
    for y in range(y1, y2, 6):
        cv2.line(img, (x1, y), (x1, min(y + 3, y2)), (0, 0, 0), 1)
        cv2.line(img, (x2, y), (x2, min(y + 3, y2)), (0, 0, 0), 1)
    return img


def _draw_offpage(img, bbox, **kwargs):
    """Обрыв линии связи с переходом на другой лист (ГОСТ 2.702): флажок."""
    p = _pen(img, bbox, (12, 6), kwargs)
    p.line((0, 3), (3, 3))
    p.polyline([(3, 0.5), (9, 0.5), (12, 3), (9, 5.5), (3, 5.5)], closed=True)
    return img


def _draw_resistor(img, bbox, **kwargs):
    """Резистор постоянный (ГОСТ 2.728-74): прямоугольник 10x4 мм."""
    p = _pen(img, bbox, (16, 4), kwargs)
    p.line((0, 2), (3, 2))
    p.rect((3, 0), (13, 4))
    p.line((13, 2), (16, 2))
    return img


def _draw_potentiometer(img, bbox, **kwargs):
    """Резистор переменный с подвижным контактом (ГОСТ 2.728-74)."""
    p = _pen(img, bbox, (16, 8), kwargs)
    p.line((0, 6), (3, 6))
    p.rect((3, 4), (13, 8))
    p.line((13, 6), (16, 6))
    p.line((8, 0), (8, 3))
    p.arrow_head((8, 4), (8, 0), size=1.3)
    return img


def _draw_fuse(img, bbox, **kwargs):
    """Предохранитель плавкий (ГОСТ 2.727-68): прямоугольник 10x4 с проводником."""
    p = _pen(img, bbox, (16, 4), kwargs)
    p.line((0, 2), (16, 2))
    p.rect((3, 0), (13, 4))
    return img


def _draw_capacitor(img, bbox, **kwargs):
    """Конденсатор постоянной ёмкости (ГОСТ 2.728-74): две обкладки."""
    p = _pen(img, bbox, (10, 7), kwargs)
    p.line((0, 3.5), (4.25, 3.5))
    p.line((4.25, 0), (4.25, 7), thick=1.3)
    p.line((5.75, 0), (5.75, 7), thick=1.3)
    p.line((5.75, 3.5), (10, 3.5))
    return img


def _draw_capacitor_polar(img, bbox, **kwargs):
    """Конденсатор оксидный поляризованный (ГОСТ 2.728-74)."""
    p = _pen(img, bbox, (10, 7), kwargs)
    p.line((0, 3.5), (3.6, 3.5))
    p.rect((3.6, 0), (4.6, 7))
    p.line((6.2, 0), (6.2, 7), thick=1.3)
    p.line((6.2, 3.5), (10, 3.5))
    p.line((1.0, 1.2), (2.6, 1.2))
    p.line((1.8, 0.4), (1.8, 2.0))
    return img


def _draw_diode(img, bbox, **kwargs):
    """Диод (ГОСТ 2.730-73): треугольник (анод) и черта (катод)."""
    p = _pen(img, bbox, (12, 5), kwargs)
    p.line((0, 2.5), (4, 2.5))
    p.polyline([(4, 0), (4, 5), (8, 2.5)], closed=True)
    p.line((8, 0), (8, 5))
    p.line((8, 2.5), (12, 2.5))
    return img


def _draw_zener(img, bbox, **kwargs):
    """Стабилитрон односторонний (ГОСТ 2.730-73): диод с флажком на катоде."""
    p = _pen(img, bbox, (12, 5), kwargs)
    p.line((0, 2.5), (4, 2.5))
    p.polyline([(4, 0), (4, 5), (8, 2.5)], closed=True)
    p.line((8, 0), (8, 5))
    p.line((8, 0), (6.6, 0))
    p.line((8, 2.5), (12, 2.5))
    return img


def _draw_led(img, bbox, **kwargs):
    """Светодиод (ГОСТ 2.730-73): диод и две стрелки излучения."""
    p = _pen(img, bbox, (12, 8), kwargs)
    p.line((0, 5.5), (4, 5.5))
    p.polyline([(4, 3), (4, 8), (8, 5.5)], closed=True)
    p.line((8, 3), (8, 8))
    p.line((8, 5.5), (12, 5.5))
    for x0 in (6.2, 8.2):
        tail, tip = (x0, 2.8), (x0 + 2.2, 0.6)
        p.line(tail, tip)
        p.arrow_head(tip, tail, size=1.0)
    return img


def _draw_relay(img, bbox, **kwargs):
    """Обмотка реле (ГОСТ 2.756-76): прямоугольник, выводы с длинных сторон."""
    p = _pen(img, bbox, (10, 10), kwargs)
    p.line((5, 0), (5, 3))
    p.rect((0, 3), (10, 7))
    p.line((5, 7), (5, 10))
    return img


def _draw_switch_no(img, bbox, **kwargs):
    """Контакт коммутационного устройства замыкающий (ГОСТ 2.755-87)."""
    p = _pen(img, bbox, (14, 6), kwargs)
    p.line((0, 5), (4, 5))
    p.line((4, 5), (10.6, 1.4))
    p.line((10, 5), (14, 5))
    return img


def _draw_switch_nc(img, bbox, **kwargs):
    """Контакт размыкающий (ГОСТ 2.755-87): подвижный контакт за упором."""
    p = _pen(img, bbox, (14, 6), kwargs)
    p.line((0, 5), (4, 5))
    p.line((4, 5), (11, 1.4))
    p.line((10, 5), (10, 1.6))
    p.line((10, 5), (14, 5))
    return img


def _draw_switch_changeover(img, bbox, **kwargs):
    """Контакт переключающий (ГОСТ 2.755-87)."""
    p = _pen(img, bbox, (14, 8), kwargs)
    p.line((0, 4), (4, 4))
    p.line((4, 4), (10.8, 1.6))
    p.line((10, 0.8), (14, 0.8))
    p.line((10, 0.8), (10, 2.2))
    p.line((10, 7), (14, 7))
    return img


def _draw_pushbutton(img, bbox, **kwargs):
    """Выключатель кнопочный нажимной с замыкающим контактом (ГОСТ 2.755-87)."""
    p = _pen(img, bbox, (14, 10), kwargs)
    p.line((0, 9), (4, 9))
    p.line((4, 9), (10.6, 5.4))
    p.line((10, 9), (14, 9))
    p.dashed((7.3, 7.2), (7.3, 1.6))
    p.line((5.4, 1), (9.2, 1))
    p.line((5.4, 1), (5.4, 2.2))
    p.line((9.2, 1), (9.2, 2.2))
    return img


def _draw_lamp(img, bbox, **kwargs):
    """Лампа накаливания осветительная/сигнальная (ГОСТ 2.732-68)."""
    p = _pen(img, bbox, (14, 8), kwargs)
    p.line((0, 4), (3, 4))
    p.circle((7, 4), 4)
    d = 4 * 0.7071
    p.line((7 - d, 4 - d), (7 + d, 4 + d))
    p.line((7 - d, 4 + d), (7 + d, 4 - d))
    p.line((11, 4), (14, 4))
    return img


def _draw_inductor(img, bbox, **kwargs):
    """Катушка индуктивности, обмотка (ГОСТ 2.723-68): четыре полуокружности."""
    p = _pen(img, bbox, (18, 2.5), kwargs)
    p.line((0, 2.0), (3, 2.0))
    for cx in (4.5, 7.5, 10.5, 13.5):
        p.arc((cx, 2.0), 1.5, 180, 360)
    p.line((15, 2.0), (18, 2.0))
    return img


def _draw_transformer(img, bbox, **kwargs):
    """Трансформатор с магнитопроводом (ГОСТ 2.723-68)."""
    p = _pen(img, bbox, (14, 14), kwargs)
    p.line((0, 2), (4, 2))
    p.line((0, 12), (4, 12))
    p.line((10, 2), (14, 2))
    p.line((10, 12), (14, 12))
    for cy in (3.25, 5.75, 8.25, 10.75):
        p.arc((4, cy), 1.25, -90, 90)
        p.arc((10, cy), 1.25, 90, 270)
    p.line((6.6, 2), (6.6, 12))
    p.line((7.4, 2), (7.4, 12))
    return img


def _draw_transistor(img, bbox, npn: bool = True, **kwargs):
    """Транзистор биполярный (ГОСТ 2.730-73) в корпусе."""
    p = _pen(img, bbox, (14, 14), kwargs)
    p.circle((8, 7), 5.5)
    p.line((0, 7), (5.5, 7))
    p.line((5.5, 3.8), (5.5, 10.2), thick=1.4)
    p.line((5.5, 5.6), (9.5, 3.2))
    p.line((9.5, 3.2), (9.5, 0))
    p.line((5.5, 8.4), (9.5, 10.8))
    p.line((9.5, 10.8), (9.5, 14))
    if npn:
        p.arrow_head((9.0, 10.5), (5.5, 8.4), size=1.6)
    else:
        p.arrow_head((6.0, 8.7), (9.5, 10.8), size=1.6)
    return img


def _draw_transistor_npn(img, bbox, **kwargs):
    return _draw_transistor(img, bbox, npn=True, **kwargs)


def _draw_transistor_pnp(img, bbox, **kwargs):
    return _draw_transistor(img, bbox, npn=False, **kwargs)


def _draw_battery(img, bbox, **kwargs):
    """Батарея гальваническая/аккумуляторная из двух элементов (ГОСТ 2.768-90).
    Длинная тонкая черта - положительный полюс, короткая толстая - отрицательный."""
    p = _pen(img, bbox, (12, 8), kwargs)
    p.line((0, 4), (4, 4))
    p.line((4, 0), (4, 8))
    p.line((5.3, 2), (5.3, 6), thick=2.0)
    p.dashed((5.3, 4), (6.7, 4), dash=0.5, gap=0.4)
    p.line((6.7, 0), (6.7, 8))
    p.line((8, 2), (8, 6), thick=2.0)
    p.line((8, 4), (12, 4))
    return img


def _draw_machine(img, bbox, letter: str, **kwargs):
    p = _pen(img, bbox, (16, 10), kwargs)
    p.line((0, 5), (3, 5))
    p.circle((8, 5), 5)
    p.letter(letter, (8, 5), 4.2)
    p.line((13, 5), (16, 5))
    return img


def _draw_motor(img, bbox, **kwargs):
    """Электродвигатель (ГОСТ 2.722-68): окружность с буквой M."""
    return _draw_machine(img, bbox, "M", **kwargs)


def _draw_generator(img, bbox, **kwargs):
    """Генератор (ГОСТ 2.722-68): окружность с буквой G."""
    return _draw_machine(img, bbox, "G", **kwargs)


def _draw_plug(img, bbox, **kwargs):
    """Контакт разъёмного соединения - штырь (ГОСТ 2.755-87)."""
    p = _pen(img, bbox, (8, 3), kwargs)
    p.line((0, 1.5), (4.5, 1.5))
    p.polyline([(4.5, 0.4), (8, 1.5), (4.5, 2.6)], closed=True, fill=True)
    return img


def _draw_socket(img, bbox, **kwargs):
    """Контакт разъёмного соединения - гнездо (ГОСТ 2.755-87)."""
    p = _pen(img, bbox, (7, 4), kwargs)
    p.line((0, 2), (3.5, 2))
    p.polyline([(7, 0), (3.5, 2), (7, 4)])
    return img


def _draw_terminal(img, bbox, **kwargs):
    """Контакт разборного соединения, зажим (ГОСТ 2.755-87): окружность."""
    p = _pen(img, bbox, (6, 2.4), kwargs)
    p.line((0, 1.2), (3.6, 1.2))
    p.circle((4.8, 1.2), 1.2)
    return img


# =============================================================================
# Реестр ГОСТ символов
# =============================================================================

def _sym(cid, name, gost, desc, size_mm, pins, designator, ctype, draw_fn, **kw) -> GOSTSymbol:
    """Создать символ; диапазоны размеров для генератора выводятся из size_mm
    (масштаб ~3-7 px/мм, т.е. 75-180 dpi), сохраняя пропорции УГО."""
    w, h = size_mm
    kw.setdefault("width_range", (max(4, int(w * 3)), max(6, int(w * 7))))
    kw.setdefault("height_range", (max(4, int(h * 3)), max(6, int(h * 7))))
    return GOSTSymbol(cid, name, gost, desc, size_mm=size_mm, pins=tuple(pins),
                      designator=designator, component_type=ctype, draw_fn=draw_fn, **kw)


def _h2(w: float, h: float, y: Optional[float] = None, names=("1", "2")) -> Tuple[PinSpec, ...]:
    """Два вывода на левом и правом краях (горизонтальная ориентация)."""
    y = h / 2 if y is None else y
    return ((names[0], 0.0, y), (names[1], w, y))


GOST_SYMBOLS: Dict[int, GOSTSymbol] = {
    0: GOSTSymbol(0, "connector_body", "ГОСТ 2.755-87", "Разъём (соединение контактное разъёмное многоконтактное)",
                  (40, 120), (60, 200), has_pins=True, pin_count_range=(2, 20),
                  draw_fn="_draw_connector_body", size_mm=(8, 24), designator="X",
                  component_type="connector", template_detect=False),
    1: GOSTSymbol(1, "pin", "ГОСТ 2.755-87", "Контакт разъёма", (6, 16), (6, 16),
                  draw_fn="_draw_pin", size_mm=(2, 2), template_detect=False),
    2: GOSTSymbol(2, "junction_dot", "ГОСТ 2.721-74", "Точка электрического соединения линий", (4, 12), (4, 12),
                  draw_fn="_draw_junction_dot", size_mm=(1.5, 1.5), component_type="junction_dot",
                  template_detect=False),
    3: _sym(3, "ground", "ГОСТ 2.721-74", "Заземление", (10, 8), (("1", 5, 0),), "", "ground", "_draw_ground"),
    4: GOSTSymbol(4, "shield", "ГОСТ 2.721-74", "Экранирование", (30, 80), (20, 50),
                  draw_fn="_draw_shield", size_mm=(20, 10), component_type="shield", template_detect=False),
    5: _sym(5, "offpage_connector", "ГОСТ 2.702-2011", "Обрыв линии связи / переход на другой лист",
            (12, 6), (("1", 0, 3),), "", "offpage_connector", "_draw_offpage", chiral=False),
    6: _sym(6, "diode", "ГОСТ 2.730-73", "Диод", (12, 5), _h2(12, 5, names=("A", "K")),
            "VD", "diode", "_draw_diode"),
    7: _sym(7, "relay", "ГОСТ 2.756-76", "Обмотка реле (катушка электромеханического устройства)",
            (10, 10), (("1", 5, 0), ("2", 5, 10)), "K", "relay", "_draw_relay"),
    8: _sym(8, "resistor", "ГОСТ 2.728-74", "Резистор постоянный", (16, 4), _h2(16, 4),
            "R", "resistor", "_draw_resistor"),
    9: _sym(9, "capacitor", "ГОСТ 2.728-74", "Конденсатор постоянной ёмкости", (10, 7), _h2(10, 7),
            "C", "capacitor", "_draw_capacitor"),
    10: _sym(10, "fuse", "ГОСТ 2.727-68", "Предохранитель плавкий", (16, 4), _h2(16, 4),
             "FU", "fuse", "_draw_fuse"),
    11: _sym(11, "switch_no", "ГОСТ 2.755-87", "Контакт замыкающий (выключатель)", (14, 6), _h2(14, 6, 5),
             "SA", "switch", "_draw_switch_no", chiral=True),
    12: _sym(12, "switch_nc", "ГОСТ 2.755-87", "Контакт размыкающий", (14, 6), _h2(14, 6, 5),
             "SA", "switch", "_draw_switch_nc", chiral=True),
    13: _sym(13, "switch_changeover", "ГОСТ 2.755-87", "Контакт переключающий", (14, 8),
             (("C", 0, 4), ("NC", 14, 0.8), ("NO", 14, 7)), "SA", "switch", "_draw_switch_changeover",
             chiral=True),
    14: _sym(14, "pushbutton_no", "ГОСТ 2.755-87", "Выключатель кнопочный нажимной замыкающий", (14, 10),
             _h2(14, 10, 9), "SB", "pushbutton", "_draw_pushbutton", chiral=True),
    15: _sym(15, "lamp", "ГОСТ 2.732-68", "Лампа накаливания осветительная и сигнальная", (14, 8), _h2(14, 8),
             "EL", "lamp", "_draw_lamp"),
    16: _sym(16, "led", "ГОСТ 2.730-73", "Светодиод", (12, 8), _h2(12, 8, 5.5, names=("A", "K")),
             "HL", "led", "_draw_led", chiral=True),
    17: _sym(17, "zener", "ГОСТ 2.730-73", "Стабилитрон", (12, 5), _h2(12, 5, names=("A", "K")),
             "VD", "zener", "_draw_zener", chiral=True),
    18: _sym(18, "capacitor_polar", "ГОСТ 2.728-74", "Конденсатор оксидный поляризованный", (10, 7),
             _h2(10, 7, names=("+", "-")), "C", "capacitor", "_draw_capacitor_polar", chiral=True),
    19: _sym(19, "inductor", "ГОСТ 2.723-68", "Катушка индуктивности, дроссель", (18, 2.5), _h2(18, 2.5, 2.0),
             "L", "inductor", "_draw_inductor", chiral=True),
    20: _sym(20, "transformer", "ГОСТ 2.723-68", "Трансформатор с магнитопроводом", (14, 14),
             (("1", 0, 2), ("2", 0, 12), ("3", 14, 2), ("4", 14, 12)), "T", "transformer",
             "_draw_transformer"),
    21: _sym(21, "transistor_npn", "ГОСТ 2.730-73", "Транзистор биполярный n-p-n", (14, 14),
             (("B", 0, 7), ("C", 9.5, 0), ("E", 9.5, 14)), "VT", "transistor", "_draw_transistor_npn",
             chiral=True),
    22: _sym(22, "transistor_pnp", "ГОСТ 2.730-73", "Транзистор биполярный p-n-p", (14, 14),
             (("B", 0, 7), ("C", 9.5, 0), ("E", 9.5, 14)), "VT", "transistor", "_draw_transistor_pnp",
             chiral=True),
    23: _sym(23, "battery", "ГОСТ 2.768-90", "Батарея аккумуляторная / гальваническая", (12, 8),
             _h2(12, 8, names=("+", "-")), "GB", "battery", "_draw_battery"),
    24: _sym(24, "motor", "ГОСТ 2.722-68", "Электродвигатель", (16, 10), _h2(16, 10), "M", "motor", "_draw_motor"),
    25: _sym(25, "generator", "ГОСТ 2.722-68", "Генератор", (16, 10), _h2(16, 10), "G", "generator",
             "_draw_generator"),
    26: _sym(26, "potentiometer", "ГОСТ 2.728-74", "Резистор переменный (потенциометр)", (16, 8),
             (("1", 0, 6), ("2", 8, 0), ("3", 16, 6)), "R", "potentiometer", "_draw_potentiometer"),
    27: _sym(27, "plug_contact", "ГОСТ 2.755-87", "Контакт разъёмного соединения - штырь", (8, 3),
             (("1", 0, 1.5),), "XP", "plug", "_draw_plug"),
    28: _sym(28, "socket_contact", "ГОСТ 2.755-87", "Контакт разъёмного соединения - гнездо", (7, 4),
             (("1", 0, 2),), "XS", "socket", "_draw_socket"),
    29: _sym(29, "terminal", "ГОСТ 2.755-87", "Контакт разборного соединения (зажим, клемма)", (6, 2.4),
             (("1", 0, 1.2),), "XT", "terminal", "_draw_terminal", template_detect=False),
    30: _sym(30, "chassis", "ГОСТ 2.721-74", "Соединение с корпусом (масса)", (10, 6), (("1", 5, 0),),
             "", "chassis", "_draw_chassis"),
}

# Маппинг имени функции к реальному callable
DRAW_FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
    name: fn for name, fn in globals().items() if name.startswith("_draw_") and callable(fn)
}

# Дополнительные (не детектируемые как УГО) классы для расширенной разметки.
# Перенумерованы с 100, чтобы не пересекаться с основными классами.
EXTENDED_SYMBOLS = {
    100: GOSTSymbol(100, "wire_crossing", "ГОСТ 2.721-74", "Пересечение без соединения", (10, 20), (10, 20),
                    template_detect=False),
    101: GOSTSymbol(101, "wire_junction_T", "ГОСТ 2.721-74", "Т-образное соединение", (10, 20), (10, 20),
                    template_detect=False),
    102: GOSTSymbol(102, "text_label", "ГОСТ 2.304-81", "Текстовая метка", (20, 100), (10, 20),
                    template_detect=False),
}


def draw_symbol(img: np.ndarray, symbol: GOSTSymbol, bbox: Tuple[int, int, int, int], **kwargs) -> np.ndarray:
    """Отрисовать символ в изображении."""
    fn_name = symbol.draw_fn
    if fn_name and fn_name in DRAW_FUNCTIONS:
        return DRAW_FUNCTIONS[fn_name](img, bbox, **kwargs)
    # Fallback: простой прямоугольник
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    cv2.putText(img, symbol.class_name[:3], (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return img


def symbol_bbox(symbol: GOSTSymbol, origin: Tuple[int, int], px_per_mm: float) -> Tuple[int, int, int, int]:
    """bbox символа в натуральных пропорциях ГОСТ при заданном масштабе."""
    x, y = origin
    w, h = symbol.size_mm
    return (int(x), int(y), int(round(x + w * px_per_mm)), int(round(y + h * px_per_mm)))


def get_symbol_by_name(name: str) -> Optional[GOSTSymbol]:
    """Найти символ по имени класса."""
    for sym in GOST_SYMBOLS.values():
        if sym.class_name == name:
            return sym
    for sym in EXTENDED_SYMBOLS.values():
        if sym.class_name == name:
            return sym
    return None


def designator_prefix(class_name: str) -> str:
    """Буквенный код по ГОСТ 2.710-81 для класса (или '' если не применяется)."""
    sym = get_symbol_by_name(class_name)
    return sym.designator if sym else ""


def render_library_sheet(px_per_mm: float = 6.0, cols: int = 6) -> np.ndarray:
    """Таблица всех УГО библиотеки (BGR): символ, выводы (красные точки), id/имя."""
    cw, ch = 175, 150
    syms = list(GOST_SYMBOLS.values())
    rows = (len(syms) + cols - 1) // cols
    img = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for i, sym in enumerate(syms):
        r, c = divmod(i, cols)
        x0, y0 = c * cw + 20, r * ch + 34
        if sym.class_name == "connector_body":
            bb = (x0, y0, x0 + 40, y0 + 90)
        else:
            bb = symbol_bbox(sym, (x0, y0), px_per_mm)
        draw_symbol(img, sym, bb, thickness=2, pin_count=4)
        for _, (px, py) in sym.pins_in_bbox(bb):
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
        label = f"{sym.class_id} {sym.class_name}" + (f" ({sym.designator})" if sym.designator else "")
        cv2.putText(img, label, (c * cw + 4, r * ch + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img, sym.gost_standard, (c * cw + 4, r * ch + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (90, 90, 90), 1, cv2.LINE_AA)
    return img


def library_table() -> List[Dict[str, object]]:
    """Сводная таблица библиотеки УГО (для API/документации)."""
    return [
        {
            "class_id": s.class_id,
            "class_name": s.class_name,
            "description": s.description,
            "gost": s.gost_standard,
            "designator": s.designator,
            "component_type": s.component_type,
            "size_mm": list(s.size_mm),
            "pins": [p[0] for p in s.pins],
            "template_detect": s.template_detect,
        }
        for s in GOST_SYMBOLS.values()
    ]
