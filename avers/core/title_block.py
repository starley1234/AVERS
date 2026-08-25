"""
Обнаружение основной надписи (штампа, ГОСТ 2.104) на чертеже.

Зачем: штамп - плотная сетка таблицы в правом нижнем углу. Без маскировки
она порождает сотни фантомных линий/"цепей" в векторизации и мусорный текст
в OCR. Найденный регион исключается из стадий 2-4 пайплайна.

Эвристика: длинные горизонтальные линии морфологией; штамп = компактный
кластер, в котором есть и вертикальные линии (сетка), прижатый к правому
нижнему углу. Схемные провода отсекаются проверкой на сетку.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from avers.core.logger import get_logger

logger = get_logger("avers.title_block")

Region = Tuple[int, int, int, int]


def region_from_fractions(
    shape: Tuple[int, ...],
    fractions: List[float],
) -> Optional[Region]:
    """Регион [x1,y1,x2,y2] из долей страницы (0..1)."""
    if not fractions or len(fractions) != 4:
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = fractions
    return (
        max(0, min(w, int(x1 * w))),
        max(0, min(h, int(y1 * h))),
        max(0, min(w, int(x2 * w))),
        max(0, min(h, int(y2 * h))),
    )


def _count_lines(mask: np.ndarray, horizontal: bool) -> int:
    """Число линий (связных компонент) в маске."""
    if mask.size == 0 or mask.max() == 0:
        return 0
    n = cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1
    return n


def find_title_block(
    image: np.ndarray,
    bottom_frac: float = 0.55,
    right_frac: float = 0.55,
    min_line_frac: float = 0.10,
    max_area_frac: float = 0.40,
    min_grid_lines: int = 3,
) -> Optional[Region]:
    """Найти основную надпись (штамп) на изображении.

    Args:
        image: BGR/gray изображение (полный чертёж).
        bottom_frac/right_frac: в какой нижне-правой части искать.
        min_line_frac: минимальная длина гориз. линии (доля ширины).
        max_area_frac: максимальная доля площади листа под штамп.
        min_grid_lines: минимум гориз./верт. линий в сетке штампа.

    Returns:
        (x1, y1, x2, y2) или None, если штамп не найден.
    """
    h, w = image.shape[:2]
    if w < 200 or h < 200:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = (gray < 140).astype(np.uint8) * 255

    # Длинные горизонтальные линии
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(w * min_line_frac)), 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kh)

    # Ищем только в правом нижнем углу (там штамп по ГОСТ 2.104)
    search = np.zeros_like(horiz)
    y0, x0 = int(h * (1 - bottom_frac)), int(w * (1 - right_frac))
    search[y0:, x0:] = horiz[y0:, x0:]

    ys, xs = np.nonzero(search)
    if len(ys) < 30:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1

    # Проверки здравого смысла
    if bw < 0.08 * w or bh < 0.025 * h:
        return None
    if bw * bh > max_area_frac * w * h:
        return None
    if x2 < 0.55 * w or y2 < 0.6 * h:
        return None
    # штамп прижат к углу листа (рамке): правый низ сетки должен быть у края
    if x2 < 0.85 * w or y2 < 0.72 * h:
        logger.debug(f"Кандидат {x1},{y1},{x2},{y2} отклонён: не прижат к правому нижнему углу")
        return None

    # Штамп - это СЕТКА: внутри должны быть и вертикальные линии
    sub = binary[y1:y2 + 1, x1:x2 + 1]
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(bh * 0.2))))
    vert_in = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kv)
    kh2 = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, int(bw * 0.2)), 1))
    hor_in = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kh2)

    n_vert = _count_lines(vert_in, horizontal=False)
    n_hor = _count_lines(hor_in, horizontal=True)
    if n_hor < min_grid_lines or n_vert < min_grid_lines:
        logger.debug(
            f"Кандидат {x1},{y1},{x2},{y2} отклонён: сетка слабая "
            f"(hor={n_hor}, vert={n_vert}) - вероятно, просто провода"
        )
        return None

    region = (x1, y1, x2, y2)
    logger.info(f"Основная надпись (штамп) найдена: {region} (сетка: hor={n_hor}, vert={n_vert})")
    return region


def point_in_region(x: float, y: float, region: Region) -> bool:
    """Центр точки (x, y) внутри региона?"""
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_overlaps_region(bbox: Tuple[float, float, float, float], region: Region) -> bool:
    """Пересекается ли bbox с регионом (по центру)."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return point_in_region(cx, cy, region)


def filter_out_region(
    items: list,
    bboxes: List[Tuple],
    region: Region,
    bbox_getter=lambda it: it["bbox"] if isinstance(it, dict) else getattr(it, "bbox", None),
) -> Tuple[list, list]:
    """Убрать элементы, чей центр попадает в регион.

    Returns:
        (оставшиеся элементы, оставшиеся bbox)
    """
    kept_items, kept_bboxes = [], []
    for it, bb in zip(items, bboxes):
        bb = bbox_getter(it) or bb
        if bb is None:
            continue
        if not bbox_overlaps_region(tuple(bb), region):
            kept_items.append(it)
            kept_bboxes.append(bb)
    return kept_items, kept_bboxes
