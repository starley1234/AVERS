"""Тесты обнаружения и маскирования основной надписи (штампа, ГОСТ 2.104)."""

import numpy as np
import cv2
import pytest

from avers.core.title_block import (
    find_title_block,
    region_from_fractions,
    filter_out_region,
    bbox_overlaps_region,
)
from avers.config import AVERSConfig
from avers.core.validators import ProductionPipeline


def draw_title_block(img, x1=1150, y1=1050, x2=1900, y2=1350, rows=5, cols=4, thickness=3):
    """Нарисовать сетку-штамп в правом нижнем углу."""
    for i in range(rows + 1):
        y = y1 + i * (y2 - y1) // rows
        cv2.line(img, (x1, y), (x2, y), (0, 0, 0), thickness)
    for j in range(cols + 1):
        x = x1 + j * (x2 - x1) // cols
        cv2.line(img, (x, y1), (x, y2), (0, 0, 0), thickness)
    return img


def make_drawing(with_stamp=True):
    """Белый лист: пара проводов сверху и (опц.) штамп снизу справа."""
    img = np.full((1400, 2000, 3), 255, dtype=np.uint8)
    # провода (длинные горизонтали в ВЕРХНЕЙ части - не должны приниматься за штамп)
    cv2.line(img, (100, 150), (1800, 150), (0, 0, 0), 3)
    cv2.line(img, (100, 260), (1500, 260), (0, 0, 0), 3)
    cv2.line(img, (300, 150), (300, 260), (0, 0, 0), 3)
    if with_stamp:
        img = draw_title_block(img)
    return img


class TestFindTitleBlock:
    def test_finds_stamp(self):
        img = make_drawing(with_stamp=True)
        region = find_title_block(img)
        assert region is not None, "штамп не найден"
        x1, y1, x2, y2 = region
        # регион должен покрывать нарисованную таблицу
        assert x1 <= 1180 and x2 >= 1870
        assert y1 <= 1080 and y2 >= 1320

    def test_no_stamp_on_plain_wires(self):
        img = make_drawing(with_stamp=False)
        region = find_title_block(img)
        assert region is None, f"ложное срабатывание на проводах: {region}"

    def test_small_image_ignored(self):
        region = find_title_block(np.full((150, 150, 3), 255, np.uint8))
        assert region is None

    def test_manual_region_from_fractions(self):
        r = region_from_fractions((1400, 2000), [0.6, 0.8, 0.99, 0.99])
        assert r == (1200, 1120, 1980, 1386)
        assert region_from_fractions((100, 100), None) is None
        assert region_from_fractions((100, 100), [0.1]) is None


class TestFilterHelpers:
    def test_filter_out_region(self):
        items = [
            {"bbox": (100, 100, 200, 200)},    # вне штампа
            {"bbox": (1200, 1100, 1300, 1200)}, # внутри штампа
        ]
        bboxes = [(100, 100, 200, 200), (1200, 1100, 1300, 1200)]
        region = (1150, 1050, 1900, 1350)
        kept, kept_bb = filter_out_region(items, bboxes, region)
        assert len(kept) == 1 and kept[0]["bbox"] == (100, 100, 200, 200)
        assert kept_bb == [(100, 100, 200, 200)]

    def test_bbox_overlaps(self):
        assert bbox_overlaps_region((1200, 1100, 1300, 1200), (1150, 1050, 1900, 1350))
        assert not bbox_overlaps_region((100, 100, 200, 200), (1150, 1050, 1900, 1350))


class TestPipelineIntegration:
    def _run(self, img, **pre):
        cfg = AVERSConfig()
        cfg.preprocess.title_block_mask = True
        for k, v in pre.items():
            setattr(cfg.preprocess, k, v)
        return ProductionPipeline(config=cfg).run(img, "tb_test")

    def test_stamp_masked_in_pipeline(self):
        img = make_drawing(with_stamp=True)
        result = self._run(img)
        assert result.title_block_region is not None, "пайплайн не нашёл штамп"
        assert any("штамп" in w for w in result.warnings)
        # компоненты не должны появиться внутри региона штампа
        for comp in result.manifest.components:
            x1, y1, x2, y2 = comp.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx1, ry1, rx2, ry2 = result.title_block_region
            assert not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2), \
                f"компонент {comp.designator} внутри замаскированного штампа"

    def test_mask_disabled(self):
        img = make_drawing(with_stamp=True)
        result = self._run(img, title_block_mask=False)
        assert result.title_block_region is None
        assert not any("штамп" in w for w in result.warnings)

    def test_manual_mode(self):
        img = make_drawing(with_stamp=False)  # штампа нет, но задаём вручную
        result = self._run(img, title_block_mode="manual", title_block_region=[0.6, 0.8, 0.99, 0.99])
        assert result.title_block_region is not None
        assert result.title_block_region == [1200, 1120, 1980, 1386]

    def test_off_mode(self):
        img = make_drawing(with_stamp=True)
        result = self._run(img, title_block_mode="off")
        assert result.title_block_region is None
