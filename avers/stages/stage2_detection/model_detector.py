"""
Детектор УГО на обученных весах (ultralytics YOLO / RT-DETR) с нарезкой на тайлы.

Заменяет прежний SAHI-путь (он обращался к несуществующим в sahi классам и
молча откатывался). Нарезка своя: большие листы (А1/А2, 300 dpi) делятся на
перекрывающиеся тайлы, детекции переводятся в координаты листа и проходят
NMS по классам.

Нейросеть даёт только рамку и класс. Ориентацию и координаты выводов (без
них нет netlist) уточняет ``GOSTTemplateDetector.refine`` - гибридный режим,
см. ``avers.core.validators.SlicedDetector``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from avers.core.logger import get_logger
from avers.core.types import DETECTION_CLASSES

logger = get_logger("avers.detection.model")


class ModelClassMismatch(RuntimeError):
    """Веса обучены не на классах библиотеки ГОСТ УГО (например, COCO)."""


def resolve_device(device: str) -> str:
    """'cuda'/'auto' -> 'cuda:0' если доступна, иначе 'cpu'."""
    want = (device or "auto").lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0" if want in ("auto", "cuda", "gpu") else want
        if want not in ("auto",):
            logger.warning(f"Device '{device}' requested but CUDA is unavailable; using CPU")
    except Exception:
        pass
    return "cpu"


def map_model_classes(names: Dict[int, str]) -> Dict[int, Tuple[int, str]]:
    """{id модели: (id ГОСТ, имя класса)} по совпадению имён классов."""
    gost_by_name = {name: cid for cid, name in DETECTION_CLASSES.items()}
    mapping = {}
    for mid, name in names.items():
        key = str(name).strip()
        if key in gost_by_name:
            mapping[int(mid)] = (gost_by_name[key], key)
    return mapping


def tile_grid(width: int, height: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Перекрывающиеся тайлы, покрывающие лист целиком (последний прижат к краю)."""
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(size):
        if size <= tile:
            return [0]
        s = list(range(0, size - tile, step))
        s.append(size - tile)
        return sorted(set(s))

    return [(x, y, min(width, x + tile), min(height, y + tile))
            for y in starts(height) for x in starts(width)]


def nms_per_class(dets: List[Dict], iou: float = 0.5) -> List[Dict]:
    """NMS внутри каждого класса + подавление «обрезков» на стыке тайлов."""
    out: List[Dict] = []
    by_class: Dict[str, List[Dict]] = {}
    for d in dets:
        by_class.setdefault(d["category"], []).append(d)
    for items in by_class.values():
        items.sort(key=lambda d: -d["confidence"])
        kept: List[Dict] = []
        for d in items:
            a = d["bbox"]
            drop = False
            for k in kept:
                b = k["bbox"]
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                if not inter:
                    continue
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                union = area_a + area_b - inter
                # обрезок символа на краю тайла почти целиком внутри полного
                if inter / max(1, union) > iou or inter / max(1, min(area_a, area_b)) > 0.8:
                    drop = True
                    break
            if not drop:
                kept.append(d)
        out.extend(kept)
    return out


class UltralyticsTiledDetector:
    """YOLO / RT-DETR (ultralytics) с нарезкой листа на тайлы."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "auto",
        tile_size: int = 1024,
        overlap_ratio: float = 0.2,
        img_size: Optional[int] = None,
        model=None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size or tile_size
        self.model = model            # можно передать готовую модель (тесты)
        self.class_map: Dict[int, Tuple[int, str]] = {}
        self.unknown_classes: List[str] = []

    def load(self) -> "UltralyticsTiledDetector":
        if self.model is None:
            path = Path(self.model_path)
            if not path.is_file():
                raise FileNotFoundError(f"Model weights not found: {path}")
            from ultralytics import RTDETR, YOLO  # ImportError -> caller falls back
            cls = RTDETR if "rtdetr" in path.name.lower() else YOLO
            self.model = cls(str(path))
            self.device = resolve_device(self.device)
            logger.info(f"Loaded {cls.__name__} weights {path.name} on {self.device}")
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self.class_map = map_model_classes(names)
        self.unknown_classes = [str(n) for i, n in names.items() if int(i) not in self.class_map]
        if not self.class_map:
            raise ModelClassMismatch(
                f"Классы модели ({', '.join(map(str, list(names.values())[:6]))}...) не совпадают "
                f"с библиотекой ГОСТ УГО - это не веса AVERS (например, COCO yolo*.pt)"
            )
        if self.unknown_classes:
            logger.warning(f"Model classes not in GOST library are ignored: {self.unknown_classes}")
        return self

    def _predict_tile(self, tile: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        results = self.model.predict(
            tile, conf=self.confidence_threshold, iou=self.iou_threshold,
            imgsz=self.img_size, device=self.device, verbose=False,
        )
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for (x0, y0, x1, y1), c, k in zip(xyxy, conf, cls):
                out.append((float(x0), float(y0), float(x1), float(y1), float(c), int(k)))
        return out

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Детекции в координатах листа: bbox, confidence, category, category_id."""
        if not self.class_map:
            self.load()
        h, w = image.shape[:2]
        rgb = image if image.ndim == 3 else np.repeat(image[:, :, None], 3, axis=2)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # ultralytics ждёт BGR для numpy
        dets: List[Dict] = []
        tiles = tile_grid(w, h, self.tile_size, self.overlap_ratio)
        for tx0, ty0, tx1, ty1 in tiles:
            for x0, y0, x1, y1, conf, k in self._predict_tile(bgr[ty0:ty1, tx0:tx1]):
                if k not in self.class_map:
                    continue
                gid, name = self.class_map[k]
                box = (max(0, int(round(x0 + tx0))), max(0, int(round(y0 + ty0))),
                       min(w - 1, int(round(x1 + tx0))), min(h - 1, int(round(y1 + ty0))))
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                dets.append({"bbox": box, "confidence": round(conf, 3), "category": name,
                             "category_id": gid, "source": "model"})
        dets = nms_per_class(dets, self.iou_threshold)
        logger.info(f"Model detector: {len(dets)} objects on {len(tiles)} tile(s)")
        return dets


def merge_detections(primary: Sequence[Dict], secondary: Sequence[Dict], overlap: float = 0.5) -> List[Dict]:
    """primary + те из secondary, что не пересекаются с primary (точки соединения
    добавляются всегда, если не совпадают с точкой из primary)."""
    out = list(primary)
    for d in secondary:
        a = d["bbox"]
        clash = False
        for p in primary:
            if (p["category"] == "junction_dot") != (d["category"] == "junction_dot"):
                continue
            b = p["bbox"]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if inter and inter / max(1, small) > overlap:
                clash = True
                break
        if not clash:
            out.append(d)
    return out
