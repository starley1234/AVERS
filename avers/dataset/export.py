"""Экспорт датасета в разные форматы: YOLO, COCO, RT-DETR."""

import json
from pathlib import Path
from typing import List, Dict
import shutil

from avers.dataset.synthetic import Annotation
from avers.dataset.gost_symbols import GOST_SYMBOLS


class YOLOExporter:
    """Экспорт в YOLO формат."""
    
    @staticmethod
    def export_annotations(annotations: List[Annotation], image_width: int, image_height: int, output_path: Path):
        """Экспорт аннотаций в YOLO txt."""
        with open(output_path, "w") as f:
            for ann in annotations:
                f.write(ann.to_yolo(image_width, image_height) + "\n")
    
    @staticmethod
    def create_dataset_yaml(output_dir: Path, class_names: Dict[int, str] = None):
        """Создать dataset.yaml."""
        if class_names is None:
            class_names = {id: sym.class_name for id, sym in GOST_SYMBOLS.items()}
        
        yaml_content = f"""# AVERS ГОСТ УГО Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names:
"""
        for id, name in class_names.items():
            yaml_content += f"  {id}: {name}\n"
        
        with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)


class COCOExporter:
    """Экспорт в COCO формат."""
    
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self._ann_id = 0
        self._img_id = 0
        
        # Создать категории
        for id, sym in GOST_SYMBOLS.items():
            self.categories.append({
                "id": id,
                "name": sym.class_name,
                "supercategory": "ugo",
                "gost": sym.gost_standard,
            })
    
    def add_image(self, file_name: str, width: int, height: int, annotations: List[Annotation]):
        """Добавить изображение с аннотациями."""
        self.images.append({
            "id": self._img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })
        
        for ann in annotations:
            self.annotations.append(ann.to_coco(self._ann_id, self._img_id))
            self._ann_id += 1
        
        self._img_id += 1
    
    def save(self, output_path: Path):
        """Сохранить COCO json."""
        coco_data = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
            "info": {
                "description": "AVERS ГОСТ УГО Dataset",
                "version": "1.0",
                "year": 2024,
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def from_yolo_dataset(yolo_dir: Path, output_path: Path):
        """Конвертировать YOLO датасет в COCO."""
        exporter = COCOExporter()
        
        images_dir = yolo_dir / "images" / "train"
        labels_dir = yolo_dir / "labels" / "train"
        
        if not images_dir.exists():
            images_dir = yolo_dir / "images"
            labels_dir = yolo_dir / "labels"
        
        import cv2
        
        for img_path in sorted(images_dir.glob("*.jpg")):
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            anns = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    class_name = GOST_SYMBOLS.get(class_id, None)
                    name = class_name.class_name if class_name else f"class_{class_id}"
                    
                    anns.append(Annotation(class_id, name, (x1, y1, x2, y2)))
            
            exporter.add_image(img_path.name, w, h, anns)
        
        exporter.save(output_path)
        return exporter
