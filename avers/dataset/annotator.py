"""
Инструмент ручной разметки датасета ГОСТ УГО.

Web-based аннотатор + конвертация в YOLO/COCO.
Предоставляет:
  - API для CRUD аннотаций
  - Экспорт
  - Импорт существующих разметок
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

from avers.dataset.synthetic import Annotation
from avers.dataset.gost_symbols import GOST_SYMBOLS


@dataclass
class DatasetImage:
    """Изображение в датасете."""
    id: str
    file_name: str
    file_path: Path
    width: int
    height: int
    annotations: List[Annotation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    annotated: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "annotations": [
                {"class_id": a.class_id, "class_name": a.class_name, "bbox": a.bbox}
                for a in self.annotations
            ],
            "annotated": self.annotated,
            "created_at": self.created_at.isoformat(),
        }


class AnnotationStore:
    """Хранилище аннотаций."""
    
    def __init__(self, root_dir: Path = Path("/tmp/avers_dataset")):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.meta_file = self.root_dir / "meta.json"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.images: Dict[str, DatasetImage] = {}
        self.load()
    
    def load(self):
        """Загрузить метаданные."""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for img_data in data.get("images", []):
                        anns = [
                            Annotation(a["class_id"], a["class_name"], tuple(a["bbox"]))
                            for a in img_data.get("annotations", [])
                        ]
                        self.images[img_data["id"]] = DatasetImage(
                            id=img_data["id"],
                            file_name=img_data["file_name"],
                            file_path=Path(img_data["file_path"]),
                            width=img_data["width"],
                            height=img_data["height"],
                            annotations=anns,
                            annotated=img_data.get("annotated", False),
                        )
            except Exception as e:
                print(f"Failed to load meta: {e}")
    
    def save(self):
        """Сохранить метаданные."""
        data = {
            "images": [img.to_dict() for img in self.images.values()],
            "classes": {id: sym.class_name for id, sym in GOST_SYMBOLS.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_image(self, file_path: Path, width: int, height: int) -> DatasetImage:
        """Добавить изображение."""
        img_id = uuid.uuid4().hex[:12]
        img = DatasetImage(
            id=img_id,
            file_name=file_path.name,
            file_path=file_path,
            width=width,
            height=height,
        )
        self.images[img_id] = img
        self.save()
        return img
    
    def update_annotations(self, image_id: str, annotations: List[Dict]) -> Optional[DatasetImage]:
        """Обновить аннотации изображения."""
        if image_id not in self.images:
            return None
        
        anns = [
            Annotation(a["class_id"], a.get("class_name", f"class_{a['class_id']}"), tuple(a["bbox"]))
            for a in annotations
        ]
        
        self.images[image_id].annotations = anns
        self.images[image_id].annotated = len(anns) > 0
        self.save()
        
        # Сохранить YOLO txt
        self._save_yolo(image_id)
        
        return self.images[image_id]
    
    def _save_yolo(self, image_id: str):
        """Сохранить YOLO аннотацию."""
        img = self.images[image_id]
        label_path = self.labels_dir / f"{Path(img.file_name).stem}.txt"
        
        with open(label_path, "w") as f:
            for ann in img.annotations:
                f.write(ann.to_yolo(img.width, img.height) + "\n")
    
    def get_image(self, image_id: str) -> Optional[DatasetImage]:
        return self.images.get(image_id)
    
    def list_images(self) -> List[DatasetImage]:
        return list(self.images.values())
    
    def delete_image(self, image_id: str) -> bool:
        if image_id in self.images:
            del self.images[image_id]
            self.save()
            return True
        return False
    
    def export_yolo(self, output_dir: Path, split_ratio: float = 0.8):
        """Экспортировать в YOLO структуру с train/val split."""
        import random
        
        output_dir = Path(output_dir)
        train_img = output_dir / "images" / "train"
        val_img = output_dir / "images" / "val"
        train_lbl = output_dir / "labels" / "train"
        val_lbl = output_dir / "labels" / "val"
        
        for d in [train_img, val_img, train_lbl, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
        
        images = list(self.images.values())
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_set = images[:split_idx]
        val_set = images[split_idx:]
        
        import shutil
        
        for img in train_set:
            # Copy image
            if img.file_path.exists():
                shutil.copy(img.file_path, train_img / img.file_name)
            # Copy label
            src_lbl = self.labels_dir / f"{Path(img.file_name).stem}.txt"
            if src_lbl.exists():
                shutil.copy(src_lbl, train_lbl / src_lbl.name)
        
        for img in val_set:
            if img.file_path.exists():
                shutil.copy(img.file_path, val_img / img.file_name)
            src_lbl = self.labels_dir / f"{Path(img.file_name).stem}.txt"
            if src_lbl.exists():
                shutil.copy(src_lbl, val_lbl / src_lbl.name)
        
        # dataset.yaml
        yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {len(GOST_SYMBOLS)}
names:
"""
        for id, sym in GOST_SYMBOLS.items():
            yaml_content += f"  {id}: {sym.class_name}\n"
        
        with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        return output_dir / "dataset.yaml"


# Глобальный стор
_global_store: Optional[AnnotationStore] = None

def get_store(root_dir: Path = Path("/tmp/avers_dataset")) -> AnnotationStore:
    global _global_store
    if _global_store is None:
        _global_store = AnnotationStore(root_dir)
    return _global_store
