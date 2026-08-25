"""
Active Learning Loop - замыкание цикла валидатор -> дообучение.

Поток:
  1. Пользователь исправляет в Web UI валидаторе (компоненты, тексты, junctions)
  2. Исправления сохраняются как FeedbackEntry в /tmp/avers_feedback/
  3. При накоплении N примеров - триггер дообучения
  4. Генерация новых синтетических данных с учетом ошибок
  5. Fine-tune модели RT-DETR/YOLO
  6. Обновление модели в ProductionPipeline

Интеграция с Vision RAG: исправления сразу индексируются в RAG для few-shot.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json
import uuid
import shutil

import numpy as np
import cv2

from avers.core.logger import get_logger

logger = get_logger("avers.active_learning")


@dataclass
class FeedbackEntry:
    """Запись обратной связи от валидатора."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    file_id: str = ""
    image_path: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    original_label: str = ""
    corrected_label: str = ""
    original_bbox: Optional[Tuple[int, int, int, int]] = None
    corrected_bbox: Optional[Tuple[int, int, int, int]] = None
    issue_type: str = ""  # low_confidence_detection, suspicious_crossing, etc.
    user_id: str = "anonymous"
    comment: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    image_crop: Optional[np.ndarray] = None  # 256x256 ROI
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "file_id": self.file_id,
            "image_path": self.image_path,
            "bbox": self.bbox,
            "original_label": self.original_label,
            "corrected_label": self.corrected_label,
            "original_bbox": self.original_bbox,
            "corrected_bbox": self.corrected_bbox,
            "issue_type": self.issue_type,
            "user_id": self.user_id,
            "comment": self.comment,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FeedbackEntry":
        return cls(
            id=data["id"],
            file_id=data.get("file_id", ""),
            image_path=data.get("image_path", ""),
            bbox=tuple(data.get("bbox", (0,0,0,0))),
            original_label=data.get("original_label", ""),
            corrected_label=data.get("corrected_label", ""),
            original_bbox=tuple(data["original_bbox"]) if data.get("original_bbox") else None,
            corrected_bbox=tuple(data["corrected_bbox"]) if data.get("corrected_bbox") else None,
            issue_type=data.get("issue_type", ""),
            user_id=data.get("user_id", "anonymous"),
            comment=data.get("comment", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


class ActiveLearningLoop:
    """
    Active Learning Loop для АВЕРС.
    
    Использование:
      loop = ActiveLearningLoop()
      
      # При исправлении в валидаторе
      loop.add_feedback(
          file_id="abc123",
          bbox=(100,100,200,200),
          original_label="diode",
          corrected_label="resistor",
          issue_type="low_confidence_detection",
          image_crop=roi_image,
          comment="Это резистор, не диод"
      )
      
      # Проверка нужно ли дообучать
      if loop.should_retrain(threshold=50):
          loop.trigger_retraining()
    """
    
    def __init__(
        self,
        feedback_dir: Path = Path("/tmp/avers_feedback"),
        rag_enabled: bool = True,
        min_feedback_for_retrain: int = 50,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.crops_dir = self.feedback_dir / "crops"
        self.crops_dir.mkdir(exist_ok=True)
        
        self.meta_file = self.feedback_dir / "feedback.jsonl"
        
        self.rag_enabled = rag_enabled
        self.min_feedback_for_retrain = min_feedback_for_retrain
        
        self.feedback_entries: List[FeedbackEntry] = []
        self.load()
        
        # RAG integration
        self._rag = None
        if rag_enabled:
            try:
                from avers.rag import get_rag
                self._rag = get_rag()
            except Exception as e:
                logger.warning(f"RAG not available for active learning: {e}")
    
    def load(self):
        """Загрузить feedback из файла."""
        if not self.meta_file.exists():
            return
        
        self.feedback_entries = []
        with open(self.meta_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    entry = FeedbackEntry.from_dict(data)
                    self.feedback_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load feedback line: {e}")
        
        logger.info(f"Loaded {len(self.feedback_entries)} feedback entries")
    
    def save_entry(self, entry: FeedbackEntry):
        """Сохранить одну запись."""
        with open(self.meta_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        
        # Save crop if available
        if entry.image_crop is not None:
            crop_path = self.crops_dir / f"{entry.id}.jpg"
            cv2.imwrite(str(crop_path), entry.image_crop)
            entry.image_path = str(crop_path)
    
    def add_feedback(
        self,
        file_id: str,
        bbox: Tuple[int, int, int, int],
        original_label: str,
        corrected_label: str,
        issue_type: str = "",
        original_bbox: Optional[Tuple[int, int, int, int]] = None,
        corrected_bbox: Optional[Tuple[int, int, int, int]] = None,
        image_crop: Optional[np.ndarray] = None,
        user_id: str = "anonymous",
        comment: str = "",
    ) -> FeedbackEntry:
        """Добавить feedback от валидатора."""
        entry = FeedbackEntry(
            file_id=file_id,
            bbox=bbox,
            original_label=original_label,
            corrected_label=corrected_label,
            original_bbox=original_bbox,
            corrected_bbox=corrected_bbox,
            issue_type=issue_type,
            user_id=user_id,
            comment=comment,
            image_crop=image_crop,
        )
        
        self.feedback_entries.append(entry)
        self.save_entry(entry)
        
        # Index in RAG immediately for few-shot improvement
        if self._rag and image_crop is not None:
            try:
                self._rag.add_example(
                    image=image_crop,
                    label=corrected_label,
                    bbox=corrected_bbox or bbox,
                    description=comment or f"Corrected from {original_label} to {corrected_label}",
                    metadata={"feedback_id": entry.id, "issue_type": issue_type}
                )
                logger.info(f"Feedback {entry.id} indexed in RAG")
            except Exception as e:
                logger.warning(f"Failed to index feedback in RAG: {e}")
        
        logger.info(f"Added feedback {entry.id}: {original_label} -> {corrected_label}")
        return entry
    
    def should_retrain(self, threshold: Optional[int] = None) -> bool:
        """Проверить нужно ли дообучать модель."""
        thresh = threshold or self.min_feedback_for_retrain
        return len(self.feedback_entries) >= thresh
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Статистика исправлений."""
        if not self.feedback_entries:
            return {"total": 0}
        
        # Count by type
        by_issue = {}
        by_label = {}
        by_correction = {}
        
        for entry in self.feedback_entries:
            by_issue[entry.issue_type] = by_issue.get(entry.issue_type, 0) + 1
            by_label[entry.original_label] = by_label.get(entry.original_label, 0) + 1
            key = f"{entry.original_label}->{entry.corrected_label}"
            by_correction[key] = by_correction.get(key, 0) + 1
        
        return {
            "total": len(self.feedback_entries),
            "by_issue_type": by_issue,
            "by_original_label": by_label,
            "by_correction": by_correction,
            "unique_files": len(set(e.file_id for e in self.feedback_entries)),
            "unique_users": len(set(e.user_id for e in self.feedback_entries)),
        }
    
    def export_for_training(self, output_dir: Path) -> Path:
        """
        Экспортировать feedback как датасет для дообучения.
        
        Returns:
            Path to dataset.yaml
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images" / "train"
        labels_dir = output_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy crops and create YOLO labels
        from avers.dataset.gost_symbols import GOST_SYMBOLS
        class_name_to_id = {s.class_name: s.class_id for s in GOST_SYMBOLS.values()}
        
        for entry in self.feedback_entries:
            crop_path = self.crops_dir / f"{entry.id}.jpg"
            if not crop_path.exists():
                continue
            
            # Copy image
            dest_img = images_dir / f"feedback_{entry.id}.jpg"
            shutil.copy(crop_path, dest_img)
            
            # Create YOLO label (full image is the object)
            class_id = class_name_to_id.get(entry.corrected_label, 0)
            # bbox is in original image coords, but crop is 256x256, so object is at center
            # For simplicity, use full crop as object
            yolo_line = f"{class_id} 0.5 0.5 0.9 0.9"
            
            label_path = labels_dir / f"feedback_{entry.id}.txt"
            with open(label_path, "w") as f:
                f.write(yolo_line + "\n")
        
        # Create dataset.yaml that includes both synthetic and feedback
        yaml_content = f"""
# AVERS Active Learning Dataset
path: {output_dir.absolute()}
train: images/train
val: images/train  # For simplicity, use train as val for feedback

nc: {len(class_name_to_id)}
names:
"""
        for id, sym in GOST_SYMBOLS.items():
            yaml_content += f"  {id}: {sym.class_name}\n"
        
        yaml_path = output_dir / "dataset_feedback.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        logger.info(f"Exported {len(self.feedback_entries)} feedback samples to {output_dir}")
        return yaml_path
    
    def trigger_retraining(
        self,
        base_dataset_yaml: Optional[Path] = None,
        output_dir: Path = Path("/tmp/avers_active_learning"),
        model_type: str = "rtdetr-l",
        epochs: int = 20,
    ) -> Dict[str, Any]:
        """
        Запустить дообучение на feedback + base dataset.
        
        Args:
            base_dataset_yaml: базовый синтетический датасет
            output_dir: куда сохранить результаты
            model_type: модель для дообучения
            epochs: эпох дообучения (меньше чем с нуля)
        
        Returns:
            Dict с результатами
        """
        logger.info(f"Triggering retraining with {len(self.feedback_entries)} feedback samples")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export feedback
        feedback_yaml = self.export_for_training(output_dir / "feedback_dataset")
        
        # If base dataset provided, merge (for now just use feedback)
        # In production: merge datasets
        
        # Train
        try:
            from avers.dataset.train import train_rtdetr, train_yolo
            
            if "rtdetr" in model_type.lower():
                train_rtdetr(
                    data_yaml=feedback_yaml,
                    model_name=f"{model_type}.pt",
                    epochs=epochs,
                    batch=4,  # Smaller batch for fine-tuning
                    project=str(output_dir / "runs"),
                )
            else:
                train_yolo(
                    data_yaml=feedback_yaml,
                    model_name=f"{model_type}.pt",
                    epochs=epochs,
                    batch=4,
                    project=str(output_dir / "runs"),
                )
            
            result = {
                "status": "success",
                "feedback_samples": len(self.feedback_entries),
                "model_type": model_type,
                "epochs": epochs,
                "output_dir": str(output_dir),
                "feedback_yaml": str(feedback_yaml),
            }
            
            logger.info(f"Retraining complete: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "feedback_samples": len(self.feedback_entries),
            }
    
    def clear(self):
        """Очистить feedback (после успешного дообучения)."""
        self.feedback_entries.clear()
        if self.meta_file.exists():
            self.meta_file.unlink()
        # Keep crops for history
        logger.info("Feedback cleared")


# Global instance
_global_loop: Optional[ActiveLearningLoop] = None

def get_active_learning_loop(feedback_dir: Path = Path("/tmp/avers_feedback")) -> ActiveLearningLoop:
    global _global_loop
    if _global_loop is None:
        _global_loop = ActiveLearningLoop(feedback_dir=feedback_dir)
    return _global_loop
