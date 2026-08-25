"""
Обучение RT-DETR / YOLO на ГОСТ УГО датасете.

Использование:
  python -m avers.dataset.train --model rtdetr --data /tmp/avers_dataset/dataset.yaml --epochs 100

Поддерживает:
  - YOLOv11 (ultralytics)
  - RT-DETRv2 (ultralytics)
  - Кастомные аугментации для ГОСТ
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from avers.core.logger import get_logger

logger = get_logger("avers.dataset.train")


def get_training_config(model_type: str = "rtdetr") -> Dict[str, Any]:
    """Получить конфиг обучения для модели."""
    
    base_config = {
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "device": "cuda",
        "workers": 4,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        # ГОСТ-специфичные аугментации (схемы - не фото)
        "hsv_h": 0.015,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 2.0,  # Малый поворот (схемы обычно прямые)
        "translate": 0.05,
        "scale": 0.1,
        "shear": 1.0,
        "perspective": 0.0,  # Нет перспективы для схем
        "flipud": 0.0,
        "fliplr": 0.0,  # Не флипаем (текст)
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    
    if model_type.startswith("yolo"):
        base_config.update({
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
        })
    elif model_type.startswith("rtdetr"):
        base_config.update({
            "box": 5.0,
            "cls": 2.0,
        })
    
    return base_config


def train_yolo(
    data_yaml: Path,
    model_name: str = "yolo11x.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cuda",
    project: str = "/tmp/avers_runs",
    **kwargs
):
    """Обучить YOLO модель."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Install: pip install ultralytics")
        return None
    
    logger.info(f"Training YOLO: {model_name} on {data_yaml}")
    
    model = YOLO(model_name)
    
    config = get_training_config("yolo")
    config.update(kwargs)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name="avers_yolo",
        **config
    )
    
    logger.info(f"Training complete. Best model: {results}")
    
    # Экспорт в ONNX
    try:
        model.export(format="onnx", dynamic=True)
        logger.info("Exported to ONNX")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    return results


def train_rtdetr(
    data_yaml: Path,
    model_name: str = "rtdetr-l.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cuda",
    project: str = "/tmp/avers_runs",
    **kwargs
):
    """Обучить RT-DETR модель."""
    try:
        from ultralytics import RTDETR
    except ImportError:
        logger.error("ultralytics not installed. Install: pip install ultralytics")
        return None
    
    logger.info(f"Training RT-DETR: {model_name} on {data_yaml}")
    
    model = RTDETR(model_name)
    
    config = get_training_config("rtdetr")
    config.update(kwargs)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name="avers_rtdetr",
        **config
    )
    
    logger.info(f"Training complete. Best model: {results}")
    
    # Экспорт
    try:
        model.export(format="onnx", dynamic=True)
        logger.info("Exported to ONNX")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    return results


def validate_model(
    model_path: Path,
    data_yaml: Path,
    model_type: str = "yolo",
):
    """Валидация модели."""
    try:
        if model_type.startswith("yolo"):
            from ultralytics import YOLO
            model = YOLO(str(model_path))
        else:
            from ultralytics import RTDETR
            model = RTDETR(str(model_path))
        
        metrics = model.val(data=str(data_yaml))
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="AVERS ГОСТ УГО Training")
    parser.add_argument("--model", type=str, default="rtdetr", choices=["yolo", "rtdetr", "yolo11x", "yolo11m", "rtdetr-l", "rtdetr-x"], help="Model type")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--project", type=str, default="/tmp/avers_runs", help="Project dir")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weights")
    
    args = parser.parse_args()
    
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        return 1
    
    # Determine model
    if args.model in ("yolo", "yolo11x"):
        model_name = args.pretrained or "yolo11x.pt"
        train_yolo(data_yaml, model_name, args.epochs, args.imgsz, args.batch, args.device, args.project)
    elif args.model in ("yolo11m",):
        model_name = args.pretrained or "yolo11m.pt"
        train_yolo(data_yaml, model_name, args.epochs, args.imgsz, args.batch, args.device, args.project)
    elif args.model in ("rtdetr", "rtdetr-l"):
        model_name = args.pretrained or "rtdetr-l.pt"
        train_rtdetr(data_yaml, model_name, args.epochs, args.imgsz, args.batch, args.device, args.project)
    elif args.model in ("rtdetr-x",):
        model_name = args.pretrained or "rtdetr-x.pt"
        train_rtdetr(data_yaml, model_name, args.epochs, args.imgsz, args.batch, args.device, args.project)
    else:
        logger.error(f"Unknown model: {args.model}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
