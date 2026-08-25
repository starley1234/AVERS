#!/usr/bin/env python3
"""
CLI для работы с датасетом ГОСТ УГО.

Использование:
  python -m avers.dataset.cli generate --num 1000 --output /tmp/avers_dataset
  python -m avers.dataset.cli train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l
  python -m avers.dataset.cli export --format coco --input /tmp/avers_dataset --output /tmp/coco.json
"""

import argparse
import sys
from pathlib import Path

from avers.core.logger import get_logger

logger = get_logger("avers.dataset.cli")


def cmd_generate(args):
    """Генерация синтетического датасета."""
    from avers.dataset.generator import generate_full_dataset
    
    output_dir = Path(args.output)
    logger.info(f"Generating dataset to {output_dir}")
    
    generate_full_dataset(
        output_dir=str(output_dir),
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        image_size=args.image_size,
    )
    
    logger.info(f"Dataset generated at {output_dir}")
    logger.info(f"Train: {args.num_train}, Val: {args.num_val}, Test: {args.num_test}")


def cmd_train(args):
    """Обучение модели."""
    from avers.dataset.train import train_yolo, train_rtdetr
    
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        return 1
    
    model_type = args.model.lower()
    
    if "yolo" in model_type:
        model_name = args.pretrained or ("yolo11x.pt" if "x" in model_type else "yolo11m.pt")
        train_yolo(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
        )
    elif "rtdetr" in model_type or "detr" in model_type:
        model_name = args.pretrained or ("rtdetr-x.pt" if "x" in model_type else "rtdetr-l.pt")
        train_rtdetr(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
        )
    else:
        logger.error(f"Unknown model type: {model_type}")
        return 1
    
    return 0


def cmd_export(args):
    """Экспорт датасета."""
    from avers.dataset.export import COCOExporter
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    
    if args.format == "coco":
        COCOExporter.from_yolo_dataset(input_dir, output_path)
        logger.info(f"Exported COCO to {output_path}")
    else:
        logger.error(f"Unknown format: {args.format}")
        return 1
    
    return 0


def cmd_preview(args):
    """Превью синтетических изображений."""
    from avers.dataset.synthetic import GOSTGenerator, SyntheticConfig
    import cv2
    
    config = SyntheticConfig(
        image_size=args.size,
        min_objects=5,
        max_objects=20,
    )
    
    gen = GOSTGenerator(config)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(args.num):
        img, anns = gen.generate_realistic_schematic()
        path = output_dir / f"preview_{i:04d}.jpg"
        cv2.imwrite(str(path), img)
        logger.info(f"Saved {path} with {len(anns)} annotations")
    
    logger.info(f"Preview images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="avers-dataset",
        description="AVERS ГОСТ УГО Dataset Tools"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--output", "-o", type=str, default="/tmp/avers_dataset", help="Output directory")
    gen_parser.add_argument("--num-train", type=int, default=1000, help="Number of train images")
    gen_parser.add_argument("--num-val", type=int, default=200, help="Number of val images")
    gen_parser.add_argument("--num-test", type=int, default=100, help="Number of test images")
    gen_parser.add_argument("--image-size", type=int, default=1024, help="Image size")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train detection model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    train_parser.add_argument("--model", type=str, default="rtdetr-l", help="Model: yolo11x, yolo11m, rtdetr-l, rtdetr-x")
    train_parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    train_parser.add_argument("--batch", type=int, default=8, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device")
    train_parser.add_argument("--project", type=str, default="/tmp/avers_runs", help="Project dir")
    train_parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weights")
    
    # Export
    export_parser = subparsers.add_parser("export", help="Export dataset")
    export_parser.add_argument("--input", "-i", type=str, required=True, help="Input dataset dir")
    export_parser.add_argument("--output", "-o", type=str, required=True, help="Output file")
    export_parser.add_argument("--format", "-f", type=str, default="coco", choices=["coco", "yolo"], help="Export format")
    
    # Preview
    preview_parser = subparsers.add_parser("preview", help="Preview synthetic images")
    preview_parser.add_argument("--output", "-o", type=str, default="/tmp/avers_preview", help="Output dir")
    preview_parser.add_argument("--num", type=int, default=10, help="Number of images")
    preview_parser.add_argument("--size", type=int, default=1024, help="Image size")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "generate":
            cmd_generate(args)
        elif args.command == "train":
            return cmd_train(args)
        elif args.command == "export":
            return cmd_export(args)
        elif args.command == "preview":
            cmd_preview(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
