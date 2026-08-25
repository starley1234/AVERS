#!/usr/bin/env python3
"""
CLI для работы с датасетом ГОСТ УГО + публичные датасеты.

Использование:
  python -m avers.dataset.cli generate --num 1000 --output /tmp/avers_dataset
  python -m avers.dataset.cli train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l
  python -m avers.dataset.cli public list
  python -m avers.dataset.cli public download --dataset masala-chai
  python -m avers.dataset.cli public mix --synthetic /tmp/gost/dataset.yaml --output /tmp/mixed
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


def cmd_public(args):
    """Public datasets commands."""
    from avers.dataset.public_datasets import PublicDatasetLoader, get_recommended_training_strategy, PUBLIC_DATASETS
    
    loader = PublicDatasetLoader()
    
    if args.public_command == "list":
        datasets = loader.list_datasets(gost_compatible_only=args.gost_only)
        print(f"\nFound {len(datasets)} public datasets:\n")
        print(f"{'Name':<30} {'Images':<8} {'Classes':<8} {'GOST':<6} {'License'}")
        print("-"*80)
        for ds in datasets:
            gost_mark = "✓" if ds.gost_compatible else ""
            print(f"{ds.name:<30} {ds.num_images:<8} {ds.num_classes:<8} {gost_mark:<6} {ds.license}")
            print(f"  {ds.description}")
            print(f"  URL: {ds.url}")
            print()
        
        if not args.gost_only:
            print("\nGOST-compatible only:")
            print("  python -m avers.dataset.cli public list --gost-only")
    
    elif args.public_command == "info":
        info = loader.get_dataset_info(args.dataset)
        if not info:
            print(f"Dataset {args.dataset} not found")
            print(f"Available: {', '.join(PUBLIC_DATASETS.keys())}")
            return 1
        
        print(f"\nDataset: {info.name}")
        print(f"Description: {info.description}")
        print(f"URL: {info.url}")
        print(f"Images: {info.num_images}, Classes: {info.num_classes}")
        print(f"Classes: {', '.join(info.classes)}")
        print(f"License: {info.license}, Format: {info.format}")
        print(f"GOST compatible: {info.gost_compatible}")
        if info.paper_url:
            print(f"Paper: {info.paper_url}")
        print(f"\nNotes: {info.notes}")
        print("\n" + loader.download_instructions(args.dataset))
    
    elif args.public_command == "download":
        print(loader.download_instructions(args.dataset))
        print(f"\nAfter download, convert to GOST:")
        print(f"  python -m avers.dataset.cli public convert --dataset {args.dataset} --input /path/to/dataset --output /tmp/gost_converted")
    
    elif args.public_command == "convert":
        output = loader.convert_to_gost(args.dataset, Path(args.input), Path(args.output))
        print(f"Converted mapping saved to {output}")
        print(f"  Mapping: {Path(args.output) / 'public_to_gost_mapping.json'}")
    
    elif args.public_command == "mix":
        # Parse public datasets list: name:path,name:path
        public_list = []
        if args.public:
            for item in args.public.split(","):
                if ":" in item:
                    name, path = item.split(":", 1)
                    public_list.append((name.strip(), Path(path.strip())))
        
        yaml_path = loader.create_mixed_dataset_config(
            synthetic_dataset_yaml=Path(args.synthetic),
            public_datasets=public_list,
            output_path=Path(args.output)
        )
        print(f"Mixed dataset created: {yaml_path}")
    
    elif args.public_command == "strategy":
        print(get_recommended_training_strategy())
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="avers-dataset",
        description="AVERS ГОСТ УГО Dataset Tools + Public Datasets"
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
    
    # Public datasets
    public_parser = subparsers.add_parser("public", help="Public datasets for training")
    public_sub = public_parser.add_subparsers(dest="public_command", help="Public dataset command")
    
    public_list = public_sub.add_parser("list", help="List public datasets")
    public_list.add_argument("--gost-only", action="store_true", help="Only GOST-compatible")
    
    public_info = public_sub.add_parser("info", help="Info about dataset")
    public_info.add_argument("--dataset", type=str, required=True, help="Dataset name")
    
    public_download = public_sub.add_parser("download", help="Download instructions")
    public_download.add_argument("--dataset", type=str, required=True, help="Dataset name")
    
    public_convert = public_sub.add_parser("convert", help="Convert public dataset to GOST")
    public_convert.add_argument("--dataset", type=str, required=True, help="Dataset name")
    public_convert.add_argument("--input", "-i", type=str, required=True, help="Input dataset path")
    public_convert.add_argument("--output", "-o", type=str, required=True, help="Output path")
    
    public_mix = public_sub.add_parser("mix", help="Create mixed dataset (synthetic + public)")
    public_mix.add_argument("--synthetic", type=str, required=True, help="Path to synthetic dataset.yaml")
    public_mix.add_argument("--public", type=str, help="Public datasets list: name:path,name:path")
    public_mix.add_argument("--output", "-o", type=str, required=True, help="Output dir for mixed")
    
    public_strategy = public_sub.add_parser("strategy", help="Show recommended training strategy")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\nPublic datasets:")
        print("  python -m avers.dataset.cli public list")
        print("  python -m avers.dataset.cli public strategy")
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
        elif args.command == "public":
            return cmd_public(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
