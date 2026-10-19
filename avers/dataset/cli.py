#!/usr/bin/env python3
"""
CLI для работы с датасетом ГОСТ УГО + публичные датасеты.

Использование:
  python -m avers.dataset.cli generate-v2 --output data/synth_v2 --num-train 2000 --num-val 200
  python -m avers.dataset.cli generate --num 1000 --output /tmp/avers_dataset   (устарел, 6 классов)
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


def cmd_generate_v2(args):
    """Датасет YOLO из генератора схем v2 (все 31 класс, эталонные netlist)."""
    from avers.dataset.schematic_gen import GenConfig, export_yolo_dataset

    cfg = GenConfig(ppm_range=(args.ppm_min, args.ppm_max))
    yaml_path = export_yolo_dataset(
        args.output, num_train=args.num_train, num_val=args.num_val, seed=args.seed,
        tile=args.tile, overlap=args.overlap, ext=args.format, degrade_p=args.degrade,
        workers=args.workers, config=cfg, progress=True,
    )
    print(f"Датасет: {yaml_path}")
    print(f"Статистика: {Path(args.output) / 'stats.json'}")
    print(f"Обучение: python -m avers.dataset.cli train --data {yaml_path} --model yolo11s "
          f"--imgsz {args.tile or 1024} --device 0")
    return 0


def cmd_train(args):
    """Обучение модели."""
    from avers.dataset.train import train_yolo, train_rtdetr
    
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        return 1
    
    model_type = args.model.lower()
    
    if "yolo" in model_type:
        # yolo11n/s/m/l/x -> yolo11s.pt и т.д. (раньше любой «не x» становился yolo11m)
        default = f"{model_type}.pt" if model_type[-1:] in "nsmlx" and model_type[-2:-1].isdigit() else "yolo11m.pt"
        model_name = args.pretrained or default
        if args.resume and not args.pretrained:
            logger.error("Для --resume нужен --pretrained с путём к last.pt прерванного прогона")
            return 1
        train_yolo(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            resume=args.resume,
        )
    elif "rtdetr" in model_type or "detr" in model_type:
        model_name = args.pretrained or ("rtdetr-x.pt" if "x" in model_type else "rtdetr-l.pt")
        if args.resume and not args.pretrained:
            logger.error("Для --resume нужен --pretrained с путём к last.pt прерванного прогона")
            return 1
        train_rtdetr(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            resume=args.resume,
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
        print("\nAfter download, convert to GOST:")
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
    
    # Generate v2 (schematic_gen)
    g2 = subparsers.add_parser("generate-v2", help="Synthetic schematics v2: YOLO dataset + reference netlists")
    g2.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    g2.add_argument("--num-train", type=int, default=2000, help="Train sheets (each gives several tiles)")
    g2.add_argument("--num-val", type=int, default=200, help="Val sheets")
    g2.add_argument("--seed", type=int, default=0)
    g2.add_argument("--tile", type=int, default=1024, help="Tile size (0 = whole sheet)")
    g2.add_argument("--overlap", type=float, default=0.2)
    g2.add_argument("--format", choices=["png", "jpg"], default="png")
    g2.add_argument("--degrade", type=float, default=0.7, help="Share of sheets with scan effects")
    g2.add_argument("--ppm-min", type=float, default=4.0, help="Min scale, px/mm")
    g2.add_argument("--ppm-max", type=float, default=12.0, help="Max scale, px/mm (300 dpi = 11.8)")
    g2.add_argument("--workers", type=int, default=4)

    # Train
    train_parser = subparsers.add_parser("train", help="Train detection model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    train_parser.add_argument("--model", type=str, default="yolo11s",
                              help="Model: yolo11n/s/m/l/x, rtdetr-l, rtdetr-x")
    train_parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    train_parser.add_argument("--batch", type=int, default=8, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=1024, help="Image size (= tile size of generate-v2)")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device")
    train_parser.add_argument("--project", type=str, default="/tmp/avers_runs", help="Project dir")
    train_parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weights")
    train_parser.add_argument("--resume", action="store_true",
                              help="Продолжить прерванный прогон (--pretrained = путь к last.pt)")
    
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
        elif args.command == "generate-v2":
            return cmd_generate_v2(args)
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
