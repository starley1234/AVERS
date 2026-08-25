#!/usr/bin/env python3
"""
AVERS — Automated Vectorization and Recognition of Schematics

Main entry point for the command-line interface.

Commands:
  avers process input.tif --output result.json  # Process schematic
  avers web --port 8000                         # Start Web UI validator
  avers dataset generate ...                    # Generate synthetic dataset
  avers dataset train ...                       # Train RT-DETR/YOLO
  avers rag query ...                           # Vision RAG query
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from avers import aversPipeline
from avers.config import AVERSConfig, get_default_config
from avers.core.logger import setup_logger


def create_main_parser() -> argparse.ArgumentParser:
    """Create main parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="avers",
        description="АВЕРС — Автоматическая Векторизация и Распознавание Схем",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s process input.tif --output result.json
  %(prog)s web --port 8000 --host 0.0.0.0
  %(prog)s dataset generate --output /tmp/avers_dataset --num-train 1000
  %(prog)s dataset train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l
  %(prog)s dataset preview --output /tmp/preview --num 20
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команда")
    
    # Process command (original) - supports images and PDF
    process_parser = subparsers.add_parser("process", help="Обработать схему (изображение или PDF)")
    process_parser.add_argument("input", type=Path, help="Входное изображение (TIF, PNG, JPG, PDF)")
    process_parser.add_argument("-o", "--output", type=Path, help="Выходной файл")
    process_parser.add_argument("-f", "--format", choices=["json", "xml"], default="json", help="Формат вывода")
    process_parser.add_argument("-c", "--config", type=Path, help="YAML конфиг")
    process_parser.add_argument("--tile-size", type=int, default=1024, help="Размер тайла SAHI")
    process_parser.add_argument("--overlap", type=float, default=0.2, help="Перекрытие тайлов")
    process_parser.add_argument("--skip-stages", nargs="+", choices=["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"], help="Пропустить стадии")
    process_parser.add_argument("--visualize", action="store_true", help="Визуализация")
    process_parser.add_argument("--debug", action="store_true", help="Debug лог")
    process_parser.add_argument("--no-vlm", action="store_true", help="Отключить VLM")
    process_parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM модель")
    process_parser.add_argument("--max-vlm-calls", type=int, default=50, help="Макс VLM вызовов")
    # PDF specific
    process_parser.add_argument("--pdf-page", type=int, default=0, help="Страница PDF для обработки (0-indexed, default: 0)")
    process_parser.add_argument("--pdf-all", action="store_true", help="Обработать все страницы PDF и объединить")
    
    # Web UI command
    web_parser = subparsers.add_parser("web", help="Запустить Web UI валидатор")
    web_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    web_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    web_parser.add_argument("--reload", action="store_true", help="Auto-reload")
    web_parser.add_argument("--workers", type=int, default=1, help="Workers")
    
    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Инструменты датасета ГОСТ УГО")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", help="Dataset command")
    
    gen_parser = dataset_sub.add_parser("generate", help="Сгенерировать синтетический датасет")
    gen_parser.add_argument("--output", "-o", type=str, default="/tmp/avers_dataset", help="Output dir")
    gen_parser.add_argument("--num-train", type=int, default=1000, help="Train images")
    gen_parser.add_argument("--num-val", type=int, default=200, help="Val images")
    gen_parser.add_argument("--num-test", type=int, default=100, help="Test images")
    gen_parser.add_argument("--image-size", type=int, default=1024, help="Image size")
    
    train_parser = dataset_sub.add_parser("train", help="Обучить модель детекции")
    train_parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    train_parser.add_argument("--model", type=str, default="rtdetr-l", help="Model: yolo11x, rtdetr-l, etc")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--project", type=str, default="/tmp/avers_runs")
    train_parser.add_argument("--pretrained", type=str, default=None)
    
    preview_parser = dataset_sub.add_parser("preview", help="Превью синтетики")
    preview_parser.add_argument("--output", "-o", type=str, default="/tmp/avers_preview")
    preview_parser.add_argument("--num", type=int, default=10)
    preview_parser.add_argument("--size", type=int, default=1024)
    
    export_parser = dataset_sub.add_parser("export", help="Экспорт датасета")
    export_parser.add_argument("--input", "-i", type=str, required=True)
    export_parser.add_argument("--output", "-o", type=str, required=True)
    export_parser.add_argument("--format", "-f", type=str, default="coco", choices=["coco"])
    
    # Public datasets (NEW)
    public_parser = dataset_sub.add_parser("public", help="Публичные датасеты для обучения")
    public_sub = public_parser.add_subparsers(dest="public_command")
    
    public_list = public_sub.add_parser("list", help="Список публичных датасетов")
    public_list.add_argument("--gost-only", action="store_true", help="Только GOST-совместимые")
    
    public_info = public_sub.add_parser("info", help="Инфо о датасете")
    public_info.add_argument("--dataset", type=str, required=True, help="Имя датасета")
    
    public_download = public_sub.add_parser("download", help="Инструкции по скачиванию")
    public_download.add_argument("--dataset", type=str, required=True, help="Имя датасета")
    
    public_convert = public_sub.add_parser("convert", help="Конвертировать в GOST")
    public_convert.add_argument("--dataset", type=str, required=True)
    public_convert.add_argument("--input", "-i", type=str, required=True)
    public_convert.add_argument("--output", "-o", type=str, required=True)
    
    public_mix = public_sub.add_parser("mix", help="Смешанный датасет synthetic+public")
    public_mix.add_argument("--synthetic", type=str, required=True, help="Путь к synthetic dataset.yaml")
    public_mix.add_argument("--public", type=str, help="Список: name:path,name:path")
    public_mix.add_argument("--output", "-o", type=str, required=True)
    
    public_strategy = public_sub.add_parser("strategy", help="Рекомендуемая стратегия обучения")
    
    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Vision RAG инструменты")
    rag_sub = rag_parser.add_subparsers(dest="rag_command")
    
    rag_query = rag_sub.add_parser("query", help="Запрос к RAG")
    rag_query.add_argument("--text", type=str, help="Текстовый запрос")
    rag_query.add_argument("--image", type=str, help="Путь к ROI изображению")
    rag_query.add_argument("--top-k", type=int, default=5)
    
    rag_index = rag_sub.add_parser("index", help="Индексировать пример")
    rag_index.add_argument("--image", type=str, required=True, help="Путь к изображению")
    rag_index.add_argument("--label", type=str, required=True)
    rag_index.add_argument("--description", type=str, default="")
    
    rag_stats = rag_sub.add_parser("stats", help="Статистика RAG")
    
    # Legacy: if no subcommand, treat first arg as input file
    parser.add_argument("legacy_input", nargs="?", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("-o", "--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("-f", "--format", choices=["json", "xml"], default="json", help=argparse.SUPPRESS)
    parser.add_argument("-c", "--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tile-size", type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument("--overlap", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-vlm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help=argparse.SUPPRESS)
    parser.add_argument("--max-vlm-calls", type=int, default=50, help=argparse.SUPPRESS)
    
    return parser


def cmd_process(args) -> int:
    """Process schematic."""
    # Handle legacy input
    input_path = getattr(args, 'input', None) or getattr(args, 'legacy_input', None)
    if not input_path:
        print("Error: input file required", file=sys.stderr)
        return 1
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("avers", level=log_level)
    
    if args.config and args.config.exists():
        config = AVERSConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = get_default_config()
    
    config.slicing.tile_size = args.tile_size
    config.slicing.overlap_ratio = args.overlap
    config.vlm_arbitrator.enabled = not args.no_vlm
    config.vlm_arbitrator.model_name = args.vlm_model
    config.vlm_arbitrator.max_vlm_calls = args.max_vlm_calls
    
    output_path = args.output or input_path.with_suffix(f".{args.format}")
    
    try:
        from avers.pipeline import load_and_process
        from avers.utils.pdf_loader import is_pdf
        
        # Detect PDF
        is_pdf_file = is_pdf(input_path) or input_path.suffix.lower() == ".pdf"
        
        if is_pdf_file:
            logger.info(f"Detected PDF: {input_path}")
            if getattr(args, 'pdf_all', False):
                logger.info(f"Processing all pages from PDF")
                result = load_and_process(input_path, output_path, config, pdf_process_all=True)
            else:
                pdf_page = getattr(args, 'pdf_page', 0)
                logger.info(f"Processing PDF page {pdf_page}")
                result = load_and_process(input_path, output_path, config, pdf_page=pdf_page)
        else:
            logger.info(f"Processing: {input_path}")
            result = load_and_process(input_path, output_path, config)
        
        logger.info("="*60)
        logger.info("Processing Complete")
        logger.info(f"Output: {output_path}")
        logger.info(f"Components: {len(result.manifest.components)}")
        logger.info(f"Nets: {len(result.manifest.nets)}")
        logger.info(f"Issues: {len(result.manifest.human_review_required)}")
        logger.info("="*60)
        
        if result.manifest.human_review_required:
            return 2
        return 0
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


def cmd_web(args) -> int:
    """Start Web UI."""
    logger = setup_logger("avers.web", level="INFO")
    logger.info(f"Starting AVERS Web UI on {args.host}:{args.port}")
    logger.info(f"Open http://{args.host}:{args.port} in browser")
    
    try:
        import uvicorn
        uvicorn.run(
            "avers.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
        )
    except ImportError:
        print("uvicorn not installed. Install: pip install uvicorn fastapi", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("Web server stopped")
        return 0
    
    return 0


def cmd_dataset(args) -> int:
    """Dataset tools."""
    if not args.dataset_command:
        print("Dataset command required: generate, train, preview, export", file=sys.stderr)
        return 1
    
    if args.dataset_command == "generate":
        from avers.dataset.generator import generate_full_dataset
        generate_full_dataset(
            output_dir=args.output,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            image_size=args.image_size,
        )
        print(f"Dataset generated at {args.output}")
        return 0
    
    elif args.dataset_command == "train":
        from avers.dataset.train import train_yolo, train_rtdetr
        data_yaml = Path(args.data)
        
        if "yolo" in args.model.lower():
            model_name = args.pretrained or "yolo11x.pt"
            train_yolo(data_yaml, model_name, args.epochs, args.imgsz, args.batch, args.device, args.project)
        else:
            model_name = args.pretrained or "rtdetr-l.pt"
            train_rtdetr(data_yaml, model_name, args.epochs, args.imgsz, args.batch, args.device, args.project)
        return 0
    
    elif args.dataset_command == "preview":
        from avers.dataset.synthetic import GOSTGenerator, SyntheticConfig
        import cv2
        
        config = SyntheticConfig(image_size=args.size)
        gen = GOSTGenerator(config)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        
        for i in range(args.num):
            img, anns = gen.generate_realistic_schematic()
            cv2.imwrite(str(out / f"preview_{i:04d}.jpg"), img)
            print(f"Saved preview_{i:04d}.jpg with {len(anns)} annotations")
        return 0
    
    elif args.dataset_command == "export":
        from avers.dataset.export import COCOExporter
        COCOExporter.from_yolo_dataset(Path(args.input), Path(args.output))
        print(f"Exported to {args.output}")
        return 0
    
    elif args.dataset_command == "public":
        from avers.dataset.public_datasets import PublicDatasetLoader, get_recommended_training_strategy
        
        loader = PublicDatasetLoader()
        
        if args.public_command == "list":
            datasets = loader.list_datasets(gost_compatible_only=getattr(args, 'gost_only', False))
            print(f"\nFound {len(datasets)} public datasets:\n")
            print(f"{'Name':<30} {'Images':<8} {'Classes':<8} {'GOST':<6} {'License'}")
            print("-"*80)
            for ds in datasets:
                gost_mark = "✓" if ds.gost_compatible else ""
                print(f"{ds.name:<30} {ds.num_images:<8} {ds.num_classes:<8} {gost_mark:<6} {ds.license}")
                print(f"  {ds.description[:70]}")
                print(f"  URL: {ds.url}")
                print()
        
        elif args.public_command == "info":
            info = loader.get_dataset_info(args.dataset)
            if not info:
                print(f"Dataset {args.dataset} not found")
                return 1
            print(f"\nDataset: {info.name}\nDescription: {info.description}\nURL: {info.url}\nImages: {info.num_images}, Classes: {info.num_classes}\nClasses: {', '.join(info.classes)}\nLicense: {info.license}\nGOST: {info.gost_compatible}\nNotes: {info.notes}")
            print("\n" + loader.download_instructions(args.dataset))
        
        elif args.public_command == "download":
            print(loader.download_instructions(args.dataset))
        
        elif args.public_command == "convert":
            out = loader.convert_to_gost(args.dataset, Path(args.input), Path(args.output))
            print(f"Converted to {out}")
        
        elif args.public_command == "mix":
            public_list = []
            if getattr(args, 'public', None):
                for item in args.public.split(","):
                    if ":" in item:
                        name, path = item.split(":", 1)
                        public_list.append((name.strip(), Path(path.strip())))
            yaml_path = loader.create_mixed_dataset_config(Path(args.synthetic), public_list, Path(args.output))
            print(f"Mixed dataset: {yaml_path}")
        
        elif args.public_command == "strategy":
            print(get_recommended_training_strategy())
        
        else:
            print("Public command required: list, info, download, convert, mix, strategy")
            return 1
        
        return 0
    
    return 1


def cmd_rag(args) -> int:
    """RAG tools."""
    from avers.rag import VisionRAG
    import cv2
    
    rag = VisionRAG()
    
    if args.rag_command == "query":
        image = None
        if args.image:
            image = cv2.imread(args.image)
        
        response = rag.query(image=image, text=args.text, top_k=args.top_k)
        print(f"Query: {args.text}")
        print(f"Found {len(response.results)} results:")
        for r in response.results:
            print(f"  - {r.entry.label} (score: {r.score:.3f}): {r.entry.description}")
        if response.vlm_answer:
            print(f"VLM answer: {response.vlm_answer}")
        return 0
    
    elif args.rag_command == "index":
        image = cv2.imread(args.image)
        if image is None:
            print(f"Failed to load image: {args.image}", file=sys.stderr)
            return 1
        entry_id = rag.add_example(image, label=args.label, description=args.description)
        print(f"Indexed {entry_id}: {args.label}")
        return 0
    
    elif args.rag_command == "stats":
        stats = rag.stats()
        print(f"RAG stats: {stats}")
        return 0
    
    print("RAG command required: query, index, stats", file=sys.stderr)
    return 1


def main() -> int:
    parser = create_main_parser()
    args = parser.parse_args()
    
    # Handle legacy mode: if legacy_input provided and no command, treat as process
    if not args.command and getattr(args, 'legacy_input', None):
        args.command = "process"
        args.input = args.legacy_input
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "process":
        return cmd_process(args)
    elif args.command == "web":
        return cmd_web(args)
    elif args.command == "dataset":
        return cmd_dataset(args)
    elif args.command == "rag":
        return cmd_rag(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
