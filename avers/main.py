#!/usr/bin/env python3
"""
AVERS — Automated Vectorization and Recognition of Schematics

Main entry point for the command-line interface.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from avers import aversPipeline, process_schematic
from avers.config import AVERSConfig, get_default_config
from avers.core.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="avers",
        description="AVERS — Automated Vectorization and Recognition of Schematics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.tif --output result.json
  %(prog)s schematic.png -o nets.xml --format xml
  %(prog)s board.tif --config config.yaml --visualize
        """,
    )

    # Input
    parser.add_argument(
        "input",
        type=Path,
        help="Input schematic image (TIF, PNG, PDF)",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "xml"],
        default="json",
        help="Output format (default: json)",
    )

    # Configuration
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Configuration YAML file",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for SAHI slicing (default: 1024)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Tile overlap ratio (default: 0.2)",
    )

    # Control
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        help="Skip specified pipeline stages",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization outputs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # VLM settings
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM arbitration",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model name",
    )
    parser.add_argument(
        "--max-vlm-calls",
        type=int,
        default=50,
        help="Maximum VLM calls (default: 50)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("avers", level=log_level)

    # Load or create configuration
    if args.config and args.config.exists():
        config = AVERSConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")

    # Override config from CLI args
    config.slicing.tile_size = args.tile_size
    config.slicing.overlap_ratio = args.overlap
    config.vlm_arbitrator.enabled = not args.no_vlm
    config.vlm_arbitrator.model_name = args.vlm_model
    config.vlm_arbitrator.max_vlm_calls = args.max_vlm_calls

    # Generate default output path if not specified
    if args.output is None:
        output_path = args.input.with_suffix(f".{args.format}")
    else:
        output_path = args.output

    try:
        # Run pipeline
        logger.info(f"Processing: {args.input}")
        
        manifest = process_schematic(
            image_path=args.input,
            output_path=output_path,
            config=config,
        )

        # Summary
        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Components: {len(manifest.components)}")
        logger.info(f"Nets: {len(manifest.nets)}")
        logger.info(f"Issues requiring review: {len(manifest.human_review_required)}")
        logger.info("=" * 60)

        # Return exit code based on issues
        if manifest.human_review_required:
            logger.warning(
                f"{len(manifest.human_review_required)} issue(s) require human review. "
                "Check human_review_required in output."
            )
            return 2  # Partial success

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
