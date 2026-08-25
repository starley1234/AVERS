"""
AVERS Pipeline - Main Entry Points

Provides both:
- Legacy Pipeline (backwards compatible)
- ProductionPipeline (recommended)
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np

from avers.core.types import AVERSManifest
from avers.core.pipeline import aversPipeline as LegacyPipeline
from avers.core.validators import (
    ProductionPipeline,
    PipelineResult,
    process_schematic_production,
    load_and_process,
)
from avers.config import AVERSConfig, DEFAULT_CONFIG


# Backwards compatible function
def process_schematic(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[AVERSConfig] = None,
    production: bool = True,
) -> Union[AVERSManifest, PipelineResult]:
    """
    Process a schematic image.
    
    Args:
        image_path: Path to input image
        output_path: Path for output file
        config: Optional configuration
        production: Use production pipeline (recommended)
        
    Returns:
        AVERSManifest (legacy) or PipelineResult (production)
    """
    if production:
        result = load_and_process(image_path, output_path, config)
        
        # Log warnings
        if result.warnings:
            import logging
            logger = logging.getLogger("avers")
            for warning in result.warnings:
                logger.warning(warning)
        
        return result
    else:
        # Legacy pipeline
        pipeline = LegacyPipeline(config=config)
        manifest = pipeline.process(image_path, output_path)
        
        if output_path:
            manifest.save(output_path)
        
        return manifest


__all__ = [
    "ProductionPipeline",
    "PipelineResult",
    "process_schematic_production",
    "load_and_process",
    "process_schematic",
]
