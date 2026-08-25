"""Stage 2: УГО Detection module."""

from avers.stages.stage2_detection.detector import (
    YOLODetector,
    SlicingDetector,
    DetectionConfig,
    DetectionResult,
    DETECTION_CLASS_MAP,
    CLASS_NAMES,
    group_detections_into_components,
    detect_components,
)
from avers.stages.stage1_slicing import Tile, DetectionBox

__all__ = [
    "YOLODetector",
    "SlicingDetector", 
    "DetectionConfig",
    "DetectionResult",
    "DETECTION_CLASS_MAP",
    "CLASS_NAMES",
    "group_detections_into_components",
    "detect_components",
    "Tile",
    "DetectionBox",
]
