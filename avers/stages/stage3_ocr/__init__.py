"""Stage 3: OCR module."""

from avers.stages.stage3_ocr.ocr_engine import (
    PaddleOCREngine,
    OCRConfig,
    OCRResult,
    TextLabel,
    TextAssociationEngine,
    DESIGNATION_PATTERNS,
    classify_text,
    normalize_designation,
    validate_designation,
    recognize_schematic_text,
)

__all__ = [
    "PaddleOCREngine",
    "OCRConfig",
    "OCRResult", 
    "TextLabel",
    "TextAssociationEngine",
    "DESIGNATION_PATTERNS",
    "classify_text",
    "normalize_designation",
    "validate_designation",
    "recognize_schematic_text",
]
