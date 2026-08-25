"""Stage 3: OCR stub module."""

from typing import List, Dict, Optional, Tuple
import re

# OCR engine will be loaded lazily
_ocr_engine = None

# Regex patterns for validation (ГОСТ designations)
PATTERNS = {
    "connector": re.compile(r"^[ХX]\d+$"),
    "connector_alt": re.compile(r"^СНЦ\d+.*$"),
    "connector_rm": re.compile(r"^2РМ\d+.*$"),
    "pin_number": re.compile(r"^\d+$"),
    "pin_letter": re.compile(r"^[A-ZА-Я]$"),
    "wire_type": re.compile(r"^БПВЛ(-\d+(\.\d+)?)?$"),
    "wire_mgtf": re.compile(r"^МГТФ.*"),
    "voltage": re.compile(r"^[+-]?\d+В?$"),
}


def load_ocr_engine(lang: str = "ru", use_angle_cls: bool = True):
    """Load PaddleOCR engine."""
    global _ocr_engine
    
    if _ocr_engine is not None:
        return _ocr_engine
    
    # Placeholder - would load actual PaddleOCR
    # from paddleocr import PaddleOCR
    # ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, show_log=False)
    
    _ocr_engine = {"loaded": True, "lang": lang}
    return _ocr_engine


def recognize_text_in_tile(image, use_angle_cls: bool = True) -> List[Dict]:
    """
    Recognize text in a tile image.

    Args:
        image: Tile image array
        use_angle_cls: Enable text direction classification

    Returns:
        List of {'text': str, 'bbox': tuple, 'confidence': float, 'angle': float}
    """
    # Placeholder - would run actual OCR
    return []


def validate_designation(text: str, designation_type: str) -> bool:
    """
    Validate text against regex patterns.

    Args:
        text: Text to validate
        designation_type: Type of designation ('connector', 'pin', 'wire')

    Returns:
        True if text matches expected pattern
    """
    pattern = PATTERNS.get(designation_type)
    if pattern is None:
        return False
    
    return bool(pattern.match(text))


def normalize_designation(text: str) -> str:
    """
    Normalize designation text (fix common OCR errors).

    Args:
        text: Raw OCR text

    Returns:
        Normalized text
    """
    # Common normalizations
    text = text.strip()
    
    # Fix common character confusions
    replacements = {
        'Х': 'Х',  # Cyrillic vs Latin
        'x': 'Х',
        'X': 'Х',
    }
    
    for old, new in replacements.items():
        if old in text and new not in text:
            text = text.replace(old, new)
    
    return text


def get_text_labels(image, tiles: Optional[List] = None) -> List[Dict]:
    """
    Get all text labels from image.

    Args:
        image: Full image or tile images
        tiles: Optional list of tiles

    Returns:
        List of {'text': str, 'coord': (x, y), 'bbox': tuple}
    """
    # Placeholder
    return []
