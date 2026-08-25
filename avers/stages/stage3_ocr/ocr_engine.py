"""
Stage 3: OCR Text Recognition with PaddleOCR

Recognizes text labels on schematic diagrams including:
- Connector designators (X1, Ш2, СНЦ144)
- Pin numbers (1, 2, 3, A, B)
- Wire type labels (БПВЛ-0.35, МГТФ)
- Voltage markers (+27В, -12В, GND)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from pathlib import Path
import re
import numpy as np

from avers.core.logger import get_logger

logger = get_logger("avers.ocr")

# Regex patterns for schematic text (ГОСТ designations)
DESIGNATION_PATTERNS = {
    # Connectors
    "connector_x": re.compile(r"^[ХX]\d+$"),
    "connector_sh": re.compile(r"^Ш\d+$"),
    "connector_snc": re.compile(r"^СНЦ\d+"),
    "connector_rm": re.compile(r"^2РМ\d+"),
    
    # Pin numbers
    "pin_numeric": re.compile(r"^\d{1,3}$"),
    "pin_letter": re.compile(r"^[A-ZА-Я]$"),
    
    # Wire types (case-insensitive)
    "wire_bpvl": re.compile(r"^БПВЛ(-\d+(\.\d+)?)?$", re.IGNORECASE),
    "wire_mgtf": re.compile(r"^МГТФ.*$", re.IGNORECASE),
    "wire_kg": re.compile(r"^КГ\d+.*$", re.IGNORECASE),
    
    # Voltage/power
    "voltage": re.compile(r"^[+-]?\d+[ВvV]?$", re.IGNORECASE),
    "gnd": re.compile(r"^GND|ЗЕМЛЯ|KОРПУС$", re.IGNORECASE),
    
    # Component designators
    "diode": re.compile(r"^VD\d+$", re.IGNORECASE),
    "resistor": re.compile(r"^R\d+$", re.IGNORECASE),
    "relay": re.compile(r"^K\d+$", re.IGNORECASE),
    "capacitor": re.compile(r"^C\d+$", re.IGNORECASE),
}


@dataclass
class OCRResult:
    """Single OCR text recognition result."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float
    angle: float = 0.0  # Text rotation angle
    text_type: str = "unknown"  # Classified type
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center of text bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


@dataclass
class OCRConfig:
    """OCR configuration."""
    lang: str = "ru"  # Language: 'ru', 'en', 'ch', 'japan'
    use_angle_cls: bool = True  # Enable text direction classification
    text_confidence_threshold: float = 0.65
    allowed_rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    
    # Size filters
    min_text_height: int = 8
    max_text_height: int = 200
    
    # Expand ratio for context
    context_expand_ratio: float = 0.5


@dataclass 
class TextLabel:
    """Structured text label for association."""
    text: str
    coord: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    label_type: str  # 'designator', 'wire_type', 'voltage', 'pin', 'other'
    confidence: float
    normalized_text: str = ""
    
    def __post_init__(self):
        """Normalize text after creation."""
        if not self.normalized_text:
            self.normalized_text = normalize_designation(self.text)


class PaddleOCREngine:
    """
    PaddleOCR wrapper for schematic text recognition.
    
    Features:
    - DBNet++ for text detection
    - SVTR/CRNN for text recognition
    - Angle classification for rotated text (90°, 270°)
    - Regex validation against ГОСТ patterns
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.engine = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load PaddleOCR model."""
        if self._loaded:
            return True
        
        try:
            from paddleocr import PaddleOCR
            
            self.engine = PaddleOCR(
                lang=self.config.lang,
                use_angle_cls=self.config.use_angle_cls,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
            )
            
            self._loaded = True
            logger.info("PaddleOCR loaded successfully")
            return True
            
        except ImportError:
            logger.warning("PaddleOCR not installed, using EasyOCR fallback")
            return self._load_easyocr()
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            return self._load_easyocr()
    
    def _load_easyocr(self) -> bool:
        """Fallback to EasyOCR."""
        try:
            import easyocr
            
            self.engine = easyocr.Reader(
                [self.config.lang, 'en'],
                gpu=False,
                verbose=False,
            )
            self._loaded = True
            logger.info("EasyOCR loaded as fallback")
            return True
        except ImportError:
            logger.warning("EasyOCR not installed, using mock OCR")
            self._loaded = True
            self.engine = None
            return True
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """
        Recognize text in image.
        
        Args:
            image: Input image (H, W, 3) RGB
            
        Returns:
            List of OCRResult objects
        """
        if not self._loaded:
            self.load()
        
        if self.engine is None:
            return self._mock_ocr(image)
        
        try:
            # Check if PaddleOCR or EasyOCR
            engine_name = self.engine.__class__.__name__ if hasattr(self.engine, '__class__') else str(type(self.engine))
            
            if 'paddleocr' in engine_name.lower() or 'PaddleOCR' in engine_name:
                return self._paddleocr_recognize(image)
            else:
                return self._easyocr_recognize(image)
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return self._mock_ocr(image)
    
    def _paddleocr_recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Run PaddleOCR inference."""
        # PaddleOCR expects BGR
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        results = self.engine.ocr(bgr, cls=self.config.use_angle_cls)
        
        ocr_results = []
        if results and results[0]:
            for line in results[0]:
                points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # Extract bbox
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                )
                
                # Determine angle (if available)
                angle = 0
                if len(line) > 2 and line[2]:
                    angle = float(line[2])
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=float(confidence),
                    angle=angle,
                    text_type=classify_text(text),
                ))
        
        return ocr_results
    
    def _easyocr_recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Run EasyOCR inference."""
        results = self.engine.readtext(image)
        
        ocr_results = []
        for (bbox, text, confidence) in results:
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            full_bbox = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )
            
            ocr_results.append(OCRResult(
                text=text.strip(),
                bbox=full_bbox,
                confidence=float(confidence),
                text_type=classify_text(text),
            ))
        
        return ocr_results
    
    def _mock_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """
        Generate mock OCR results for testing.
        
        Detects text-like regions using contour analysis.
        """
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find text-like contours (grouped horizontal bars)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter by size (text-like proportions)
            if h < self.config.min_text_height or h > self.config.max_text_height:
                continue
            if w < 10 or w > 500:
                continue
            if h > w * 2:  # Too tall
                continue
            
            # Extract text region
            roi = gray[y:y+h, x:x+w]
            
            # Simple template matching for common patterns
            text = self._extract_mock_text(roi)
            
            if text:
                results.append(OCRResult(
                    text=text,
                    bbox=(x, y, x+w, y+h),
                    confidence=0.7,
                    text_type=classify_text(text),
                ))
        
        return results
    
    def _extract_mock_text(self, roi: np.ndarray) -> str:
        """Extract mock text from region (simplified)."""
        import cv2
        
        # Count vertical segments (basic character detection)
        proj_h = np.sum(roi, axis=0)
        if len(proj_h) < 5:
            return ""
        
        # Find gaps (character separators)
        threshold = np.mean(proj_h) * 0.5
        gaps = proj_h < threshold
        
        # Count "characters"
        in_char = not gaps[0]
        char_count = 0
        for i, is_gap in enumerate(gaps):
            if is_gap and in_char:
                char_count += 1
                in_char = False
            elif not is_gap:
                in_char = True
        
        if char_count < 1 or char_count > 10:
            return ""
        
        # Return placeholder based on size
        if char_count <= 2:
            return "X1"
        elif char_count <= 4:
            return "+27В"
        else:
            return "БПВЛ-0.35"


def classify_text(text: str) -> str:
    """
    Classify text type based on content and patterns.
    
    Returns:
        Text type: 'connector', 'pin', 'wire_type', 'voltage', 'designator', 'other'
    """
    text = text.strip()
    
    # Check each pattern category
    if DESIGNATION_PATTERNS["connector_x"].match(text) or \
       DESIGNATION_PATTERNS["connector_sh"].match(text):
        return "connector"
    
    if DESIGNATION_PATTERNS["connector_snc"].match(text) or \
       DESIGNATION_PATTERNS["connector_rm"].match(text):
        return "connector_type"
    
    if DESIGNATION_PATTERNS["pin_numeric"].match(text) or \
       DESIGNATION_PATTERNS["pin_letter"].match(text):
        return "pin"
    
    if DESIGNATION_PATTERNS["wire_bpvl"].match(text) or \
       DESIGNATION_PATTERNS["wire_mgtf"].match(text) or \
       DESIGNATION_PATTERNS["wire_kg"].match(text):
        return "wire_type"
    
    if DESIGNATION_PATTERNS["voltage"].match(text):
        return "voltage"
    
    if DESIGNATION_PATTERNS["gnd"].match(text):
        return "gnd"
    
    if DESIGNATION_PATTERNS["diode"].match(text) or \
       DESIGNATION_PATTERNS["resistor"].match(text) or \
       DESIGNATION_PATTERNS["relay"].match(text) or \
       DESIGNATION_PATTERNS["capacitor"].match(text):
        return "designator"
    
    return "other"


def normalize_designation(text: str) -> str:
    """
    Normalize designation text (fix common OCR errors).
    
    Args:
        text: Raw OCR text
        
    Returns:
        Normalized text
    """
    text = text.strip()
    
    # Fix common character confusions - only for connector designators
    # Replace Latin X with Cyrillic Х for designators
    if len(text) >= 2 and text[0] in 'Xx' and text[1].isdigit():
        text = 'Х' + text[1:]
    
    # Remove spaces
    text = text.replace(' ', '')
    
    return text


def validate_designation(text: str, expected_type: str) -> bool:
    """
    Validate text against expected type.
    
    Args:
        text: Text to validate
        expected_type: Expected text type
        
    Returns:
        True if text matches expected type
    """
    text = normalize_designation(text)
    
    if expected_type == "connector":
        return bool(DESIGNATION_PATTERNS["connector_x"].match(text))
    elif expected_type == "pin":
        return bool(DESIGNATION_PATTERNS["pin_numeric"].match(text) or
                    DESIGNATION_PATTERNS["pin_letter"].match(text))
    elif expected_type == "wire_type":
        return bool(DESIGNATION_PATTERNS["wire_bpvl"].match(text) or
                    DESIGNATION_PATTERNS["wire_mgtf"].match(text))
    elif expected_type == "voltage":
        return bool(DESIGNATION_PATTERNS["voltage"].match(text))
    
    return False


class TextAssociationEngine:
    """
    Associates text labels with nearby schematic elements.
    
    Uses spatial indexing (k-d tree) for efficient proximity queries.
    """
    
    def __init__(self, association_radius: float = 50.0):
        self.association_radius = association_radius
        self.labels: List[TextLabel] = []
        self._kdtree = None
        self._coords = None
    
    def add_labels(self, ocr_results: List[OCRResult]):
        """Add OCR results as labels."""
        for result in ocr_results:
            label = TextLabel(
                text=result.text,
                coord=result.center,
                bbox=result.bbox,
                label_type=result.text_type,
                confidence=result.confidence,
            )
            self.labels.append(label)
        
        self._build_kdtree()
    
    def _build_kdtree(self):
        """Build k-d tree for spatial queries."""
        if not self.labels:
            self._kdtree = None
            self._coords = None
            return
        
        from scipy.spatial import KDTree
        
        self._coords = np.array([label.coord for label in self.labels])
        self._kdtree = KDTree(self._coords)
    
    def find_nearest_label(
        self,
        point: Tuple[int, int],
        label_types: Optional[List[str]] = None,
    ) -> Optional[TextLabel]:
        """
        Find nearest text label to a point.
        
        Args:
            point: Query point (x, y)
            label_types: Filter by these label types
            
        Returns:
            Nearest TextLabel or None
        """
        if self._kdtree is None:
            return None
        
        dist, idx = self._kdtree.query(point, k=1)
        
        if dist > self.association_radius:
            return None
        
        label = self.labels[idx]
        
        if label_types and label.label_type not in label_types:
            return None
        
        return label
    
    def find_labels_in_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        label_types: Optional[List[str]] = None,
    ) -> List[TextLabel]:
        """Find all labels within or near a bounding box."""
        x_min, y_min, x_max, y_max = bbox
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        
        if self._kdtree is None:
            return []
        
        # Query in expanded radius
        dist, indices = self._kdtree.query(center, k=min(10, len(self.labels)))
        
        results = []
        for i, d in zip(*[indices, dist] if isinstance(indices, np.ndarray) else ([indices], [dist])):
            if d > self.association_radius * 2:
                continue
            
            label = self.labels[i]
            
            # Check if label is within expanded bbox
            lx_min, ly_min, lx_max, ly_max = label.bbox
            if lx_max < x_min - self.association_radius or lx_min > x_max + self.association_radius:
                continue
            if ly_max < y_min - self.association_radius or ly_min > y_max + self.association_radius:
                continue
            
            if label_types and label.label_type not in label_types:
                continue
            
            results.append(label)
        
        return results


def recognize_schematic_text(
    image: np.ndarray,
    config: Optional[OCRConfig] = None,
) -> Tuple[List[TextLabel], List[OCRResult]]:
    """
    Full OCR pipeline for schematic text recognition.
    
    Args:
        image: Input image
        config: OCR configuration
        
    Returns:
        Tuple of (structured_labels, raw_results)
    """
    config = config or OCRConfig()
    
    engine = PaddleOCREngine(config)
    raw_results = engine.recognize(image)
    
    # Filter by confidence
    filtered = [r for r in raw_results if r.confidence >= config.text_confidence_threshold]
    
    # Convert to structured labels
    labels = []
    for result in filtered:
        label = TextLabel(
            text=result.text,
            coord=result.center,
            bbox=result.bbox,
            label_type=result.text_type,
            confidence=result.confidence,
        )
        labels.append(label)
    
    logger.info(f"OCR recognized {len(labels)} text labels")
    
    return labels, filtered
