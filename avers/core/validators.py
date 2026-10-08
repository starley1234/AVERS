"""
AVERS Pipeline - Production Ready Version

Key improvements:
1. Stateless pipeline API
2. SAHI integration
3. Proper validation
4. Error handling with fallbacks
5. Type safety
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from pathlib import Path
import time
import threading
import logging

import numpy as np
import cv2

from avers.core.logger import get_logger
from avers.core.types import (
    AVERSManifest,
    SchemaMetadata,
    Component,
    Net,
    WireConnection,
    HumanReviewIssue,
    IssueType,
    ComponentType,
    Pin,
)
from avers.config import AVERSConfig, DEFAULT_CONFIG
from avers.stages.stage5_graph_synthesis import (
    GraphBuilder,
    WireSegment,
    PinReference,
)

logger = get_logger("avers.pipeline")


# =============================================================================
# SAHI Integration
# =============================================================================

def get_sahi_integration() -> Optional[Any]:
    """
    Get SAHI integration if available.
    
    Returns:
        SAHI SlicingInference class or None
    """
    try:
        from sahi import SlicingInference, Model, AutoDetectionModel
        return {"SlicingInference": SlicingInference, "Model": Model, "AutoDetectionModel": AutoDetectionModel}
    except ImportError:
        return None


_DETECTOR_CACHE: Dict[Tuple, "SlicedDetector"] = {}
_DETECTOR_LOCK = threading.Lock()


class SlicedDetector:
    """
    УГО detection: trained weights (ultralytics, tiled) with GOST refinement,
    or the CPU template detector, or nothing - never silently.

    * ``model_path`` set and loadable -> backend ``"model"``: YOLO/RT-DETR boxes
      on overlapping tiles, then ``GOSTTemplateDetector.refine`` derives the
      rotation and pin coordinates (hybrid). With ``merge_templates`` the
      template detector's own finds that the model missed are added.
    * no weights (or they cannot be loaded) -> ``"gost_templates"`` if
      ``template_fallback`` else ``"unavailable"``. Why the weights were not
      used is kept in ``load_error`` and reported by the pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "yolov11",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        slice_size: int = 1024,
        overlap_ratio: float = 0.2,
        template_fallback: bool = False,
        template_min_score: float = 0.62,
        template_px_per_mm: Optional[float] = None,
        merge_templates: bool = True,
        iou_threshold: float = 0.5,
        model: Any = None,
    ):
        self.template_fallback = template_fallback
        self.template_min_score = template_min_score
        self.template_px_per_mm = template_px_per_mm
        self.merge_templates = merge_templates
        self._template = None
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        
        self._model = None
        self._injected_model = model  # tests / custom predictors
        self._loaded = False
        self.backend = "uninitialized"
        self.load_error: Optional[str] = None
        self.last_stats: Dict[str, Any] = {}
    
    def load(self) -> bool:
        """Load detection model."""
        if self._loaded:
            return True
        
        if not self.model_path and self._injected_model is None:
            return self._load_fallback_model()

        try:
            return self._load_ultralytics_model()
        except ImportError:
            self.load_error = ("веса указаны, но пакет ultralytics не установлен "
                               "(pip install ultralytics)")
        except Exception as e:
            self.load_error = f"не удалось загрузить веса {self.model_path}: {e}"
        logger.warning(f"Trained detector not used: {self.load_error}")
        return self._load_fallback_model()
    
    def _load_ultralytics_model(self) -> bool:
        from avers.stages.stage2_detection.model_detector import UltralyticsTiledDetector
        self._model = UltralyticsTiledDetector(
            model_path=self.model_path or "<injected>",
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            tile_size=self.slice_size,
            overlap_ratio=self.overlap_ratio,
            model=self._injected_model,
        ).load()
        from avers.stages.stage2_detection.template_detector import GOSTTemplateDetector
        self._template = GOSTTemplateDetector(
            min_score=self.template_min_score, px_per_mm=self.template_px_per_mm,
        )
        self._loaded = True
        self.backend = "model"
        return True
    
    def _load_fallback_model(self) -> bool:
        """No trained model: use the GOST template detector (if enabled) or skip."""
        self._loaded = True
        if self.template_fallback:
            from avers.stages.stage2_detection.template_detector import GOSTTemplateDetector
            self._template = GOSTTemplateDetector(
                min_score=self.template_min_score,
                px_per_mm=self.template_px_per_mm,
            )
            self.backend = "gost_templates"
            logger.info("No trained detector; using GOST template detector (CPU)")
            return True
        logger.warning("No trained detector available; component classification skipped")
        self.backend = "unavailable"
        return True
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run sliced detection on image.
        
        Args:
            image: Input image (H, W, 3) RGB
            
        Returns:
            List of detection dicts with bbox, confidence, category
        """
        if not self._loaded:
            self.load()
        
        if self.backend == "model" and self._model is not None:
            return self._model_detect(image)
        if self.backend == "gost_templates" and self._template is not None:
            return self._template.detect(image)
        return self._fallback_detect(image)
    
    def _model_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Hybrid: model boxes -> GOST refinement (rotation, pins) [+ template finds]."""
        from avers.stages.stage2_detection.model_detector import merge_detections
        raw = self._model.detect(image)
        refined = self._template.refine(image, raw)
        added: List[Dict[str, Any]] = []
        if self.merge_templates:
            merged = merge_detections(refined, self._template.detect(image))
            added = merged[len(refined):]
            refined = merged
        sources: Dict[str, int] = {}
        for d in refined:
            key = d.get("pins_source") or d.get("source", "?")
            sources[key] = sources.get(key, 0) + 1
        self.last_stats = {"model": len(raw), "added_by_templates": len(added), "by_source": sources}
        logger.info(f"Hybrid detection: {self.last_stats}")
        return refined
    
    def _fallback_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Do not invent component classes when a trained model is unavailable."""
        return []


# =============================================================================
# OCR Integration
# =============================================================================

_EASYOCR_READERS: Dict[Tuple[str, ...], Any] = {}
_EASYOCR_LOCK = threading.Lock()


def _get_easyocr_reader(langs: Tuple[str, ...]):
    """Process-wide EasyOCR reader (first call downloads/loads the models)."""
    with _EASYOCR_LOCK:
        reader = _EASYOCR_READERS.get(langs)
        if reader is None:
            import easyocr
            try:
                import torch
                gpu = bool(torch.cuda.is_available())
            except Exception:
                gpu = False
            logger.info(f"Loading EasyOCR {list(langs)} ({'GPU' if gpu else 'CPU'})...")
            reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
            _EASYOCR_READERS[langs] = reader
            logger.info("EasyOCR loaded")
        return reader


class SchematicOCR:
    """
    OCR engine with automatic backend selection.
    
    Priority: PaddleOCR > EasyOCR > no text (if unavailable)
    """
    
    def __init__(
        self,
        lang: str = "ru",
        confidence_threshold: float = 0.65,
    ):
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        
        self._ocr = None
        self._backend = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load OCR backend."""
        if self._loaded:
            return True
        
        # Try PaddleOCR (API 2.x; 3.x changed the constructor -> fall through)
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=True,
                show_log=False,
            )
            self._backend = "paddleocr"
            self._loaded = True
            logger.info("PaddleOCR loaded")
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PaddleOCR installed but failed to load ({e}); trying EasyOCR")
        
        # Try EasyOCR. The reader (~100 MB of weights) is cached per process:
        # the pipeline creates a new SchematicOCR for every request.
        try:
            self._ocr = _get_easyocr_reader((self.lang, "en"))
            self._backend = "easyocr"
            self._loaded = True
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"EasyOCR failed to load ({e}); text recognition skipped")
        
        # No OCR backend: do not fabricate designators from contour sizes.
        self._backend = "unavailable"
        self._loaded = True
        logger.warning("No OCR backend available; text recognition skipped")
        return True
    
    def recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize text in image.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            List of text results with bbox, text, confidence
        """
        if not self._loaded:
            self.load()
        
        if self._backend == "paddleocr":
            return self._paddleocr_recognize(image)
        elif self._backend == "easyocr":
            return self._easyocr_recognize(image)
        else:
            return self._unavailable_recognize(image)
    
    def _paddleocr_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """PaddleOCR recognition."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self._ocr.ocr(bgr, cls=True)
        
        texts = []
        if results and results[0]:
            for line in results[0]:
                points = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                
                if not self._keep(text, confidence):
                    continue
                
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                texts.append({
                    "text": text.strip(),
                    "bbox": (int(min(x_coords)), int(min(y_coords)), 
                            int(max(x_coords)), int(max(y_coords))),
                    "confidence": confidence,
                })
        
        return texts
    
    def _keep(self, text: str, confidence: float) -> bool:
        """Main threshold for free text; short designators / pin numbers get
        lower scores from OCR models, so they pass from 0.25 (they are later
        validated by the ГОСТ 2.710 pattern and by geometry)."""
        if confidence >= self.confidence_threshold:
            return True
        from avers.core.designators import normalize_designator
        short = text.strip()
        return confidence >= 0.25 and (normalize_designator(short) is not None
                                       or (short.isdigit() and len(short) <= 3))

    def _easyocr_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """EasyOCR recognition."""
        with _EASYOCR_LOCK:  # the cached reader is shared between requests
            results = self._ocr.readtext(image)
        
        texts = []
        for bbox, text, confidence in results:
            if not self._keep(text, float(confidence)):
                continue
            
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            texts.append({
                "text": text.strip(),
                "bbox": (int(min(x_coords)), int(min(y_coords)),
                        int(max(x_coords)), int(max(y_coords))),
                "confidence": confidence,
            })
        
        return texts
    
    DESIGNATOR_CHARS = "ABCDEFGHKLMPQRSTUVWXYZ0123456789."

    def recognize_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        blank: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Second-pass OCR of a small region (e.g. around one УГО).

        Text detectors miss short labels like "R1" sitting next to lines. Here
        the region is cleaned (the symbol itself and long wires are painted
        white), upscaled and read with a designator-only alphabet.
        Returned bboxes are in full-image coordinates.
        """
        if self._backend != "easyocr":
            return []
        h, w = image.shape[:2]
        x0, y0, x1, y1 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        crop = image[y0:y1, x0:x1].copy()
        if blank is not None:
            bx0, by0 = max(0, blank[0] - x0), max(0, blank[1] - y0)
            bx1, by1 = min(x1 - x0, blank[2] - x0), min(y1 - y0, blank[3] - y0)
            if bx1 > bx0 and by1 > by0:
                crop[by0:by1, bx0:bx1] = 255
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        length = 25
        lines = cv2.bitwise_or(
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))),
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))),
        )
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        clean = gray.copy()
        clean[lines > 0] = 255
        big = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        with _EASYOCR_LOCK:
            results = self._ocr.readtext(big, allowlist=self.DESIGNATOR_CHARS)
        texts = []
        for pts, text, confidence in results:
            if not self._keep(text, float(confidence)):
                continue
            xs = [p[0] / scale + x0 for p in pts]
            ys = [p[1] / scale + y0 for p in pts]
            texts.append({
                "text": text.strip(),
                "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                "confidence": float(confidence),
                "pass": "region",
            })
        return texts

    def _unavailable_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Return no text instead of invented labels when OCR is unavailable."""
        return []


# =============================================================================
# Wire Vectorization
# =============================================================================

class WireVectorizer:
    """
    Production wire vectorization.
    
    Uses OpenCV skeletonization + Hough Transform.
    """
    
    def __init__(
        self,
        rdp_epsilon: float = 2.0,
        min_line_length: int = 20,
        exclusion_padding: int = 5,
        hough_threshold: int = 20,
    ):
        self.hough_threshold = hough_threshold
        self.rdp_epsilon = rdp_epsilon
        self.min_line_length = min_line_length
        self.exclusion_padding = exclusion_padding
    
    @staticmethod
    def _extend_along_ink(
        origin: Tuple[int, int], end: Tuple[int, int], support: np.ndarray, max_ext: int = 40,
    ) -> Tuple[int, int]:
        """Move ``end`` away from ``origin`` while ``support`` has ink."""
        dx, dy = end[0] - origin[0], end[1] - origin[1]
        norm = float(np.hypot(dx, dy))
        if norm == 0:
            return end
        ux, uy = dx / norm, dy / norm
        height, width = support.shape[:2]
        best = end
        for k in range(1, max_ext + 1):
            x = int(round(end[0] + ux * k))
            y = int(round(end[1] + uy * k))
            if not (0 <= x < width and 0 <= y < height) or support[y, x] == 0:
                break
            best = (x, y)
        return best

    @staticmethod
    def _stroke_width(binary: np.ndarray) -> float:
        """Mean stroke width = ink area / skeleton length (inf if no ink)."""
        ink = binary > 0
        if not ink.any():
            return float("inf")
        from skimage import morphology
        length = int(morphology.skeletonize(ink).sum())
        return float(ink.sum()) / max(1, length)

    @staticmethod
    def _erase_text(thresh: np.ndarray, text_bboxes: List[Tuple[int, int, int, int]]) -> None:
        """In-place: inside text boxes keep only long horizontal/vertical strokes."""
        height, width = thresh.shape[:2]
        for x0, y0, x1, y1 in text_bboxes:
            x0, y0 = max(0, int(x0) - 1), max(0, int(y0) - 1)
            x1, y1 = min(width, int(x1) + 2), min(height, int(y1) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            length = int(1.3 * max(y1 - y0, 8)) + 1
            # crop with context so a wire longer than the box is still "long"
            cx0, cy0 = max(0, x0 - length), max(0, y0 - length)
            cx1, cy1 = min(width, x1 + length), min(height, y1 + length)
            crop = thresh[cy0:cy1, cx0:cx1]
            horiz = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
            vert = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
            keep = cv2.bitwise_or(horiz, vert)
            box = (slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
            thresh[y0:y1, x0:x1] = cv2.bitwise_and(crop[box], keep[box])

    def vectorize(
        self,
        image: np.ndarray,
        exclusion_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        text_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[List[WireSegment], set]:
        """
        Vectorize wires in image.
        
        Args:
            image: Input image (H, W, 3) RGB
            exclusion_bboxes: Bboxes to exclude from vectorization
            
        Returns:
            Tuple of (segments, junctions)
        """
        # Preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2,
        )
        # The local threshold alone turns paper grain / sensor noise into ink.
        # Keep only pixels that are also dark globally (Otsu, with a margin),
        # unless the page is not bimodal at all (then Otsu is meaningless).
        otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dark = gray < min(255, otsu_t + 20)
        if 0.0 < dark.mean() < 0.35:
            thresh[~dark] = 0
            # remove speckles much smaller than any wire piece
            n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 12) + 1
            if small.size:
                thresh[np.isin(labels, small)] = 0
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Opening with a 3x3 kernel removes noise but also erases every line
        # thinner than 3 px (typical for CAD exports / 150 dpi scans). Apply it
        # only when the strokes are clearly thicker than the kernel.
        if self._stroke_width(thresh) >= 4.0:
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Text labels: erase letters but keep wires running through/along the
        # (often loose) OCR box. Inside each box only ink belonging to straight
        # strokes longer than the text height survives.
        if text_bboxes:
            self._erase_text(thresh, text_bboxes)
        
        # Apply exclusion mask
        if exclusion_bboxes:
            mask = np.ones(gray.shape[:2], dtype=np.uint8) * 255
            for bbox in exclusion_bboxes:
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, x_min - self.exclusion_padding)
                y_min = max(0, y_min - self.exclusion_padding)
                x_max = min(gray.shape[1], x_max + self.exclusion_padding)
                y_max = min(gray.shape[0], y_max + self.exclusion_padding)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 0, -1)
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # Skeletonize
        from skimage import morphology
        skeleton = morphology.skeletonize((thresh > 0).astype(np.uint8))
        skeleton = (skeleton * 255).astype(np.uint8)
        
        # Extract lines
        # threshold = minimum number of skeleton pixels voting for a line. 50
        # dropped every wire shorter than ~50 px, i.e. most leads between two
        # closely placed УГО (after their bboxes are erased) on CAD drawings.
        lines = cv2.HoughLinesP(
            skeleton, rho=1, theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=10,
        )
        
        # HoughLinesP can bridge up to 10 px of blank space. On scans this
        # stitches the dashes of a page frame into apparent wires. Check the
        # original (pre-Hough) skeleton for repeated gaps, but only near the
        # page margins: a broken wire in the circuit interior is ambiguous.
        support = cv2.dilate(skeleton, kernel) if lines is not None else None
        height, width = gray.shape[:2]
        segments = []
        junctions = set()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = (int(v) for v in line.reshape(4))
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                horizontal_frame = (dy <= max(2, dx * 0.03) and
                                    (max(y1, y2) < height * 0.12 or
                                     min(y1, y2) > height * 0.88))
                vertical_frame = (dx <= max(2, dy * 0.03) and
                                  (max(x1, x2) < width * 0.12 or
                                   min(x1, x2) > width * 0.88))
                if (horizontal_frame or vertical_frame) and max(dx, dy) >= 20:
                    count = max(dx, dy) + 1
                    xs = np.rint(np.linspace(x1, x2, count)).astype(int)
                    ys = np.rint(np.linspace(y1, y2, count)).astype(int)
                    ink = support[ys, xs] != 0
                    changes = np.diff(np.r_[0, (~ink).astype(np.int8), 0])
                    gaps = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
                    if ink.mean() < 0.85 and np.count_nonzero(gaps >= 3) >= 2:
                        continue

                # Near-axis Hough lines (1 px drift) -> exactly orthogonal, so
                # T/collinear joins in Stage 5 compare equal coordinates.
                if dy <= 1 and dx >= 10:
                    y1 = y2 = int(round((y1 + y2) / 2))
                elif dx <= 1 and dy >= 10:
                    x1 = x2 = int(round((x1 + x2) / 2))
                # HoughLinesP usually stops a few px short of corners, T-joins
                # and the erased component bboxes. Extend both ends while the
                # original skeleton continues in the same direction.
                (x1, y1), (x2, y2) = (
                    self._extend_along_ink((x2, y2), (x1, y1), support),
                    self._extend_along_ink((x1, y1), (x2, y2), support),
                )

                # RDP simplification (single segment = no simplification needed)
                segments.append(WireSegment(
                    segment_id=len(segments),
                    start=(x1, y1),
                    end=(x2, y2),
                    points=[(x1, y1), (x2, y2)],
                    confidence=0.9,
                ))

        segments = self._drop_contained(segments)
        segments = self._drop_isolated_short(segments, exclusion_bboxes or [])
        for seg in segments:
            # Track endpoints as potential junctions
            junctions.add(seg.start)
            junctions.add(seg.end)

        return segments, junctions

    def _drop_isolated_short(
        self,
        segments: List[WireSegment],
        exclusion_bboxes: List[Tuple[int, int, int, int]],
        short: float = 50.0,
        tol: float = 4.0,
    ) -> List[WireSegment]:
        """Drop short segments touching neither another wire nor a component.

        The low Hough threshold is needed for short leads between closely
        placed УГО, but it also turns single dashes of a page frame and
        strokes of letters into "wires". Those are isolated; real short wires
        end at a component (its erased bbox) or at another wire.
        """
        if not segments:
            return segments
        reach = self.exclusion_padding + tol
        boxes = [(x0 - reach, y0 - reach, x1 + reach, y1 + reach)
                 for x0, y0, x1, y1 in exclusion_bboxes]

        def near_box(p):
            return any(b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3] for b in boxes)

        def dist(p, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            ll = dx * dx + dy * dy
            if ll == 0:
                return float(np.hypot(p[0] - a[0], p[1] - a[1]))
            t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ll))
            return float(np.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))

        cell = 32
        grid: Dict[Tuple[int, int], List[int]] = {}
        for i, seg in enumerate(segments):
            x0 = int((min(seg.start[0], seg.end[0]) - tol) // cell)
            x1 = int((max(seg.start[0], seg.end[0]) + tol) // cell)
            y0 = int((min(seg.start[1], seg.end[1]) - tol) // cell)
            y1 = int((max(seg.start[1], seg.end[1]) + tol) // cell)
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    grid.setdefault((cx, cy), []).append(i)

        kept = []
        for i, seg in enumerate(segments):
            if seg.length >= short:
                kept.append(seg)
                continue
            ok = False
            for p in (seg.start, seg.end):
                if near_box(p):
                    ok = True
                    break
                cand = grid.get((int(p[0] // cell), int(p[1] // cell)), ())
                if any(j != i and dist(p, segments[j].start, segments[j].end) <= tol for j in cand):
                    ok = True
                    break
            if ok:
                kept.append(seg)
        for i, seg in enumerate(kept):
            seg.segment_id = i
        return kept

    @staticmethod
    def _drop_contained(segments: List[WireSegment], tol: float = 2.0) -> List[WireSegment]:
        """Remove Hough duplicates lying inside a longer collinear segment.

        A duplicate that stops mid-wire (typically at a crossing without a
        junction dot) would otherwise look like a T-joint and merge two nets.
        """
        def dist(p, a, b):
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy
            if ll == 0:
                return float(np.hypot(p[0] - ax, p[1] - ay))
            t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / ll))
            return float(np.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy))

        cell = 32
        grid: Dict[Tuple[int, int], List[WireSegment]] = {}
        kept: List[WireSegment] = []
        for seg in sorted(segments, key=lambda s: -s.length):
            gx, gy = int(seg.start[0] // cell), int(seg.start[1] // cell)
            contained = any(
                dist(seg.start, k.start, k.end) <= tol and dist(seg.end, k.start, k.end) <= tol
                for cx in (gx - 1, gx, gx + 1) for cy in (gy - 1, gy, gy + 1)
                for k in grid.get((cx, cy), ())
            )
            if contained:
                continue
            kept.append(seg)
            x0 = int((min(seg.start[0], seg.end[0]) - tol) // cell)
            x1 = int((max(seg.start[0], seg.end[0]) + tol) // cell)
            y0 = int((min(seg.start[1], seg.end[1]) - tol) // cell)
            y1 = int((max(seg.start[1], seg.end[1]) + tol) // cell)
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    grid.setdefault((cx, cy), []).append(seg)
        kept.sort(key=lambda s: s.segment_id)
        for i, seg in enumerate(kept):
            seg.segment_id = i
        return kept


# =============================================================================
# Production Pipeline
# =============================================================================

@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    manifest: AVERSManifest
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    success: bool = True


class ProductionPipeline:
    """
    Production-ready AVERS pipeline.
    
    Features:
    - Stateless execution
    - Comprehensive error handling
    - Validation at each stage
    - Clear error messages
    - Stage timings
    """
    
    def __init__(self, config: Optional[AVERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        errors = []
        
        if self.config.slicing.tile_size < 256:
            errors.append("tile_size must be >= 256")
        if self.config.slicing.overlap_ratio < 0 or self.config.slicing.overlap_ratio > 0.5:
            errors.append("overlap_ratio must be 0.0-0.5")
        if self.config.graph_synthesis.snap_radius < 0:
            errors.append("snap_radius must be >= 0")
        
        if errors:
            raise ValueError(f"Invalid config: {', '.join(errors)}")
    
    def run(
        self,
        image: np.ndarray,
        source_file: str = "unknown",
        dpi: int = 300,
    ) -> PipelineResult:
        """
        Run production pipeline on image.
        
        Args:
            image: Input image (H, W, 3) RGB
            source_file: Source file name for metadata
            dpi: Image resolution DPI
            
        Returns:
            PipelineResult with manifest and metadata
        """
        start_time = time.time()
        result = PipelineResult(
            manifest=AVERSManifest(
                schema_metadata=SchemaMetadata(
                    source_file=source_file,
                    resolution_dpi=dpi,
                    width=image.shape[1],
                    height=image.shape[0],
                )
            )
        )
        
        # Stage 1: Detection
        stage_start = time.time()
        try:
            detections, component_bboxes, backend = self._run_detection(image)
            result.manifest.components = self._group_components(detections)
            detector = getattr(self, "_last_detector", None)
            load_error = getattr(detector, "load_error", None)
            if load_error:
                result.warnings.append(f"Обученный детектор не используется: {load_error}")
            if backend == "gost_templates":
                result.warnings.append(
                    "Детекция УГО: обученных весов нет, применён шаблонный детектор по библиотеке "
                    "ГОСТ УГО (CPU). Надёжен для чистых схем по ЕСКД; классы и обозначения "
                    "проверьте в валидаторе"
                )
            elif backend == "model":
                stats = getattr(detector, "last_stats", {}) or {}
                geometry = stats.get("by_source", {}).get("geometry", 0)
                if geometry:
                    result.warnings.append(
                        f"Детекция УГО (обученная модель): у {geometry} УГО выводы определены "
                        f"только по рамке (шаблон ГОСТ не подтвердил) - проверьте их подключения"
                    )
            else:
                result.warnings.append("Детекция УГО недоступна: нет обученных весов; классы не определены")
        except Exception as e:
            result.errors.append(f"Detection failed: {e}")
            result.warnings.append("Using empty components")
            result.manifest.components = []
            component_bboxes = []
        
        result.stage_timings["detection"] = time.time() - stage_start
        
        # Stage 2: OCR
        stage_start = time.time()
        try:
            texts, text_bboxes, ocr_available = self._run_ocr(image)
            if not ocr_available:
                result.warnings.append("OCR недоступен: маркировки не определены")
            # Associate texts with components (may add second-pass labels)
            self._associate_texts(result.manifest.components, texts, image=image)
            text_bboxes = [t["bbox"] for t in texts]
        except Exception as e:
            result.errors.append(f"OCR failed: {e}")
            result.warnings.append("Text recognition skipped")
            texts = []
            text_bboxes = []
        
        result.stage_timings["ocr"] = time.time() - stage_start
        
        # Stage 3: Vectorization
        stage_start = time.time()
        try:
            segments, junctions = self._run_vectorization(
                image,
                component_bboxes,
                text_bboxes=text_bboxes,
            )
        except Exception as e:
            result.errors.append(f"Vectorization failed: {e}")
            result.warnings.append("Using empty wires")
            segments = []
            junctions = set()
        
        result.stage_timings["vectorization"] = time.time() - stage_start
        
        # Stage 4: Graph synthesis
        stage_start = time.time()
        try:
            nets = self._run_graph_synthesis(
                segments,
                junctions,
                result.manifest.components,
                image=image,
            )
            result.manifest.nets = nets
        except Exception as e:
            result.errors.append(f"Graph synthesis failed: {e}")
            result.warnings.append("Using empty nets")
            result.manifest.nets = []
        
        result.stage_timings["graph_synthesis"] = time.time() - stage_start
        
        # Stage 5: VLM arbitration (if enabled and issues exist)
        stage_start = time.time()
        if self.config.vlm_arbitrator.enabled:
            try:
                self._run_vlm_arbitration(image, result.manifest)
            except Exception as e:
                result.warnings.append(f"VLM arbitration skipped: {e}")
        
        result.stage_timings["vlm_arbitration"] = time.time() - stage_start
        
        # Finalize
        result.success = len(result.errors) == 0
        result.manifest.processing_warnings = result.warnings.copy()
        result.manifest.schema_metadata.processing_time_seconds = time.time() - start_time
        
        logger.info(
            f"Pipeline complete: {len(result.manifest.components)} components, "
            f"{len(result.manifest.nets)} nets, {len(result.manifest.human_review_required)} issues"
        )
        
        return result
    
    def _resolve_model_path(self) -> Tuple[Optional[str], str]:
        """Weights to use: explicit ``detection.model_path``, else the current
        version of the active-learning registry (so promote/rollback actually
        switch the production detector), else none."""
        det_cfg = self.config.detection
        if det_cfg.model_path:
            return det_cfg.model_path, "config"
        al = getattr(self.config, "active_learning", None)
        if not getattr(det_cfg, "use_registry", True) or al is None:
            return None, "none"
        index = Path(al.registry_dir) / "registry.json"
        if not index.is_file():  # do not create registry dirs as a side effect
            return None, "none"
        try:
            from avers.active_learning.registry import ModelRegistry
            current = ModelRegistry(Path(al.registry_dir)).get_current()
        except Exception as e:
            logger.warning(f"Model registry unreadable: {e}")
            return None, "none"
        if current and Path(current.weights_path).is_file():
            return current.weights_path, f"registry {current.version_id}"
        return None, "none"

    def _run_detection(
        self, image: np.ndarray
    ) -> Tuple[List[Dict], List[Tuple[int, int, int, int]], bool]:
        """Run component detection."""
        det_cfg = self.config.detection
        model_path, self._model_source = self._resolve_model_path()
        kwargs = dict(
            model_path=model_path,
            model_type=det_cfg.model_type,
            confidence_threshold=det_cfg.confidence_threshold,
            device=det_cfg.device,
            slice_size=self.config.slicing.tile_size,
            overlap_ratio=self.config.slicing.overlap_ratio,
            template_fallback=getattr(det_cfg, "template_fallback", True),
            template_min_score=getattr(det_cfg, "template_min_score", 0.62),
            template_px_per_mm=getattr(det_cfg, "template_px_per_mm", None),
            merge_templates=getattr(det_cfg, "merge_templates", True),
            iou_threshold=det_cfg.iou_threshold,
        )
        # Loading weights takes seconds; keep one detector per configuration.
        key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
        if model_path:
            try:
                key += (("mtime", str(Path(model_path).stat().st_mtime)),)
            except OSError:
                pass
        with _DETECTOR_LOCK:
            detector = _DETECTOR_CACHE.get(key)
            if detector is None:
                detector = SlicedDetector(**kwargs)
                detector.load()
                if model_path:
                    _DETECTOR_CACHE.clear()
                    _DETECTOR_CACHE[key] = detector
        self._last_detector = detector
        
        detections = detector.detect(image)
        # Точки соединения - часть провода: их нельзя вырезать из векторизации.
        bboxes = [d["bbox"] for d in detections
                  if d.get("exclude_from_wires", d.get("category") != "junction_dot")]
        
        return detections, bboxes, detector.backend
    
    def _group_components(
        self, detections: List[Dict]
    ) -> List[Component]:
        """Group detections into semantic components."""
        from scipy.spatial import KDTree
        
        if any("pins" in d for d in detections):
            return self._components_with_pins(detections)
        
        components = []
        used = set()
        next_id = 1
        
        # Separate connectors and pins
        connectors = [d for d in detections if d["category"] == "connector_body"]
        pins = [d for d in detections if d["category"] == "pin"]
        
        # Group pins with connectors
        if pins and connectors:
            pin_coords = np.array([self._bbox_center(d["bbox"]) for d in pins])
            pin_tree = KDTree(pin_coords)
        
        for conn in connectors:
            if id(conn) in used:
                continue
            
            conn_center = self._bbox_center(conn["bbox"])
            conn_pins = []
            
            # Find nearby pins
            if pins:
                dists, indices = pin_tree.query(conn_center, k=min(10, len(pins)))
                for dist, idx in zip(dists, indices):
                    if dist < self.config.graph_synthesis.snap_radius * 3:
                        pin = pins[idx]
                        conn_pins.append(Pin(
                            pin_number=str(len(conn_pins) + 1),
                            coord=self._bbox_center(pin["bbox"]),
                            confidence=pin["confidence"],
                        ))
            
            comp_type = self._category_to_component_type(conn["category"])
            
            components.append(Component(
                id=f"comp_{next_id:03d}",
                designator=f"X{next_id}",
                type=comp_type,
                bbox=conn["bbox"],
                pins=conn_pins,
                confidence=conn["confidence"],
            ))
            
            used.add(id(conn))
            next_id += 1
        
        # Add other components
        for det in detections:
            if id(det) in used:
                continue
            if det["category"] in ("connector_body", "pin"):
                continue
            
            components.append(Component(
                id=f"comp_{next_id:03d}",
                designator=f"{det['category'][:2].upper()}{next_id}",
                type=self._category_to_component_type(det["category"]),
                bbox=det["bbox"],
                confidence=det["confidence"],
            ))
            next_id += 1
        
        return components
    
    # Обозначения для УГО без буквенного кода по ГОСТ 2.710-81
    _UNNAMED_PREFIX = {
        "ground": "GND", "chassis": "CHS", "junction_dot": "J",
        "offpage_connector": "OFF", "shield": "SH",
    }

    def _components_with_pins(self, detections: List[Dict]) -> List[Component]:
        """Компоненты из детекций, уже содержащих выводы (шаблонный детектор).

        Позиционные обозначения назначаются по ГОСТ 2.710-81 (R, C, VD, K, FU,
        SA, EL, HL, GB, M, X ...) с нумерацией сверху вниз, слева направо, как
        принято на схемах. Без OCR это предположение - реальные обозначения
        со схемы распознаёт OCR и переносит в text_associations."""
        from avers.dataset.gost_symbols import designator_prefix

        def order_key(det):
            x0, y0, x1, y1 = det["bbox"]
            return (round(((y0 + y1) / 2) / 40), (x0 + x1) / 2)

        counters: Dict[str, int] = {}
        components = []
        for idx, det in enumerate(sorted(detections, key=order_key), start=1):
            category = det["category"]
            prefix = designator_prefix(category) or self._UNNAMED_PREFIX.get(
                category, category[:2].upper())
            counters[prefix] = counters.get(prefix, 0) + 1
            pins = []
            seen = set()
            det_pins = det.get("pins", [])
            if category == "junction_dot" and not det_pins:
                # Точка соединения (ГОСТ 2.721-74) - явное свидетельство
                # электрического узла: провода, подходящие к ней, соединяются.
                x0, y0, x1, y1 = det["bbox"]
                det_pins = [{"name": "J", "coord": ((x0 + x1) // 2, (y0 + y1) // 2)}]
            for pin in det_pins:
                key = (pin["name"], tuple(pin["coord"]))
                if key in seen:
                    continue
                seen.add(key)
                pins.append(Pin(pin_number=str(pin["name"]), coord=tuple(int(v) for v in pin["coord"]),
                                confidence=float(det.get("confidence", 1.0))))
            comp = Component(
                id=f"comp_{idx:03d}",
                designator=f"{prefix}{counters[prefix]}",
                type=self._category_to_component_type(category),
                bbox=tuple(int(v) for v in det["bbox"]),
                pins=pins,
                confidence=float(det.get("confidence", 1.0)),
            )
            comp.text_associations["gost_class"] = category
            if "rotation" in det:
                comp.text_associations["rotation"] = str(det["rotation"])
            components.append(comp)
        return components

    def _run_ocr(
        self, image: np.ndarray
    ) -> Tuple[List[Dict], List[Tuple[int, int, int, int]], bool]:
        """Run OCR."""
        ocr = SchematicOCR(
            lang=self.config.ocr.lang,
            confidence_threshold=self.config.ocr.text_confidence_threshold,
        )
        
        texts = ocr.recognize(image)
        bboxes = [t["bbox"] for t in texts]
        self._ocr_engine = ocr if ocr._backend not in (None, "unavailable") else None
        
        return texts, bboxes, ocr._backend != "unavailable"
    
    def _associate_texts(
        self,
        components: List[Component],
        texts: List[Dict],
        image: Optional[np.ndarray] = None,
    ) -> None:
        """Associate texts with nearby components."""
        if not components:
            return
        
        if any("gost_class" in c.text_associations for c in components):
            if texts or getattr(self, "_ocr_engine", None) is not None:
                self._apply_ocr_designators(components, texts, image)
            return
        if not texts:
            return
        
        from scipy.spatial import KDTree
        
        comp_centers = np.array([
            ((c.bbox[0] + c.bbox[2]) // 2, (c.bbox[1] + c.bbox[3]) // 2)
            for c in components
        ])
        tree = KDTree(comp_centers)
        
        for text in texts:
            text_center = self._bbox_center(text["bbox"])
            
            dist, idx = tree.query(text_center)
            if dist < self.config.graph_synthesis.text_association_radius:
                components[idx].text_associations["designator"] = text["text"]
    
    def _ocr_second_pass(self, components: List[Component], texts: List[Dict],
                         assigned: Dict[int, Dict], image: np.ndarray) -> List[Dict]:
        """Re-read the neighbourhood of every labelled-class УГО left without a
        designator (EasyOCR's detector often skips short labels near lines)."""
        from avers.core.designators import CLASS_PREFIXES
        engine = getattr(self, "_ocr_engine", None)
        if engine is None or image is None:
            return []
        h, w = image.shape[:2]
        extra: List[Dict] = []
        todo = [i for i, c in enumerate(components)
                if i not in assigned and c.text_associations.get("gost_class") in CLASS_PREFIXES]
        for i in todo[:40]:
            x0, y0, x1, y1 = components[i].bbox
            pad = int(max(45, 0.7 * max(x1 - x0, y1 - y0)))
            region = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
            try:
                extra.extend(engine.recognize_region(image, region, blank=components[i].bbox))
            except Exception as e:  # OCR must never break the pipeline
                logger.warning(f"Second-pass OCR failed: {e}")
                break
        return extra

    def _fix_decimal_points(self, texts: List[Dict], image: Optional[np.ndarray]) -> None:
        from avers.core.designators import restore_decimal_point
        if image is None:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        for t in texts:
            x0, y0, x1, y1 = (int(v) for v in t["bbox"])
            crop = gray[max(0, y0):max(0, y1) + 1, max(0, x0):max(0, x1) + 1]
            if crop.size == 0:
                continue
            _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            fixed = restore_decimal_point(t["text"], ink > 0)
            if fixed != t["text"]:
                t["ocr_raw"] = t["text"]
                t["text"] = fixed

    def _apply_ocr_designators(self, components: List[Component], texts: List[Dict],
                               image: Optional[np.ndarray] = None) -> None:
        """Replace auto designators with ГОСТ 2.710 labels read by OCR.

        A label goes to the nearest УГО whose class admits its letter code
        (K1.1 -> relay contact, not a lamp), one-to-one. Components left
        without a label keep an auto designator, renumbered so it never
        collides with a recognised one. Connector pin numbers are read from
        the digits printed inside the connector cells.
        """
        import re
        from avers.core.designators import assign_designators, connector_pin_labels

        candidates = [
            {"bbox": c.bbox, "gost_class": c.text_associations.get("gost_class", "")}
            for c in components
        ]
        radius = self.config.graph_synthesis.text_association_radius
        self._fix_decimal_points(texts, image)
        assigned = assign_designators(candidates, texts, max_distance=radius)
        extra = self._ocr_second_pass(components, texts, assigned, image)
        if extra:
            self._fix_decimal_points(extra, image)
            texts.extend(extra)
            assigned = assign_designators(candidates, texts, max_distance=radius)
        taken = {info["designator"] for info in assigned.values()}
        for idx, info in assigned.items():
            comp = components[idx]
            comp.text_associations["auto_designator"] = comp.designator
            comp.text_associations["designator_source"] = "ocr"
            comp.text_associations["ocr_text"] = info["text"]
            comp.text_associations["ocr_confidence"] = f"{info['confidence']:.2f}"
            comp.designator = info["designator"]

        # Renumber remaining auto designators away from recognised ones.
        counters: Dict[str, int] = {}
        for i, comp in enumerate(components):
            if i in assigned:
                continue
            comp.text_associations.setdefault("designator_source", "auto")
            m = re.match(r"^([A-Z]+)(\d+)$", comp.designator)
            if not m:
                continue
            prefix = m.group(1)
            n = counters.get(prefix, 0)
            while True:
                n += 1
                if f"{prefix}{n}" not in taken:
                    break
            counters[prefix] = n
            comp.designator = f"{prefix}{n}"
            taken.add(comp.designator)

        for comp in components:
            if comp.type != ComponentType.CONNECTOR or not comp.pins:
                continue
            coords: Dict[str, List[Tuple[int, int]]] = {}
            for pin in comp.pins:
                coords.setdefault(pin.pin_number, []).append(tuple(pin.coord))
            mapping = connector_pin_labels(comp.bbox, coords, texts)
            if mapping:
                for pin in comp.pins:
                    pin.pin_number = mapping.get(pin.pin_number, pin.pin_number)
                comp.text_associations["pin_numbers_source"] = "ocr"

    def _run_vectorization(
        self,
        image: np.ndarray,
        exclusion_bboxes: List[Tuple[int, int, int, int]],
        text_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[List[WireSegment], set]:
        """Run wire vectorization."""
        vectorizer = WireVectorizer(
            rdp_epsilon=self.config.vectorization.rdp_epsilon,
            min_line_length=self.config.vectorization.line_thickness_threshold,
            exclusion_padding=self.config.vectorization.snap_radius,
        )
        
        return vectorizer.vectorize(image, exclusion_bboxes, text_bboxes=text_bboxes)
    
    def _run_graph_synthesis(
        self,
        segments: List[WireSegment],
        junctions: set,
        components: List[Component],
        image: Optional[np.ndarray] = None,
    ) -> List[Net]:
        """Run graph synthesis, verifying longer wire gaps against source pixels."""
        builder = GraphBuilder(
            snap_enabled=self.config.graph_synthesis.snap_enabled,
            snap_radius=self.config.graph_synthesis.snap_radius,
            merge_collinear=self.config.graph_synthesis.merge_collinear_segments,
            gap_image=image,
            max_supported_gap=self.config.graph_synthesis.max_supported_gap,
            # Hough segment ends at a skeleton corner/T are typically 2-5 px
            # apart; 2 px left most L-corners of real drawings disconnected.
            junction_tolerance=self.config.graph_synthesis.junction_tolerance,
        )
        
        # Add segments
        for seg in segments:
            builder.add_wire_segment(seg.start, seg.end, seg.points, seg.confidence)
        
        # Add pins. Stage 3 erased the component bbox plus `pad` pixels, so a
        # wire attached to a pin on the bbox edge now ends ~pad px outside it:
        # shift edge pins outward by the same amount before snapping.
        pad = self.config.vectorization.snap_radius
        pins = []
        for comp in components:
            x0, y0, x1, y1 = comp.bbox
            for pin in comp.pins:
                px, py = pin.coord
                if comp.type not in (ComponentType.JUNCTION_DOT,):
                    # nearest bbox edge (pins sit on the edge, ±rounding)
                    edges = [(px - x0, -1, 0), (x1 - px, 1, 0), (py - y0, 0, -1), (y1 - py, 0, 1)]
                    gap, sx, sy = min(edges, key=lambda e: e[0])
                    if gap <= 3:
                        px += sx * pad
                        py += sy * pad
                pins.append(PinReference(
                    component_id=comp.id,
                    pin_number=pin.pin_number,
                    coord=(max(0, int(px)), max(0, int(py))),
                ))
        builder.add_component_pins(pins)
        
        # Snap and build
        builder.snap_wire_to_pins()
        if self.config.graph_synthesis.merge_collinear_segments:
            builder.merge_collinear_segments()
        
        graph = builder.build_graph()
        nets_data = builder.extract_nets([c.model_dump() for c in components])
        
        # Junction dots are wire nodes, not netlist members.
        dot_ids = {c.id for c in components if c.type == ComponentType.JUNCTION_DOT}
        nets = []
        for net_data in nets_data:
            connections = [c for c in net_data.get("connections", [])
                           if c["component_id"] not in dot_ids]
            # A lone unconnected pin (or dot) is not a net.
            if not net_data.get("wire_segments") and len(connections) <= 1:
                continue
            nets.append(Net(
                net_id=net_data["net_id"],
                connections=[WireConnection(**c) for c in connections],
                path_points=net_data.get("path_points", []),
                wire_segments=net_data.get("wire_segments", []),
                confidence=net_data.get("confidence", 1.0),
            ))
        
        return nets
    
    def _run_vlm_arbitration(
        self,
        image: np.ndarray,
        manifest: AVERSManifest,
    ) -> None:
        """Run VLM arbitration for uncertain cases with Vision RAG."""
        if not manifest.human_review_required:
            return
        
        max_calls = self.config.vlm_arbitrator.max_vlm_calls
        issues_to_process = manifest.human_review_required[:max_calls]
        
        # Try Vision RAG if enabled
        use_rag = False
        rag = None
        if hasattr(self.config, 'vision_rag') and self.config.vision_rag.enabled:
            try:
                from avers.rag import get_rag
                rag = get_rag()
                use_rag = True
                logger.info(f"Using Vision RAG with {rag.stats().get('total', 0)} examples")
            except Exception as e:
                logger.warning(f"Vision RAG not available: {e}")
        
        # Try real VLM if available
        vlm_wrapper = None
        if self.config.vlm_arbitrator.enabled:
            try:
                from avers.stages.stage6_vlm_arbitrator import VLMWrapper, VLMConfig, ArbitrationEngine
                vlm_config = VLMConfig(
                    model_name=self.config.vlm_arbitrator.model_name,
                    device=self.config.vlm_arbitrator.device,
                    roi_size=self.config.vlm_arbitrator.roi_size,
                    max_calls=max_calls,
                )
                vlm_wrapper = VLMWrapper(vlm_config)
                vlm_wrapper.load()
                engine = ArbitrationEngine(vlm_wrapper)
                logger.info(f"VLM loaded: {self.config.vlm_arbitrator.model_name}")
            except Exception as e:
                logger.warning(f"VLM not available, using RAG/mock: {e}")
        
        # Process each issue
        for issue in issues_to_process:
            try:
                # Extract ROI
                x1, y1, x2, y2 = issue.bbox
                # Expand for context
                h, w = image.shape[:2]
                cx, cy = (x1+x2)//2, (y1+y2)//2
                roi_size = self.config.vlm_arbitrator.roi_size
                x1_roi = max(0, cx - roi_size//2)
                y1_roi = max(0, cy - roi_size//2)
                x2_roi = min(w, cx + roi_size//2)
                y2_roi = min(h, cy + roi_size//2)
                roi = image[y1_roi:y2_roi, x1_roi:x2_roi]
                
                if roi.size == 0:
                    continue
                
                # Resize to standard
                import cv2
                roi_resized = cv2.resize(roi, (roi_size, roi_size))
                
                # RAG query first
                rag_answer = None
                if use_rag and rag:
                    try:
                        rag_response = rag.query(
                            image=roi_resized,
                            text=issue.description,
                            top_k=self.config.vision_rag.top_k if hasattr(self.config, 'vision_rag') else 5,
                            use_vlm=vlm_wrapper is not None
                        )
                        if rag_response.vlm_answer:
                            rag_answer = rag_response.vlm_answer
                            logger.debug(f"RAG answer for {issue.bbox}: {rag_answer}")
                    except Exception as e:
                        logger.warning(f"RAG query failed: {e}")
                
                # VLM arbitration
                if vlm_wrapper and 'engine' in locals():
                    # Use engine
                    if issue.issue_type.value == "suspicious_crossing":
                        result = engine.resolve_crossing(image, issue.bbox)
                    elif issue.issue_type.value == "low_confidence_text":
                        result = engine.resolve_text(image, issue.bbox, issue.suggestions)
                    else:
                        result = engine.resolve_junction(image, issue.bbox)
                    
                    if result.resolved:
                        issue.resolved = True
                        if result.connected is not None:
                            issue.resolution = f"connected={result.connected}, conf={result.confidence:.2f}, rag={rag_answer is not None}"
                        else:
                            issue.resolution = f"text={result.selected_text}, conf={result.confidence:.2f}"
                elif rag_answer:
                    # Use RAG answer
                    issue.resolved = True
                    if "connected" in rag_answer:
                        issue.resolution = f"rag_connected={rag_answer['connected']}, conf={rag_answer.get('confidence', 0.5):.2f}, method=rag"
                    else:
                        issue.resolution = f"rag_{rag_answer}"
                else:
                    # Mock fallback
                    issue.resolved = True
                    issue.resolution = "mock_vlm_resolution_no_vlm"
            
            except Exception as e:
                logger.warning(f"Failed to arbitrate issue {issue.bbox}: {e}")
                issue.resolved = False
    
    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center of bounding box."""
        return (
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2,
        )
    
    @staticmethod
    def _category_to_component_type(category: str) -> ComponentType:
        """Map detection category to ComponentType."""
        from avers.core.types import CLASS_TO_COMPONENT_TYPE
        return CLASS_TO_COMPONENT_TYPE.get(category, ComponentType.UNKNOWN)


# =============================================================================
# Convenience Functions
# =============================================================================

def process_schematic_production(
    image: np.ndarray,
    source_file: str = "unknown",
    dpi: int = 300,
    config: Optional[AVERSConfig] = None,
) -> PipelineResult:
    """
    Process schematic with production pipeline.
    
    Args:
        image: Input image (H, W, 3) RGB
        source_file: Source filename
        dpi: Image DPI
        config: Pipeline config
        
    Returns:
        PipelineResult
    """
    pipeline = ProductionPipeline(config)
    return pipeline.run(image, source_file, dpi)


def load_and_process(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[AVERSConfig] = None,
    pdf_page: int = 0,
    pdf_process_all: bool = False,
) -> PipelineResult:
    """
    Load image/PDF and process with production pipeline.
    
    Supports:
      - Images: PNG, JPG, TIF, etc.
      - PDFs: single page (pdf_page) or all pages (pdf_process_all)
    
    Args:
        image_path: Path to image or PDF file
        output_path: Path for output (optional)
        config: Pipeline config
        pdf_page: Page number for PDF (0-indexed)
        pdf_process_all: Process all PDF pages and merge
        
    Returns:
        PipelineResult
    """
    from avers.utils.image_helpers import load_image_auto, get_image_info, is_pdf
    from PIL import Image
    
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Get info
    info = get_image_info(path)
    is_pdf_file = info.get("is_pdf", False) or is_pdf(path)
    
    pipeline = ProductionPipeline(config)
    
    if is_pdf_file and pdf_process_all:
        # Process all PDF pages and merge
        pages = load_image_auto(path, dpi=300)
        logger.info(f"Processing {len(pages)} pages from PDF {path}")
        
        merged_components = []
        merged_nets = []
        merged_issues = []
        total_timings = {}
        errors = []
        warnings = [f"PDF with {len(pages)} pages"]
        
        for page_idx, page_image in enumerate(pages):
            result = pipeline.run(page_image, f"{path.name}_page_{page_idx}", dpi=300)
            
            # Prefix IDs with page number
            for comp in result.manifest.components:
                comp.id = f"p{page_idx}_{comp.id}"
                comp.text_associations["pdf_page"] = str(page_idx)
            for net in result.manifest.nets:
                net.net_id = f"p{page_idx}_{net.net_id}"
            for issue in result.manifest.human_review_required:
                issue.description = f"[Page {page_idx}] {issue.description}"
            
            merged_components.extend(result.manifest.components)
            merged_nets.extend(result.manifest.nets)
            merged_issues.extend(result.manifest.human_review_required)
            
            for k, v in result.stage_timings.items():
                total_timings[k] = total_timings.get(k, 0) + v
            
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        # Create merged manifest
        first_page = pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
        merged_manifest = AVERSManifest(
            schema_metadata=SchemaMetadata(
                source_file=path.name,
                resolution_dpi=300,
                width=first_page.shape[1],
                height=first_page.shape[0],
                format="PDF",
            ),
            components=merged_components,
            nets=merged_nets,
            human_review_required=merged_issues,
        )
        
        result = PipelineResult(
            manifest=merged_manifest,
            errors=errors,
            warnings=warnings,
            stage_timings=total_timings,
            success=len(errors) == 0,
        )
    
    elif is_pdf_file:
        # Single PDF page
        pages = load_image_auto(path, dpi=300, page_numbers=[pdf_page])
        if not pages:
            raise ValueError(f"Failed to load page {pdf_page} from {path}")
        image = pages[0]
        
        # Try to get DPI from PDF info if available
        dpi = 300
        try:
            from avers.utils.pdf_loader import PDFLoader
            loader = PDFLoader()
            # PDF DPI is not stored, use 300 as default
        except Exception:
            pass
        
        result = pipeline.run(image, f"{path.name}_page_{pdf_page}", dpi=dpi)
    
    else:
        # Regular image
        pil_img = Image.open(path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        image = np.array(pil_img)
        dpi = pil_img.info.get("dpi", (300, 300))
        if isinstance(dpi, tuple):
            dpi = int(dpi[0])
        else:
            try:
                dpi = int(dpi)
            except Exception:
                dpi = 300
        
        result = pipeline.run(image, path.name, dpi)
    
    # Save output
    if output_path:
        result.manifest.save(output_path, format="xml" if str(output_path).endswith(".xml") else "json")
    
    return result


def load_and_process_pdf(
    pdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[AVERSConfig] = None,
    page_numbers: Optional[List[int]] = None,
    process_all: bool = False,
) -> Union[PipelineResult, List[PipelineResult]]:
    """
    Load PDF and process pages.
    
    Args:
        pdf_path: Path to PDF
        output_path: Output path (for merged result if process_all)
        config: Pipeline config
        page_numbers: Specific pages to process
        process_all: Merge all pages into single manifest
    
    Returns:
        Single PipelineResult (if process_all) or List[PipelineResult] (per page)
    """
    from avers.utils.image_helpers import load_image_auto
    
    pdf_path = Path(pdf_path)
    pages = load_image_auto(pdf_path, dpi=300, page_numbers=page_numbers)
    
    pipeline = ProductionPipeline(config)
    
    if process_all:
        return load_and_process(pdf_path, output_path, config, pdf_process_all=True)
    else:
        results = []
        for i, page_img in enumerate(pages):
            result = pipeline.run(page_img, f"{pdf_path.name}_page_{i}", dpi=300)
            if output_path:
                # Save per-page
                out_path = Path(output_path)
                page_out = out_path.parent / f"{out_path.stem}_page_{i}{out_path.suffix}"
                result.manifest.save(page_out)
            results.append(result)
        return results
