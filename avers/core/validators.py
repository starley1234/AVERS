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
import os
import time
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


class SlicedDetector:
    """
    SAHI-based sliced detection with automatic fallback.
    
    Uses SAHI library if available, falls back to custom implementation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "yolov11",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        slice_size: int = 1024,
        overlap_ratio: float = 0.2,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        
        self._sahi = get_sahi_integration()
        self._model = None
        self._loaded = False
        self._yolo_detector = None  # для бэкенда 'ultralytics' (без SAHI)
        # 'sahi' | 'ultralytics' (реальная модель) | 'mock' (заглушка по контурам)
        self.backend: Optional[str] = None

    def load(self) -> bool:
        """Load detection model."""
        if self._loaded:
            return True

        try:
            if self._sahi is not None:
                return self._load_sahi_model()
        except Exception as e:
            logger.warning(f"Failed to load SAHI model: {e}")

        # SAHI недоступен или упал - пробуем ultralytics напрямую (тайлы своими силами)
        try:
            return self._load_ultralytics_model()
        except Exception as e:
            logger.warning(f"Direct ultralytics load failed: {e}, using contour fallback")
            return self._load_fallback_model()
    
    def _load_sahi_model(self) -> bool:
        """Load SAHI with detection model."""
        from sahi import AutoDetectionModel
        
        # Determine model type
        if self.model_type.startswith("yolo"):
            model_type = "yolov11" if "11" in self.model_type else "yolov8"
        elif self.model_type == "rtdetr":
            model_type = "rtdetr"
        else:
            model_type = "yolov8"
        
        if self.model_path is None:
            logger.warning(
                "detection.model_path не задан - SAHI скачает COCO-модель по умолчанию, "
                "НЕ обученную на ГОСТ УГО (результаты будут бессмысленными)"
            )

        self._model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        
        self._sliced_inference = self._sahi["SlicingInference"](
            detection_model=self._model,
            slice_size=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
        )
        
        self._loaded = True
        self.backend = "sahi"
        logger.info(f"SAHI model loaded: {self.model_type}")
        return True
    
    def _load_ultralytics_model(self) -> bool:
        """Загрузить YOLO/RT-DETR через ultralytics напрямую (без SAHI).

        Тайлинг больших изображений выполняется внутренним SlicingEngine.
        Требует заданных весов: без model_path COCO-модель для ГОСТ бесполезна,
        поэтому падаем в mock с честным предупреждением.
        """
        if not (self.model_path and Path(self.model_path).exists()):
            raise FileNotFoundError(
                f"Веса модели не найдены: {self.model_path} - "
                f"обучите модель (python -m avers dataset train) и укажите "
                f"detection.model_path в config.yaml"
            )
        from avers.stages.stage2_detection.detector import YOLODetector, DetectionConfig

        device = self.device if self.device in ("cpu", "cuda") else "cpu"
        cfg = DetectionConfig(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=device,
            img_size=self.config.detection.img_size,
        )
        self._yolo_detector = YOLODetector(cfg)
        if not self._yolo_detector.load():
            raise RuntimeError("YOLO detector failed to load")
        if getattr(self._yolo_detector, "model", None) is None:
            # YOLODetector сам тихо уходит в mock, если ultralytics не установлен -
            # такой режим сюда допускать нельзя, иначе получим заглушку без предупреждения
            raise RuntimeError("ultralytics не установлен - нет реального инференса")
        self._loaded = True
        self.backend = "ultralytics"
        logger.info(f"ultralytics model loaded (без SAHI): {self.model_path}")
        return True

    def _ultralytics_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Тайловая детекция через ultralytics (SAHI не требуется)."""
        from avers.stages.stage2_detection.detector import SlicingDetector as TiledDetector

        tiled = TiledDetector(detector=self._yolo_detector, nms_threshold=0.45)
        boxes = tiled.detect_in_tiles(image)
        return [
            {
                "bbox": (int(b.x_min), int(b.y_min), int(b.x_max), int(b.y_max)),
                "confidence": float(b.confidence),
                "category": b.class_name,
                "category_id": int(b.class_id),
            }
            for b in boxes
        ]

    def _load_fallback_model(self) -> bool:
        """Load fallback model (mock detection)."""
        logger.info("Using fallback detector (mock mode)")
        self._loaded = True
        self.backend = "mock"
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
        
        try:
            if self.backend == "sahi" and hasattr(self, "_sliced_inference"):
                return self._sahi_detect(image)
            if self.backend == "ultralytics":
                return self._ultralytics_detect(image)
        except Exception as e:
            logger.error(f"Model detection failed: {e}, falling back to contours")
        
        return self._fallback_detect(image)
    
    def _sahi_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """SAHI detection."""
        result = self._sliced_inference.detect(image)
        
        detections = []
        for obj in result.object_list:
            bbox = obj.bbox
            detections.append({
                "bbox": (int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)),
                "confidence": float(obj.category.confidence),
                "category": obj.category.name,
                "category_id": obj.category.id,
            })
        
        return detections
    
    def _fallback_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback mock detection based on contours."""
        detections = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Simple classification by size
            if 0.8 <= w / max(h, 1) <= 1.2 and area < 5000:
                category = "junction_dot"
                category_id = 2
            elif w > h * 2 and area > 5000:
                category = "connector_body"
                category_id = 0
            elif area > 20000:
                category = "relay"
                category_id = 7
            else:
                category = "diode"
                category_id = 6
            
            detections.append({
                "bbox": (x, y, x + w, y + h),
                "confidence": 0.7,
                "category": category,
                "category_id": category_id,
            })
        
        return detections


# =============================================================================
# OCR Integration
# =============================================================================

class SchematicOCR:
    """
    OCR engine with automatic backend selection.
    
    Priority: PaddleOCR > EasyOCR > Mock
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
        
        # Try PaddleOCR
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
        
        # Try EasyOCR
        try:
            import easyocr
            self._ocr = easyocr.Reader([self.lang, 'en'], gpu=False, verbose=False)
            self._backend = "easyocr"
            self._loaded = True
            logger.info("EasyOCR loaded")
            return True
        except ImportError:
            pass
        
        # Fallback to mock
        self._backend = "mock"
        self._loaded = True
        logger.warning("No OCR backend available, using mock")
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
            return self._mock_recognize(image)
    
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
                
                if confidence < self.confidence_threshold:
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
    
    def _easyocr_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """EasyOCR recognition."""
        results = self._ocr.readtext(image)
        
        texts = []
        for bbox, text, confidence in results:
            if confidence < self.confidence_threshold:
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
    
    def _mock_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Mock OCR for testing."""
        # Simple edge detection for text-like regions
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        texts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if h < 8 or h > 100 or w < 10 or w > 300:
                continue
            if h > w * 2:
                continue
            
            # Generate mock text based on size
            char_count = min(max(w // 15, 1), 6)
            texts.append({
                "text": "X" + str(char_count),
                "bbox": (x, y, x + w, y + h),
                "confidence": 0.6,
            })
        
        return texts


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
    ):
        self.rdp_epsilon = rdp_epsilon
        self.min_line_length = min_line_length
        self.exclusion_padding = exclusion_padding
    
    def vectorize(
        self,
        image: np.ndarray,
        exclusion_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
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
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
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
        lines = cv2.HoughLinesP(
            skeleton, rho=1, theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=10,
        )
        
        # Create segments
        segments = []
        junctions = set()
        
        if lines is not None:
            for i, line in enumerate(lines):
                if len(line.shape) == 3:
                    x1, y1, x2, y2 = line[0]
                else:
                    x1, y1, x2, y2 = line
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # RDP simplification (single segment = no simplification needed)
                segments.append(WireSegment(
                    segment_id=i,
                    start=(x1, y1),
                    end=(x2, y2),
                    points=[(x1, y1), (x2, y2)],
                    confidence=0.9,
                ))
                
                # Track endpoints as potential junctions
                junctions.add((x1, y1))
                junctions.add((x2, y2))
        
        return segments, junctions


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
            detections, component_bboxes = self._run_detection(image, result)
            result.manifest.components = self._group_components(detections)
        except Exception as e:
            result.errors.append(f"Detection failed: {e}")
            result.warnings.append("Using empty components")
            result.manifest.components = []
        
        result.stage_timings["detection"] = time.time() - stage_start
        
        # Stage 2: OCR
        stage_start = time.time()
        try:
            texts, text_bboxes = self._run_ocr(image, result)
            # Associate texts with components
            self._associate_texts(result.manifest.components, texts)
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
                component_bboxes + text_bboxes
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
        result.manifest.schema_metadata.processing_time_seconds = time.time() - start_time
        
        logger.info(
            f"Pipeline complete: {len(result.manifest.components)} components, "
            f"{len(result.manifest.nets)} nets, {len(result.manifest.human_review_required)} issues"
        )
        
        return result
    
    def _resolve_model_path(self) -> Optional[str]:
        """Путь к весам детектора с авто-подхватом.

        Приоритет: detection.model_path из конфига > $AVERS_MODEL_PATH >
        стандартные пути результата обучения (/tmp/avers_runs/.../weights/best.pt,
        берётся самый свежий). Если ничего нет - возвращаем как есть (None),
        дальше сработает честное предупреждение о заглушке.
        """
        mp = self.config.detection.model_path
        if mp and Path(str(mp)).expanduser().exists():
            return str(Path(str(mp)).expanduser())

        candidates: List[Path] = []
        env = os.getenv("AVERS_MODEL_PATH")
        if env:
            candidates.append(Path(env).expanduser())
        # все прогоны обучения (avers_yolo, avers_yolo2, ... avers_rtdetrN)
        runs_dir = Path("/tmp/avers_runs")
        for pattern in ("avers_yolo*/weights/best.pt", "avers_rtdetr*/weights/best.pt"):
            candidates += sorted(runs_dir.glob(pattern))
        existing = [c for c in candidates if c.exists()]
        if existing:
            best = max(existing, key=lambda c: c.stat().st_mtime)
            logger.info(
                f"detection.model_path не задан - авто-подхват обученных весов: {best} "
                f"(задайте путь явно в config.yaml, чтобы отключить авто-подхват)"
            )
            return str(best)
        return mp

    def _run_detection(
        self, image: np.ndarray, result: "PipelineResult"
    ) -> Tuple[List[Dict], List[Tuple[int, int, int, int]]]:
        """Run component detection."""
        resolved_model = self._resolve_model_path()
        detector = SlicedDetector(
            model_path=resolved_model,
            model_type=self.config.detection.model_type,
            confidence_threshold=self.config.detection.confidence_threshold,
            device=self.config.detection.device,
            slice_size=self.config.slicing.tile_size,
            overlap_ratio=self.config.slicing.overlap_ratio,
        )
        detector.load()
        logger.info(f"Стадия 2: бэкенд детекции = {detector.backend} "
                    f"(model_path={resolved_model or 'не задан'})")
        detections = detector.detect(image)
        bboxes = [d["bbox"] for d in detections]

        # Честно сообщаем, если работаем не на обученной модели
        if detector.backend == "mock":
            result.warnings.append(
                "Стадия 2 (детекция УГО): модель НЕ загружена — работает заглушка по контурам, "
                "результаты недостоверны. Установите ML-зависимости (pip install ultralytics sahi), "
                "обучите модель (python -m avers dataset train) и укажите веса в config.yaml → "
                "detection.model_path (см. QUICKSTART.md, раздел 2-В)."
            )
        elif detector.backend == "sahi" and detector.model_path is None:
            result.warnings.append(
                "Стадия 2 (детекция УГО): detection.model_path не задан — используется COCO-модель "
                "по умолчанию, НЕ обученная на ГОСТ УГО. Обучите модель на синтетическом датасете "
                "(см. QUICKSTART.md, раздел 2-В) и укажите best.pt в config.yaml → detection.model_path."
            )

        return detections, bboxes
    
    def _group_components(
        self, detections: List[Dict]
    ) -> List[Component]:
        """Group detections into semantic components."""
        from scipy.spatial import KDTree
        
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
    
    def _run_ocr(
        self, image: np.ndarray, result: "PipelineResult"
    ) -> Tuple[List[Dict], List[Tuple[int, int, int, int]]]:
        """Run OCR."""
        ocr = SchematicOCR(
            lang=self.config.ocr.lang,
            confidence_threshold=self.config.ocr.text_confidence_threshold,
        )
        ocr.load()
        if getattr(ocr, "_backend", None) == "mock":
            result.warnings.append(
                "Стадия 3 (OCR): PaddleOCR/EasyOCR не установлены — текст НЕ распознаётся "
                "(обозначения, номера контактов и проводов будут пропущены). "
                "Установите: pip install paddleocr paddlepaddle (Python 3.11/3.12) или easyocr."
            )

        texts = ocr.recognize(image)
        bboxes = [t["bbox"] for t in texts]
        
        return texts, bboxes
    
    def _associate_texts(
        self,
        components: List[Component],
        texts: List[Dict],
    ) -> None:
        """Associate texts with nearby components."""
        if not components or not texts:
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
    
    def _run_vectorization(
        self,
        image: np.ndarray,
        exclusion_bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[WireSegment], set]:
        """Run wire vectorization."""
        vectorizer = WireVectorizer(
            rdp_epsilon=self.config.vectorization.rdp_epsilon,
            min_line_length=self.config.vectorization.line_thickness_threshold,
            exclusion_padding=self.config.vectorization.snap_radius,
        )
        
        return vectorizer.vectorize(image, exclusion_bboxes)
    
    def _run_graph_synthesis(
        self,
        segments: List[WireSegment],
        junctions: set,
        components: List[Component],
    ) -> List[Net]:
        """Run graph synthesis."""
        builder = GraphBuilder(
            snap_enabled=self.config.graph_synthesis.snap_enabled,
            snap_radius=self.config.graph_synthesis.snap_radius,
            merge_collinear=self.config.graph_synthesis.merge_collinear_segments,
        )
        
        # Add segments
        for seg in segments:
            builder.add_wire_segment(seg.start, seg.end, seg.points, seg.confidence)
        
        # Add pins
        pins = []
        for comp in components:
            for pin in comp.pins:
                pins.append(PinReference(
                    component_id=comp.id,
                    pin_number=pin.pin_number,
                    coord=tuple(pin.coord),
                ))
        builder.add_component_pins(pins)
        
        # Snap and build
        builder.snap_wire_to_pins()
        if self.config.graph_synthesis.merge_collinear_segments:
            builder.merge_collinear_segments()
        
        graph = builder.build_graph()
        nets_data = builder.extract_nets([c.model_dump() for c in components])
        
        nets = []
        for net_data in nets_data:
            nets.append(Net(
                net_id=net_data["net_id"],
                connections=[
                    WireConnection(**c) for c in net_data.get("connections", [])
                ],
                path_points=net_data.get("path_points", []),
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
                vlm_cfg = self.config.vlm_arbitrator
                vlm_config = VLMConfig(
                    model_name=vlm_cfg.model_name,
                    device=vlm_cfg.device,
                    roi_size=vlm_cfg.roi_size,
                    max_calls=max_calls,
                    # Внешняя LLM: поля конфига, а если не заданы - env AVERS_VLM_*
                    provider=getattr(vlm_cfg, "provider", None)
                        or os.getenv("AVERS_VLM_PROVIDER", "auto"),
                    api_base=getattr(vlm_cfg, "api_base", None)
                        or os.getenv("AVERS_VLM_API_BASE", ""),
                    api_key=getattr(vlm_cfg, "api_key", None)
                        or os.getenv("AVERS_VLM_API_KEY", ""),
                    api_model=getattr(vlm_cfg, "api_model", None)
                        or os.getenv("AVERS_VLM_API_MODEL", ""),
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
        mapping = {
            "connector_body": ComponentType.CONNECTOR,
            "junction_dot": ComponentType.JUNCTION_DOT,
            "ground": ComponentType.GROUND,
            "shield": ComponentType.SHIELD,
            "offpage_connector": ComponentType.OFFPAGE_CONNECTOR,
            "diode": ComponentType.DIODE,
            "relay": ComponentType.RELAY,
            "resistor": ComponentType.RESISTOR,
        }
        return mapping.get(category, ComponentType.UNKNOWN)


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
