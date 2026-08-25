"""
Stage 2: УГО Detection with RT-DETR/YOLO + SAHI

Full implementation of component detection using sliding window inference.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import cv2

from avers.stages.stage1_slicing import Tile, DetectionBox, SlicingEngine
from avers.core.types import Component, Pin, ComponentType, BoundingBox
from avers.core.logger import get_logger

logger = get_logger("avers.detection")


# Detection class mapping (ГОСТ УГО)
DETECTION_CLASS_MAP = {
    0: ("connector_body", ComponentType.CONNECTOR),
    1: ("pin", ComponentType.UNKNOWN),
    2: ("junction_dot", ComponentType.JUNCTION_DOT),
    3: ("ground", ComponentType.GROUND),
    4: ("shield", ComponentType.SHIELD),
    5: ("offpage_connector", ComponentType.OFFPAGE_CONNECTOR),
    6: ("diode", ComponentType.DIODE),
    7: ("relay", ComponentType.RELAY),
    8: ("resistor", ComponentType.RESISTOR),
}

CLASS_NAMES = {k: v[0] for k, v in DETECTION_CLASS_MAP.items()}


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    model_path: Optional[str] = None
    model_type: str = "yolo"  # 'yolo' or 'rtdetr'
    model_name: str = "yolov11x"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    batch_size: int = 4
    img_size: int = 640


@dataclass
class DetectionResult:
    """Result of detection on a tile."""
    tile_id: int
    detections: List[DetectionBox] = field(default_factory=list)
    processing_time_ms: float = 0.0


class YOLODetector:
    """
    YOLO detector wrapper for УГО detection.
    
    Supports:
    - YOLOv8/YOLOv11
    - RT-DETR models via ultralytics
    - ONNX Runtime for CPU inference
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load detection model."""
        if self._loaded:
            return True
        
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            if self.config.model_path and Path(self.config.model_path).exists():
                model_path = self.config.model_path
                logger.info(f"Loading model from {model_path}")
            else:
                # Use default pretrained model
                model_path = self.config.model_name
                logger.info(f"Using default model: {model_path}")
            
            self.model = YOLO(model_path)
            
            # Move to device
            if self.config.device != "cpu":
                try:
                    self.model.to(self.config.device)
                except Exception as e:
                    logger.warning(f"Could not move model to {self.config.device}: {e}")
            
            self._loaded = True
            logger.info(f"Model loaded successfully")
            return True
            
        except ImportError:
            logger.warning("ultralytics not installed, using mock detector")
            self._loaded = True
            self.model = None
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> List[DetectionBox]:
        """
        Run detection on a single image.
        
        Args:
            image: Input image (H, W, 3) RGB
            
        Returns:
            List of DetectionBox objects
        """
        if not self._loaded:
            self.load()
        
        if self.model is None:
            # Mock detection for testing
            return self._mock_detect(image)
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                verbose=False,
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    detections.append(DetectionBox(
                        x_min=int(x1),
                        y_min=int(y1),
                        x_max=int(x2),
                        y_max=int(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=CLASS_NAMES.get(cls_id, "unknown"),
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _mock_detect(self, image: np.ndarray) -> List[DetectionBox]:
        """
        Generate mock detections for testing without actual model.
        
        Creates plausible УГО-like detections based on contours.
        """
        detections = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Classify by aspect ratio and size
            aspect = w / max(h, 1)
            
            if 0.8 <= aspect <= 1.2 and area < 5000:
                # Likely a junction dot
                cls_id = 2
            elif w > h * 2 and area > 5000:
                # Likely a connector body
                cls_id = 0
            elif area > 20000:
                # Large component
                cls_id = 7  # relay
            else:
                cls_id = 6  # diode
            
            detections.append(DetectionBox(
                x_min=int(x),
                y_min=int(y),
                x_max=int(x + w),
                y_max=int(y + h),
                confidence=0.7,
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, "unknown"),
            ))
        
        return detections


class SlicingDetector:
    """
    SAHI-style tiled detection for large schematic images.
    
    Processes large images in overlapping tiles and projects
    detections back to global coordinate space.
    """
    
    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        slicing_engine: Optional[SlicingEngine] = None,
        nms_threshold: float = 0.45,
    ):
        self.detector = detector or YOLODetector(DetectionConfig())
        self.slicing_engine = slicing_engine or SlicingEngine()
        self.nms_threshold = nms_threshold
    
    def detect_in_tiles(
        self,
        image: np.ndarray,
        parallel: bool = True,
        num_workers: int = 4,
    ) -> List[DetectionBox]:
        """
        Detect objects in image using tiled inference.
        
        Args:
            image: Full input image
            parallel: Use parallel processing
            num_workers: Number of parallel workers
            
        Returns:
            List of detections in global coordinates
        """
        import time
        
        logger.info(f"Starting tiled detection on {image.shape[1]}x{image.shape[0]} image")
        
        # Generate tiles
        tiles = list(self.slicing_engine.generate_tiles(image))
        logger.info(f"Generated {len(tiles)} tiles")
        
        # Ensure detector is loaded
        self.detector.load()
        
        all_detections = []
        total_time = 0
        
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._detect_tile, tile): tile
                    for tile in tiles
                }
                
                for future in as_completed(futures):
                    tile = futures[future]
                    try:
                        result = future.result()
                        all_detections.extend(result.detections)
                        total_time += result.processing_time_ms
                    except Exception as e:
                        logger.error(f"Tile {tile.tile_id} failed: {e}")
        else:
            for tile in tiles:
                result = self._detect_tile(tile)
                all_detections.extend(result.detections)
                total_time += result.processing_time_ms
        
        logger.info(f"Detected {len(all_detections)} total objects before NMS")
        
        # Project to global coordinates
        global_detections = self.slicing_engine.project_detections_to_global(
            all_detections, tiles
        )
        
        # NMS deduplication
        filtered = self.slicing_engine.nms(global_detections, self.nms_threshold)
        
        logger.info(f"After NMS: {len(filtered)} unique detections ({total_time:.1f}ms)")
        
        return filtered
    
    def _detect_tile(self, tile: Tile) -> DetectionResult:
        """Detect objects in a single tile."""
        import time
        
        start = time.time()
        detections = self.detector.predict(tile.image)
        
        # Set tile ID on all detections
        for det in detections:
            det.tile_id = tile.tile_id
        
        return DetectionResult(
            tile_id=tile.tile_id,
            detections=detections,
            processing_time_ms=(time.time() - start) * 1000,
        )


def group_detections_into_components(
    detections: List[DetectionBox],
    min_pin_distance: float = 50.0,
) -> List[Component]:
    """
    Group raw detections into semantic components.
    
    Logic:
    - connector_body + nearby pins -> single Component with pins
    - junction_dot -> standalone junction component
    - other types -> individual components
    """
    components = []
    used_detections = set()
    next_id = 1
    
    # First, find connector bodies and their associated pins
    connectors = [d for d in detections if d.class_id == 0]  # connector_body
    pins = [d for d in detections if d.class_id == 1]  # pin
    
    for conn in connectors:
        if id(conn) in used_detections:
            continue
        
        # Find pins inside or near this connector
        component_pins = []
        conn_center = conn.center
        
        for pin in pins:
            if id(pin) in used_detections:
                continue
            
            pin_center = pin.center
            dist = np.sqrt(
                (conn_center[0] - pin_center[0])**2 +
                (conn_center[1] - pin_center[1])**2
            )
            
            if dist < min_pin_distance * 3:  # Pins should be close to connector
                # Determine pin number from position (simplified)
                # In real implementation, would use OCR or spatial ordering
                pin_num = str(len(component_pins) + 1)
                
                component_pins.append(Pin(
                    pin_number=pin_num,
                    coord=pin.center,
                    confidence=pin.confidence,
                ))
                used_detections.add(id(pin))
        
        # Create component
        component = Component(
            id=f"comp_{next_id:03d}",
            designator=f"X{next_id}",  # Would be determined by OCR
            type=ComponentType.CONNECTOR,
            bbox=conn.bbox,
            pins=component_pins,
            confidence=conn.confidence,
        )
        
        components.append(component)
        used_detections.add(id(conn))
        next_id += 1
    
    # Process remaining detections as individual components
    for det in detections:
        if id(det) in used_detections:
            continue
        
        class_name, comp_type = DETECTION_CLASS_MAP.get(det.class_id, ("unknown", ComponentType.UNKNOWN))
        
        # Skip pins that weren't assigned to connectors
        if det.class_id == 1:
            continue
        
        component = Component(
            id=f"comp_{next_id:03d}",
            designator=f"{class_name[:2].upper()}{next_id}",
            type=comp_type,
            bbox=det.bbox,
            confidence=det.confidence,
        )
        
        components.append(component)
        used_detections.add(id(det))
        next_id += 1
    
    return components


def detect_components(
    image: np.ndarray,
    config: Optional[DetectionConfig] = None,
    use_slicing: bool = True,
) -> Tuple[List[Component], List[DetectionBox]]:
    """
    Full component detection pipeline.
    
    Args:
        image: Input image (H, W, 3) RGB
        config: Detection configuration
        use_slicing: Use SAHI slicing for large images
        
    Returns:
        Tuple of (components, raw_detections)
    """
    config = config or DetectionConfig()
    
    detector = YOLODetector(config)
    slicing_detector = SlicingDetector(detector=detector)
    
    # Detect with tiled inference
    detections = slicing_detector.detect_in_tiles(image)
    
    # Group into semantic components
    components = group_detections_into_components(detections)
    
    return components, detections
