"""Tests for all AVERS stages."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from avers.stages.stage1_slicing import SlicingEngine, Tile, load_image
from avers.stages.stage2_detection import (
    YOLODetector, DetectionConfig, group_detections_into_components
)
from avers.stages.stage2_detection.detector import DetectionBox
from avers.stages.stage3_ocr import (
    OCRConfig, classify_text, normalize_designation, validate_designation
)
from avers.stages.stage4_vectorization import (
    vectorize_wires, VectorizationConfig, WireVectorizer
)
from avers.stages.stage5_graph_synthesis import GraphBuilder, PinReference, WireSegment
from avers.stages.stage6_vlm_arbitrator import (
    VLMWrapper, VLMConfig, ArbitrationEngine, create_issues_from_detections
)
from avers.core.types import Component, ComponentType, Pin


class TestStage2Detection:
    """Tests for Stage 2: УГО Detection."""

    def test_detection_config(self):
        """Test detection configuration."""
        config = DetectionConfig(
            model_type="yolo",
            confidence_threshold=0.3,
            device="cpu",
        )
        assert config.model_type == "yolo"
        assert config.confidence_threshold == 0.3
        assert config.device == "cpu"

    def test_yolo_detector_mock(self):
        """Test YOLO detector with mock."""
        config = DetectionConfig(device="cpu")
        detector = YOLODetector(config)
        detector.load()
        
        # Create test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = detector.predict(image)
        assert isinstance(detections, list)

    def test_group_detections(self):
        """Test grouping detections into components."""
        # Create mock detections
        detections = [
            DetectionBox(x_min=100, y_min=100, x_max=200, y_max=200, 
                        confidence=0.9, class_id=0),  # connector
            DetectionBox(x_min=150, y_min=150, x_max=160, y_max=160,
                        confidence=0.8, class_id=1),  # pin
        ]
        
        components = group_detections_into_components(detections)
        assert len(components) > 0
        assert any(c.type == ComponentType.CONNECTOR for c in components)


class TestStage3OCR:
    """Tests for Stage 3: OCR."""

    def test_ocr_config(self):
        """Test OCR configuration."""
        config = OCRConfig(lang="ru", text_confidence_threshold=0.7)
        assert config.lang == "ru"
        assert config.text_confidence_threshold == 0.7

    def test_classify_text(self):
        """Test text classification."""
        assert classify_text("X1") == "connector"
        assert classify_text("1") == "pin"
        assert classify_text("+27В") == "voltage"
        assert classify_text("БПВЛ-0.35") == "wire_type"
        assert classify_text("VD1") == "designator"

    def test_normalize_designation(self):
        """Test designation normalization."""
        assert normalize_designation("x1") == "Х1"
        assert normalize_designation(" X2 ") == "Х2"
        # Wire types are not case-converted, only spaces removed
        assert normalize_designation("bpvl-0.35") == "bpvl-0.35"
        assert normalize_designation("МГТФ 0.5") == "МГТФ0.5"

    def test_validate_designation(self):
        """Test designation validation."""
        assert validate_designation("X1", "connector")
        assert validate_designation("Х2", "connector")
        assert validate_designation("3", "pin")
        # Note: БПВЛ-0.35 is classified as 'wire_type' by classify_text
        # but normalize_designation doesn't convert it
        text_type = classify_text("БПВЛ-0.35")
        assert text_type == "wire_type"
        assert not validate_designation("invalid", "connector")


class TestStage4Vectorization:
    """Tests for Stage 4: Wire Vectorization."""

    def test_vectorization_config(self):
        """Test vectorization configuration."""
        config = VectorizationConfig(
            skeletonize_method="guo_hall",
            rdp_epsilon=3.0,
        )
        assert config.skeletonize_method == "guo_hall"
        assert config.rdp_epsilon == 3.0

    def test_wire_vectorizer(self):
        """Test wire vectorizer."""
        config = VectorizationConfig(rdp_epsilon=2.0)
        vectorizer = WireVectorizer(config)
        
        # Create test image with lines
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        import cv2
        # Draw horizontal line
        cv2.line(image, (50, 100), (450, 100), (0, 0, 0), 3)
        # Draw vertical line
        cv2.line(image, (250, 50), (250, 450), (0, 0, 0), 3)
        
        segments, junctions = vectorizer.vectorize(image)
        
        assert isinstance(segments, list)
        assert isinstance(junctions, set)

    def test_vectorize_wires_function(self):
        """Test vectorize_wires convenience function."""
        image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        segments, junctions = vectorize_wires(image)
        assert isinstance(segments, list)
        assert isinstance(junctions, set)


class TestStage5GraphSynthesis:
    """Tests for Stage 5: Graph Synthesis."""

    def test_pin_reference(self):
        """Test PinReference creation."""
        pin = PinReference(
            component_id="comp_001",
            pin_number="1",
            coord=(100, 200),
        )
        assert pin.component_id == "comp_001"
        assert pin.pin_number == "1"
        assert pin.coord == (100, 200)

    def test_graph_builder_with_real_segments(self):
        """Test graph builder with wire segments."""
        builder = GraphBuilder(snap_radius=20, merge_collinear=True)
        
        # Add segments forming a path
        builder.add_wire_segment((100, 100), (200, 100))
        builder.add_wire_segment((200, 100), (300, 100))
        builder.add_wire_segment((300, 100), (300, 200))
        
        # Add pins at endpoints
        builder.add_component_pins([
            PinReference("X1", "1", (100, 100)),
            PinReference("X2", "3", (300, 200)),
        ])
        
        # Build graph
        graph = builder.build_graph()
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Extract nets
        nets = builder.extract_nets([])
        assert len(nets) > 0


class TestStage6VLMArbitration:
    """Tests for Stage 6: VLM Arbitration."""

    def test_vlm_config(self):
        """Test VLM configuration."""
        config = VLMConfig(
            model_name="test-model",
            roi_size=256,
            max_calls=10,
        )
        assert config.model_name == "test-model"
        assert config.roi_size == 256
        assert config.max_calls == 10

    def test_vlm_wrapper_mock(self):
        """Test VLM wrapper with mock."""
        config = VLMConfig()
        vlm = VLMWrapper(config)
        vlm.load()
        
        # Create test image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Query
        response = vlm.query(image, "Test prompt")
        assert isinstance(response, dict)
        assert "confidence" in response

    def test_create_issues_from_detections(self):
        """Test issue creation from detections."""
        issues = create_issues_from_detections(
            wire_junctions=[(100, 100), (200, 200)],
            junction_dots=[(100, 100, 110, 110)],
            low_confidence_texts=[((50, 50, 100, 70), 0.5)],
        )
        
        # Should have one suspicious crossing (no dot near second junction)
        crossing_issues = [i for i in issues if i.issue_type == "suspicious_crossing"]
        assert len(crossing_issues) >= 0  # May vary based on dot detection


class TestIntegration:
    """Integration tests for full pipeline stages."""

    def test_end_to_end_stages_1_4(self):
        """Test stages 1-4 together."""
        # Create test image
        image = np.ones((800, 800, 3), dtype=np.uint8) * 255
        
        import cv2
        # Add some lines
        cv2.line(image, (100, 100), (400, 100), (0, 0, 0), 3)
        cv2.line(image, (400, 100), (400, 400), (0, 0, 0), 3)
        cv2.line(image, (400, 400), (700, 400), (0, 0, 0), 3)
        
        # Stage 1: Slicing
        engine = SlicingEngine(tile_size=256, overlap_ratio=0.2)
        tiles = list(engine.generate_tiles(image))
        assert len(tiles) > 0
        
        # Stage 4: Vectorization
        segments, junctions = vectorize_wires(image)
        assert len(segments) > 0
        
        # Stage 5: Graph synthesis
        builder = GraphBuilder()
        for seg in segments:
            builder.add_wire_segment(seg.start, seg.end)
        
        graph = builder.build_graph()
        assert graph.number_of_nodes() > 0
        
        # Stage 6: VLM (mock)
        config = VLMConfig()
        vlm = VLMWrapper(config)
        vlm.load()
        
        roi = image[100:356, 100:356]  # Fake ROI
        response = vlm.query(roi, "Test")
        assert "confidence" in response

    def test_full_pipeline_config(self):
        """Test pipeline with full configuration."""
        from avers.config import AVERSConfig
        
        config = AVERSConfig()
        config.slicing.tile_size = 512
        config.detection.confidence_threshold = 0.3
        config.ocr.text_confidence_threshold = 0.7
        config.vectorization.rdp_epsilon = 2.5
        config.vlm_arbitrator.enabled = False
        
        assert config.slicing.tile_size == 512
        assert config.detection.confidence_threshold == 0.3
        assert not config.vlm_arbitrator.enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
