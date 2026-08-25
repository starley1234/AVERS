"""Tests for Production Pipeline."""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from avers.pipeline import (
    ProductionPipeline,
    PipelineResult,
    process_schematic_production,
    load_and_process,
)
from avers.core.validators import (
    SlicedDetector,
    SchematicOCR,
    WireVectorizer,
)
from avers.core.types import AVERSManifest
from avers.config import AVERSConfig


class TestSlicedDetector:
    """Tests for SAHI-based detector."""
    
    def test_detector_init(self):
        """Test detector initialization."""
        detector = SlicedDetector(
            model_type="yolov11",
            confidence_threshold=0.3,
        )
        assert detector.model_type == "yolov11"
        assert detector.confidence_threshold == 0.3
    
    def test_detector_load_fallback(self):
        """Test detector loads in fallback mode."""
        detector = SlicedDetector(device="cpu")
        assert detector.load() is True
        assert detector._loaded is True
    
    def test_detector_detect(self):
        """Test detection on image."""
        detector = SlicedDetector()
        detector.load()
        
        # Create test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(image)
        assert isinstance(detections, list)


class TestSchematicOCR:
    """Tests for OCR."""
    
    def test_ocr_init(self):
        """Test OCR initialization."""
        ocr = SchematicOCR(lang="ru", confidence_threshold=0.7)
        assert ocr.lang == "ru"
        assert ocr.confidence_threshold == 0.7
    
    def test_ocr_load(self):
        """Test OCR loads."""
        ocr = SchematicOCR()
        assert ocr.load() is True
        assert ocr._loaded is True
    
    def test_ocr_recognize(self):
        """Test OCR recognition."""
        ocr = SchematicOCR()
        ocr.load()
        
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        results = ocr.recognize(image)
        
        assert isinstance(results, list)


class TestWireVectorizer:
    """Tests for WireVectorizer."""
    
    def test_vectorizer_init(self):
        """Test vectorizer initialization."""
        vec = WireVectorizer(rdp_epsilon=3.0, min_line_length=30)
        assert vec.rdp_epsilon == 3.0
        assert vec.min_line_length == 30
    
    def test_vectorize(self):
        """Test vectorization."""
        import cv2
        
        vec = WireVectorizer()
        
        # Create image with line
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.line(image, (50, 100), (150, 100), (0, 0, 0), 3)
        
        segments, junctions = vec.vectorize(image)
        
        assert isinstance(segments, list)
        assert isinstance(junctions, set)
    
    def test_vectorize_with_exclusions(self):
        """Test vectorization with exclusion bboxes."""
        import cv2
        
        vec = WireVectorizer()
        
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.line(image, (50, 100), (150, 100), (0, 0, 0), 3)
        
        # Exclude center region
        segments, junctions = vec.vectorize(image, [(80, 80, 120, 120)])
        
        assert isinstance(segments, list)


class TestProductionPipeline:
    """Tests for Production Pipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        pipeline = ProductionPipeline()
        assert pipeline.config is not None
    
    def test_pipeline_config_validation(self):
        """Test config validation."""
        config = AVERSConfig()
        config.slicing.tile_size = 128  # Too small
        
        with pytest.raises(ValueError, match="tile_size"):
            ProductionPipeline(config)
    
    def test_pipeline_run(self):
        """Test full pipeline run."""
        pipeline = ProductionPipeline()
        
        # Create test image
        image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        
        result = pipeline.run(image, "test.tif", dpi=300)
        
        assert isinstance(result, PipelineResult)
        assert isinstance(result.manifest, AVERSManifest)
        assert result.manifest.schema_metadata.source_file == "test.tif"
        assert result.manifest.schema_metadata.resolution_dpi == 300
    
    def test_pipeline_error_handling(self):
        """Test error handling in pipeline."""
        pipeline = ProductionPipeline()
        
        # Empty image should not crash
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = pipeline.run(image)
        
        # Should complete with warnings, not crash
        assert result.success or len(result.warnings) > 0 or len(result.errors) > 0
        assert isinstance(result.manifest, AVERSManifest)
    
    def test_pipeline_stage_timings(self):
        """Test stage timings are recorded."""
        pipeline = ProductionPipeline()
        
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = pipeline.run(image)
        
        assert "detection" in result.stage_timings
        assert "ocr" in result.stage_timings
        assert "vectorization" in result.stage_timings
        assert "graph_synthesis" in result.stage_timings
    
    def test_pipeline_validation(self):
        """Test manifest validation."""
        pipeline = ProductionPipeline()
        
        image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        result = pipeline.run(image)
        
        # Check metadata
        assert result.manifest.schema_metadata.width == 400
        assert result.manifest.schema_metadata.height == 400
    
    def test_component_grouping(self):
        """Test component grouping logic."""
        pipeline = ProductionPipeline()
        
        # Create image with simple shapes
        import cv2
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        # Draw connector-like shape
        cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 0), 2)
        
        result = pipeline.run(image)
        
        # Should have at least one component or complete without errors
        assert isinstance(result.manifest, AVERSManifest)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_process_schematic_production(self):
        """Test process_schematic_production function."""
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        result = process_schematic_production(image, "test.tif")
        
        assert isinstance(result, PipelineResult)
        assert result.manifest.schema_metadata.source_file == "test.tif"
    
    def test_load_and_process(self):
        """Test load_and_process with temp file."""
        # Create temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save test image
            Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)).save(temp_path)
            
            # Process
            result = load_and_process(temp_path)
            
            assert isinstance(result, PipelineResult)
            assert result.manifest.schema_metadata.source_file.endswith(".png")
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_load_and_process_with_output(self):
        """Test load_and_process saves output."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as input_f:
            input_path = Path(input_f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_f:
            output_path = Path(output_f.name)
        
        try:
            Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)).save(input_path)
            
            result = load_and_process(input_path, output_path)
            
            assert output_path.exists()
            
            # Verify saved content
            content = output_path.read_text()
            assert "schema_metadata" in content
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_image(self):
        """Test pipeline on empty image."""
        pipeline = ProductionPipeline()
        
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = pipeline.run(image)
        
        # Should complete without crash
        assert isinstance(result, PipelineResult)
    
    def test_single_pixel_image(self):
        """Test pipeline on single pixel image."""
        pipeline = ProductionPipeline()
        
        image = np.ones((1, 1, 3), dtype=np.uint8) * 255
        result = pipeline.run(image)
        
        assert isinstance(result, PipelineResult)
    
    def test_very_large_image(self):
        """Test pipeline on large image (but not too large for memory)."""
        pipeline = ProductionPipeline()
        
        # 2000x2000 should be fine
        image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        result = pipeline.run(image)
        
        assert isinstance(result, PipelineResult)
        assert result.manifest.schema_metadata.width == 2000
        assert result.manifest.schema_metadata.height == 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
