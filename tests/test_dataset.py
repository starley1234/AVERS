"""Tests for Dataset module - Synthetic GOST generation."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from avers.dataset.gost_symbols import GOST_SYMBOLS, draw_symbol, get_symbol_by_name
from avers.dataset.synthetic import SyntheticGenerator, GOSTGenerator, SyntheticConfig, Annotation
from avers.dataset.generator import SchematicComposer
from avers.dataset.export import YOLOExporter, COCOExporter
from avers.dataset.annotator import AnnotationStore


class TestGOSTSymbols:
    """Test GOST symbols definitions."""
    
    def test_symbols_exist(self):
        assert len(GOST_SYMBOLS) >= 10
        assert 0 in GOST_SYMBOLS
        assert GOST_SYMBOLS[0].class_name == "connector_body"
    
    def test_get_symbol_by_name(self):
        sym = get_symbol_by_name("junction_dot")
        assert sym is not None
        assert sym.class_name == "junction_dot"
        
        sym_none = get_symbol_by_name("nonexistent")
        assert sym_none is None
    
    def test_draw_symbol(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        sym = GOST_SYMBOLS[2]  # junction_dot
        bbox = (40, 40, 60, 60)
        result = draw_symbol(img, sym, bbox)
        assert result.shape == img.shape
        # Should have drawn something (not all white)
        assert not np.all(result == 255)


class TestAnnotation:
    """Test Annotation class."""
    
    def test_yolo_conversion(self):
        ann = Annotation(class_id=0, class_name="connector_body", bbox=(100, 100, 200, 200))
        yolo_str = ann.to_yolo(1000, 1000)
        # Format: class_id cx cy w h
        parts = yolo_str.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        # Center at 150,150 -> 0.15,0.15
        assert abs(float(parts[1]) - 0.15) < 0.01
        assert abs(float(parts[2]) - 0.15) < 0.01
    
    def test_coco_conversion(self):
        ann = Annotation(class_id=1, class_name="pin", bbox=(10, 20, 30, 40))
        coco = ann.to_coco(ann_id=1, image_id=0)
        assert coco["id"] == 1
        assert coco["image_id"] == 0
        assert coco["category_id"] == 1
        assert coco["bbox"] == [10, 20, 20, 20]
        assert coco["area"] == 400


class TestSyntheticGenerator:
    """Test synthetic generation."""
    
    def test_generator_init(self):
        gen = SyntheticGenerator()
        assert gen.config is not None
        assert len(gen.class_name_to_id) > 0
    
    def test_generate_image(self):
        config = SyntheticConfig(image_size=256, min_objects=3, max_objects=5, enable_wires=False, enable_noise=False)
        gen = SyntheticGenerator(config)
        img, anns = gen.generate_image()
        
        assert img.shape == (256, 256, 3)
        assert len(anns) >= 3
        assert all(isinstance(a, Annotation) for a in anns)
    
    def test_generate_with_wires(self):
        config = SyntheticConfig(image_size=256, min_objects=5, max_objects=8, enable_wires=True, enable_noise=False)
        gen = SyntheticGenerator(config)
        img, anns = gen.generate_image()
        
        assert img.shape[0] == 256
        assert len(anns) >= 5
    
    def test_gost_generator_realistic(self):
        config = SyntheticConfig(image_size=512, enable_wires=True, enable_noise=False)
        gen = GOSTGenerator(config)
        img, anns = gen.generate_realistic_schematic()
        
        assert img.shape == (512, 512, 3)
        assert len(anns) > 0
        # Should have connector_body
        has_connector = any(a.class_name == "connector_body" for a in anns)
        assert has_connector
    
    def test_generate_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config = SyntheticConfig(image_size=256, min_objects=2, max_objects=3, enable_wires=False, enable_noise=False)
            gen = SyntheticGenerator(config)
            # Use larger size to avoid empty range
            config.image_size = 512
            gen = SyntheticGenerator(config)
            gen.generate_dataset(num_images=2, output_dir=tmpdir, split="train")
            
            # Check files created
            assert (tmpdir / "images" / "train").exists()
            assert (tmpdir / "labels" / "train").exists()
            assert (tmpdir / "dataset.yaml").exists()
            
            images = list((tmpdir / "images" / "train").glob("*.jpg"))
            labels = list((tmpdir / "labels" / "train").glob("*.txt"))
            assert len(images) == 2
            assert len(labels) == 2


class TestSchematicComposer:
    """Test large schematic composer."""
    
    def test_compose_a2x6_small(self):
        composer = SchematicComposer()
        img, anns = composer.compose_a2x6(width=1000, height=500)
        
        assert img.shape == (500, 1000, 3)
        assert len(anns) > 5


class TestExporters:
    """Test YOLO/COCO exporters."""
    
    def test_yolo_exporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            anns = [
                Annotation(0, "connector_body", (100, 100, 200, 200)),
                Annotation(2, "junction_dot", (300, 300, 310, 310)),
            ]
            label_path = tmpdir / "test.txt"
            YOLOExporter.export_annotations(anns, 1000, 1000, label_path)
            
            assert label_path.exists()
            content = label_path.read_text()
            assert "0" in content
            assert "2" in content
            
            YOLOExporter.create_dataset_yaml(tmpdir)
            assert (tmpdir / "dataset.yaml").exists()
    
    def test_coco_exporter(self):
        exporter = COCOExporter()
        anns = [Annotation(0, "connector_body", (10, 10, 50, 50))]
        exporter.add_image("test.jpg", 100, 100, anns)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        
        try:
            exporter.save(tmp_path)
            assert tmp_path.exists()
            import json
            data = json.loads(tmp_path.read_text())
            assert "images" in data
            assert "annotations" in data
            assert "categories" in data
            assert len(data["images"]) == 1
            assert len(data["annotations"]) == 1
        finally:
            tmp_path.unlink(missing_ok=True)


class TestAnnotationStore:
    """Test manual annotation store."""
    
    def test_store_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AnnotationStore(Path(tmpdir))
            assert store.root_dir == Path(tmpdir)
            assert store.images_dir.exists()
    
    def test_add_and_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AnnotationStore(Path(tmpdir))
            
            # Create dummy image file
            dummy_img = Path(tmpdir) / "dummy.jpg"
            dummy_img.write_bytes(b"fake")
            
            img = store.add_image(dummy_img, 100, 100)
            assert img.id in store.images
            
            # Update annotations
            anns = [{"class_id": 0, "class_name": "connector_body", "bbox": (10, 10, 50, 50)}]
            updated = store.update_annotations(img.id, anns)
            assert updated is not None
            assert len(updated.annotations) == 1
            assert updated.annotated is True
