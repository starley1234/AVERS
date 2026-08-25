"""Tests for PDF support."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from avers.utils.pdf_loader import is_pdf, PDFLoader
from avers.utils.image_helpers import load_image_auto, get_image_info, load_single_image


class TestPDFDetection:
    """Test PDF detection."""
    
    def test_is_pdf_false(self):
        assert is_pdf(Path("README.md")) is False
        assert is_pdf(Path("test.png")) is False
    
    def test_is_pdf_true(self):
        # Create dummy PDF header file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy")
            tmp_path = Path(f.name)
        
        try:
            assert is_pdf(tmp_path) is True
        finally:
            tmp_path.unlink(missing_ok=True)


class TestPDFLoader:
    """Test PDF loader."""
    
    def test_loader_init(self):
        loader = PDFLoader()
        # Should detect some backend or None
        assert loader._backend in (None, "pymupdf", "pdf2image", "pil", "auto")
    
    def test_get_info_non_pdf(self):
        loader = PDFLoader()
        info = loader.get_info(Path("README.md"))
        assert info["is_pdf"] is False
    
    def test_load_pdf_with_pymupdf(self):
        # Create test PDF if reportlab available
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
        except ImportError:
            pytest.skip("reportlab not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdf_path = tmpdir / "test.pdf"
            
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            c.drawString(100, 750, "Test Page 1")
            c.rect(100, 600, 80, 100)
            c.showPage()
            c.drawString(100, 750, "Test Page 2")
            c.showPage()
            c.save()
            
            loader = PDFLoader()
            # Force pymupdf if available
            try:
                import fitz
                loader._backend = "pymupdf"
                pages = loader.load_pages(pdf_path, dpi=72)
                assert len(pages) == 2
                assert pages[0].shape[2] == 3  # RGB
                
                info = loader.get_info(pdf_path)
                assert info["num_pages"] == 2
            except ImportError:
                pytest.skip("PyMuPDF not installed")


class TestImageHelpers:
    """Test unified image helpers."""
    
    def test_get_image_info_image(self):
        # Create dummy image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        
        try:
            from PIL import Image
            img = Image.new("RGB", (100, 50), color="white")
            img.save(tmp_path)
            
            info = get_image_info(tmp_path)
            assert info["width"] == 100
            assert info["height"] == 50
            assert info["is_pdf"] is False
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_load_image_auto_single(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        
        try:
            from PIL import Image
            img = Image.new("RGB", (64, 64), color="white")
            img.save(tmp_path)
            
            pages = load_image_auto(tmp_path)
            assert len(pages) == 1
            assert pages[0].shape == (64, 64, 3)
            
            single = load_single_image(tmp_path)
            assert single.shape == (64, 64, 3)
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_load_image_auto_pdf(self):
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            import fitz
        except ImportError:
            pytest.skip("reportlab or PyMuPDF not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdf_path = tmpdir / "test.pdf"
            
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            c.drawString(100, 750, "Test")
            c.showPage()
            c.save()
            
            pages = load_image_auto(pdf_path, dpi=72)
            assert len(pages) == 1
            assert len(pages[0].shape) == 3


class TestPublicDatasets:
    """Test public datasets registry."""
    
    def test_list_datasets(self):
        from avers.dataset.public_datasets import PUBLIC_DATASETS, PublicDatasetLoader
        
        assert len(PUBLIC_DATASETS) >= 8
        
        loader = PublicDatasetLoader()
        all_ds = loader.list_datasets()
        assert len(all_ds) >= 8
        
        gost_ds = loader.list_datasets(gost_compatible_only=True)
        assert len(gost_ds) >= 2  # masala-chai, electronet
        assert all(ds.gost_compatible for ds in gost_ds)
    
    def test_dataset_info(self):
        from avers.dataset.public_datasets import PublicDatasetLoader
        
        loader = PublicDatasetLoader()
        info = loader.get_dataset_info("masala-chai")
        assert info is not None
        assert info.name == "Masala-CHAI"
        assert info.num_images == 4300
        assert info.gost_compatible is True
    
    def test_download_instructions(self):
        from avers.dataset.public_datasets import PublicDatasetLoader
        
        loader = PublicDatasetLoader()
        instr = loader.download_instructions("juhccr-v1")
        assert "JUHCCR-v1" in instr
        assert "github.com" in instr
    
    def test_recommended_strategy(self):
        from avers.dataset.public_datasets import get_recommended_training_strategy
        
        strategy = get_recommended_training_strategy()
        assert "Masala-CHAI" in strategy
        assert "Pre-training" in strategy
        assert "Fine-tuning" in strategy
