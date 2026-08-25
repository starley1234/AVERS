"""
PDF Loader for AVERS - поддержка загрузки схем в виде PDF.

Поддерживает:
  - Одностраничные и многостраничные PDF
  - Конвертация страниц в изображения (RGB)
  - Fallback: PyMuPDF -> pdf2image -> PIL
  - Сохранение метаданных (DPI, размеры)

Использование:
  loader = PDFLoader()
  pages = loader.load_pages("schema.pdf", dpi=300)
  # pages: List[np.ndarray] - RGB images

  # Или быстрый метод
  pages = load_pdf_pages("schema.pdf", dpi=300)
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import io

import numpy as np

from avers.core.logger import get_logger

logger = get_logger("avers.pdf")


def is_pdf(path: Union[str, Path]) -> bool:
    """Проверить является ли файл PDF."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return True
    # Check magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


class PDFLoader:
    """Загрузчик PDF с fallback на разные бэкенды."""
    
    def __init__(self, preferred_backend: str = "auto"):
        """
        Args:
            preferred_backend: 'auto', 'pymupdf', 'pdf2image', 'pil'
        """
        self.preferred_backend = preferred_backend
        self._backend = None
        self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Определить доступный бэкенд."""
        if self.preferred_backend != "auto":
            self._backend = self.preferred_backend
            return self._backend
        
        # Try PyMuPDF (best for schematics - preserves vector quality)
        try:
            import fitz  # PyMuPDF
            self._backend = "pymupdf"
            logger.info("Using PyMuPDF backend for PDF")
            return self._backend
        except ImportError:
            pass
        
        # Try pdf2image (poppler)
        try:
            from pdf2image import convert_from_path
            self._backend = "pdf2image"
            logger.info("Using pdf2image backend for PDF")
            return self._backend
        except ImportError:
            pass
        
        # Fallback to PIL (limited)
        try:
            from PIL import Image
            self._backend = "pil"
            logger.info("Using PIL backend for PDF (limited)")
            return self._backend
        except ImportError:
            self._backend = None
            logger.warning("No PDF backend available")
            return None
    
    def load_pages(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Загрузить страницы PDF как изображения.
        
        Args:
            pdf_path: путь к PDF
            dpi: DPI для рендеринга
            max_pages: макс количество страниц (None = все)
            page_numbers: конкретные страницы (0-indexed)
        
        Returns:
            List of RGB images (H, W, 3) np.ndarray
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if self._backend is None:
            self._detect_backend()
        
        if self._backend == "pymupdf":
            return self._load_pymupdf(pdf_path, dpi, max_pages, page_numbers)
        elif self._backend == "pdf2image":
            return self._load_pdf2image(pdf_path, dpi, max_pages, page_numbers)
        elif self._backend == "pil":
            return self._load_pil(pdf_path, dpi, max_pages, page_numbers)
        else:
            raise RuntimeError("No PDF backend available. Install PyMuPDF: pip install PyMuPDF")
    
    def _load_pymupdf(
        self,
        pdf_path: Path,
        dpi: int,
        max_pages: Optional[int],
        page_numbers: Optional[List[int]],
    ) -> List[np.ndarray]:
        """Загрузка через PyMuPDF (лучшее качество)."""
        import fitz
        
        doc = fitz.open(str(pdf_path))
        pages = []
        
        # Determine pages to load
        if page_numbers:
            indices = page_numbers
        else:
            indices = list(range(len(doc)))
            if max_pages:
                indices = indices[:max_pages]
        
        zoom = dpi / 72.0  # PDF default 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        for idx in indices:
            if idx >= len(doc):
                break
            
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            if pix.n == 3:
                img = img_data.reshape(pix.h, pix.w, 3)
            elif pix.n == 4:
                # RGBA to RGB
                img_rgba = img_data.reshape(pix.h, pix.w, 4)
                img = img_rgba[:, :, :3]
            elif pix.n == 1:
                # Grayscale to RGB
                gray = img_data.reshape(pix.h, pix.w)
                img = np.stack([gray, gray, gray], axis=2)
            else:
                img = img_data.reshape(pix.h, pix.w, pix.n)[:, :, :3]
            
            pages.append(img)
            logger.debug(f"Loaded PDF page {idx}: {img.shape[1]}x{img.shape[0]}")
        
        doc.close()
        logger.info(f"Loaded {len(pages)} pages from {pdf_path} via PyMuPDF")
        return pages
    
    def _load_pdf2image(
        self,
        pdf_path: Path,
        dpi: int,
        max_pages: Optional[int],
        page_numbers: Optional[List[int]],
    ) -> List[np.ndarray]:
        """Загрузка через pdf2image (poppler)."""
        from pdf2image import convert_from_path
        
        kwargs = {"dpi": dpi}
        if page_numbers:
            # pdf2image uses 1-indexed pages
            kwargs["first_page"] = min(page_numbers) + 1
            kwargs["last_page"] = max(page_numbers) + 1
        
        pil_images = convert_from_path(str(pdf_path), **kwargs)
        
        if max_pages:
            pil_images = pil_images[:max_pages]
        
        # Filter by page_numbers if needed
        if page_numbers and len(page_numbers) != len(pil_images):
            # Need to select specific pages
            # convert_from_path with first/last gives range, so filter
            # For simplicity, if page_numbers is not contiguous, load all and filter
            if len(pil_images) > 0 and max(page_numbers) < len(pil_images):
                pil_images = [pil_images[i] for i in page_numbers if i < len(pil_images)]
        
        pages = []
        for pil_img in pil_images:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img = np.array(pil_img)
            pages.append(img)
        
        logger.info(f"Loaded {len(pages)} pages from {pdf_path} via pdf2image")
        return pages
    
    def _load_pil(
        self,
        pdf_path: Path,
        dpi: int,
        max_pages: Optional[int],
        page_numbers: Optional[List[int]],
    ) -> List[np.ndarray]:
        """Загрузка через PIL (ограниченная, только первый фрейм обычно)."""
        from PIL import Image
        
        pages = []
        try:
            img = Image.open(pdf_path)
            # PIL PDF support is limited
            # Try to get n_frames
            n_frames = getattr(img, "n_frames", 1)
            if max_pages:
                n_frames = min(n_frames, max_pages)
            
            indices = page_numbers if page_numbers else list(range(n_frames))
            
            for idx in indices:
                try:
                    img.seek(idx)
                    frame = img.copy()
                    if frame.mode != "RGB":
                        frame = frame.convert("RGB")
                    # Resize based on DPI (approximate)
                    # PDF default 72 DPI, so scale
                    scale = dpi / 72.0
                    if scale != 1.0:
                        new_size = (int(frame.width * scale), int(frame.height * scale))
                        frame = frame.resize(new_size, Image.LANCZOS)
                    
                    pages.append(np.array(frame))
                except EOFError:
                    break
        except Exception as e:
            logger.error(f"PIL PDF load failed: {e}")
            raise
        
        logger.info(f"Loaded {len(pages)} pages from {pdf_path} via PIL")
        return pages
    
    def get_info(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """Получить информацию о PDF."""
        pdf_path = Path(pdf_path)
        
        info = {
            "path": str(pdf_path),
            "is_pdf": is_pdf(pdf_path),
            "backend": self._backend,
        }
        
        if self._backend == "pymupdf":
            try:
                import fitz
                doc = fitz.open(str(pdf_path))
                info["num_pages"] = len(doc)
                if len(doc) > 0:
                    page = doc.load_page(0)
                    info["first_page_size"] = (page.rect.width, page.rect.height)
                doc.close()
            except Exception as e:
                info["error"] = str(e)
        else:
            info["num_pages"] = "unknown (install PyMuPDF for info)"
        
        return info


def load_pdf_pages(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    max_pages: Optional[int] = None,
    page_numbers: Optional[List[int]] = None,
    backend: str = "auto",
) -> List[np.ndarray]:
    """
    Быстрый метод загрузки PDF страниц.
    
    Args:
        pdf_path: путь к PDF
        dpi: DPI рендеринга
        max_pages: макс страниц
        page_numbers: конкретные страницы
        backend: бэкенд
    
    Returns:
        List of RGB images
    """
    loader = PDFLoader(preferred_backend=backend)
    return loader.load_pages(pdf_path, dpi, max_pages, page_numbers)


def pdf_page_to_image(
    pdf_path: Union[str, Path],
    page_number: int = 0,
    dpi: int = 300,
) -> np.ndarray:
    """Загрузить одну страницу PDF."""
    pages = load_pdf_pages(pdf_path, dpi=dpi, page_numbers=[page_number])
    if not pages:
        raise ValueError(f"Failed to load page {page_number} from {pdf_path}")
    return pages[0]
