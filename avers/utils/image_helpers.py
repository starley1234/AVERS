"""
Image helpers - unified loading for images and PDFs.

Supports:
  - Images: PNG, JPG, TIF, TIFF, BMP, etc.
  - PDFs: multi-page, each page as separate image
  - Auto-detection
"""

from pathlib import Path
from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np
from PIL import Image

from avers.utils.pdf_loader import is_pdf, PDFLoader, load_pdf_pages
from avers.core.logger import get_logger

logger = get_logger("avers.image_helpers")


def get_image_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get image/PDF info without full loading."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if is_pdf(path):
        loader = PDFLoader()
        return loader.get_info(path)
    else:
        try:
            pil_img = Image.open(path)
            return {
                "path": str(path),
                "is_pdf": False,
                "width": pil_img.width,
                "height": pil_img.height,
                "mode": pil_img.mode,
                "format": pil_img.format,
                "num_pages": getattr(pil_img, "n_frames", 1),
            }
        except Exception as e:
            return {"path": str(path), "error": str(e), "is_pdf": False}


def load_image_auto(
    path: Union[str, Path],
    dpi: int = 300,
    max_pages: Optional[int] = None,
    page_numbers: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Unified loader - images and PDFs.
    
    Args:
        path: path to image or PDF
        dpi: DPI for PDF rendering
        max_pages: max pages for PDF
        page_numbers: specific pages for PDF
    
    Returns:
        List of RGB images (even for single image, returns list with 1 element)
        For PDF: each page is separate image
    """
    path = Path(path)
    
    if is_pdf(path):
        logger.info(f"Loading PDF: {path}")
        pages = load_pdf_pages(path, dpi=dpi, max_pages=max_pages, page_numbers=page_numbers)
        logger.info(f"Loaded {len(pages)} pages from PDF")
        return pages
    else:
        # Regular image
        pil_img = Image.open(path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        
        # Handle multi-frame TIF
        n_frames = getattr(pil_img, "n_frames", 1)
        if n_frames > 1:
            pages = []
            for i in range(n_frames):
                if max_pages and i >= max_pages:
                    break
                if page_numbers and i not in page_numbers:
                    continue
                try:
                    pil_img.seek(i)
                    frame = pil_img.copy()
                    if frame.mode != "RGB":
                        frame = frame.convert("RGB")
                    pages.append(np.array(frame))
                except EOFError:
                    break
            logger.info(f"Loaded {len(pages)} frames from multi-page image {path}")
            return pages
        else:
            return [img]


def load_single_image(
    path: Union[str, Path],
    dpi: int = 300,
    page_number: int = 0,
) -> np.ndarray:
    """Load single image (first page for PDF/TIF)."""
    pages = load_image_auto(path, dpi=dpi, page_numbers=[page_number])
    if not pages:
        raise ValueError(f"Failed to load image from {path}")
    return pages[0]


def save_image_list(
    images: List[np.ndarray],
    output_dir: Path,
    base_name: str = "page",
    format: str = "jpg",
):
    """Save list of images to directory."""
    import cv2
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, img in enumerate(images):
        # Convert RGB to BGR for cv2
        if len(img.shape) == 3 and img.shape[2] == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img
        
        out_path = output_dir / f"{base_name}_{i:04d}.{format}"
        cv2.imwrite(str(out_path), bgr)
        paths.append(out_path)
    
    return paths
