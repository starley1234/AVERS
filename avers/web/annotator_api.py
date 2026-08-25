"""Annotator API for manual dataset labeling."""

from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import uuid

from avers.dataset.annotator import get_store
from avers.dataset.gost_symbols import GOST_SYMBOLS

router = APIRouter(prefix="/api/annotator")

store = get_store()


@router.get("/classes")
async def get_classes():
    """Get GOST classes."""
    return {
        id: {
            "id": sym.class_id,
            "name": sym.class_name,
            "gost": sym.gost_standard,
            "description": sym.description,
        }
        for id, sym in GOST_SYMBOLS.items()
    }


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image for annotation."""
    file_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".jpg"
    save_path = store.images_dir / f"{file_id}{ext}"
    
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    
    # Get dimensions
    try:
        pil_img = Image.open(save_path)
        w, h = pil_img.size
    except:
        w, h = 1024, 1024
    
    img = store.add_image(save_path, w, h)
    return img.to_dict()


@router.get("/images")
async def list_images():
    """List images."""
    return [img.to_dict() for img in store.list_images()]


@router.get("/images/{image_id}")
async def get_image_meta(image_id: str):
    """Get image metadata."""
    img = store.get_image(image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    return img.to_dict()


@router.get("/image-file/{image_id}")
async def get_image_file(image_id: str):
    """Get image file."""
    img = store.get_image(image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    if not img.file_path.exists():
        raise HTTPException(404, "File not found on disk")
    return FileResponse(str(img.file_path))


@router.put("/images/{image_id}/annotations")
async def update_annotations(image_id: str, data: Dict):
    """Update annotations for image."""
    anns = data.get("annotations", [])
    img = store.update_annotations(image_id, anns)
    if not img:
        raise HTTPException(404, "Image not found")
    return img.to_dict()


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """Delete image."""
    if not store.delete_image(image_id):
        raise HTTPException(404, "Image not found")
    return {"status": "deleted"}


@router.post("/export")
async def export_dataset(data: Dict):
    """Export dataset to YOLO format."""
    output_dir = Path(data.get("output_dir", "/tmp/avers_dataset_export"))
    split_ratio = data.get("split_ratio", 0.8)
    
    yaml_path = store.export_yolo(output_dir, split_ratio)
    return {
        "output_dir": str(output_dir),
        "yaml_path": str(yaml_path),
        "images": len(store.images),
    }


@router.get("/stats")
async def get_stats():
    """Get annotator stats."""
    images = store.list_images()
    annotated = sum(1 for img in images if img.annotated)
    total_anns = sum(len(img.annotations) for img in images)
    
    class_counts = {}
    for img in images:
        for ann in img.annotations:
            class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1
    
    return {
        "total_images": len(images),
        "annotated_images": annotated,
        "total_annotations": total_anns,
        "class_counts": class_counts,
    }
