"""AVERS Web API - FastAPI routes."""

import uuid
import time
import json
import base64
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from io import BytesIO

import numpy as np
import cv2
from PIL import Image

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

from avers.web.models import (
    UploadResponse, JobStatusResponse, JobStatus,
    ComponentEditRequest, IssueResolutionRequest,
    ProcessRequest, RAGQueryRequest, RAGIndexRequest, RAGQueryResponse, RAGResult
)
from avers.pipeline import ProductionPipeline
from avers.config import AVERSConfig
from avers.core.types import AVERSManifest

# In-memory storage (for demo, replace with DB in production)
UPLOAD_DIR = Path("/tmp/avers_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("/tmp/avers_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global state
files_db: Dict[str, dict] = {}
jobs_db: Dict[str, dict] = {}
manifests_db: Dict[str, AVERSManifest] = {}

router = APIRouter(prefix="/api")


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload schematic image."""
    file_id = _generate_id()
    ext = Path(file.filename).suffix.lower() or ".png"
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    
    # Save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Get image dimensions
    try:
        pil_img = Image.open(save_path)
        width, height = pil_img.size
    except Exception:
        width, height = 0, 0
    
    files_db[file_id] = {
        "id": file_id,
        "filename": file.filename,
        "path": str(save_path),
        "width": width,
        "height": height,
        "size": len(content),
        "created_at": datetime.now(),
    }
    
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        width=width,
        height=height,
        size_bytes=len(content),
        preview_url=f"/api/image/{file_id}"
    )


@router.get("/files")
async def list_files():
    """List uploaded files."""
    return list(files_db.values())


@router.get("/image/{file_id}")
async def get_image(file_id: str):
    """Get original image."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    path = files_db[file_id]["path"]
    return FileResponse(path)


@router.post("/process/{file_id}")
async def start_processing(
    file_id: str,
    request: ProcessRequest = ProcessRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Start processing pipeline for file."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    job_id = _generate_id()
    
    jobs_db[job_id] = {
        "job_id": job_id,
        "file_id": file_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_stage": None,
        "stage_timings": {},
        "errors": [],
        "warnings": [],
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "config_overrides": request.config_overrides,
    }
    
    # Start background task
    background_tasks.add_task(_run_pipeline, job_id, file_id, request)
    
    return {"job_id": job_id, "file_id": file_id, "status": "pending"}


async def _run_pipeline(job_id: str, file_id: str, request: ProcessRequest):
    """Background pipeline execution."""
    job = jobs_db[job_id]
    job["status"] = JobStatus.RUNNING
    job["updated_at"] = datetime.now()
    
    try:
        # Load image
        file_info = files_db[file_id]
        image_path = Path(file_info["path"])
        
        pil_img = Image.open(image_path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        image = np.array(pil_img)
        
        # Config
        config = AVERSConfig()
        if request.config_overrides:
            # Apply overrides
            for key, value in request.config_overrides.items():
                if hasattr(config, key):
                    # Nested update
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            if hasattr(getattr(config, key), subkey):
                                setattr(getattr(config, key), subkey, subval)
                    else:
                        setattr(config, key, value)
        
        # Pipeline with progress tracking
        pipeline = ProductionPipeline(config)
        
        # Simulate stage progress
        stages = ["detection", "ocr", "vectorization", "graph_synthesis", "vlm_arbitration"]
        
        # Run actual pipeline
        job["current_stage"] = "detection"
        job["progress"] = 10
        job["updated_at"] = datetime.now()
        
        result = pipeline.run(image, file_info["filename"], dpi=300)
        
        # Update job with results
        job["stage_timings"] = result.stage_timings
        job["errors"] = result.errors
        job["warnings"] = result.warnings
        job["progress"] = 100
        job["current_stage"] = "completed"
        job["status"] = JobStatus.COMPLETED if result.success else JobStatus.COMPLETED
        job["updated_at"] = datetime.now()
        
        # Save manifest
        manifests_db[file_id] = result.manifest
        
        # Save to disk
        result_path = RESULTS_DIR / f"{file_id}.json"
        result.manifest.save(result_path, format="json")
        
    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["errors"].append(str(e))
        job["updated_at"] = datetime.now()
        import traceback
        print(f"Pipeline failed: {traceback.format_exc()}")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    return JobStatusResponse(**job)


@router.get("/result/{file_id}")
async def get_result(file_id: str):
    """Get processing result manifest."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    # Try memory first
    if file_id in manifests_db:
        return manifests_db[file_id].model_dump()
    
    # Try disk
    result_path = RESULTS_DIR / f"{file_id}.json"
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    raise HTTPException(404, "Result not found - run processing first")


@router.get("/visualization/{file_id}/{stage}")
async def get_visualization(file_id: str, stage: str):
    """Get visualization for specific stage."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    file_info = files_db[file_id]
    image_path = Path(file_info["path"])
    
    pil_img = Image.open(image_path)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    image = np.array(pil_img)
    
    vis_path = RESULTS_DIR / f"{file_id}_vis_{stage}.png"
    
    try:
        if stage == "detection":
            from avers.core.validators import SlicedDetector
            detector = SlicedDetector(device="cpu")
            detections = detector.detect(image)
            
            vis = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{det['category']} {det['confidence']:.2f}", 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "ocr":
            from avers.core.validators import SchematicOCR
            ocr = SchematicOCR()
            texts = ocr.recognize(image)
            
            vis = image.copy()
            for t in texts:
                x1, y1, x2, y2 = t["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(vis, t["text"], (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "vectorization":
            from avers.core.validators import WireVectorizer
            vec = WireVectorizer()
            segments, junctions = vec.vectorize(image)
            
            vis = image.copy()
            for seg in segments:
                cv2.line(vis, seg.start, seg.end, (100, 100, 100), 2)
            for j in junctions:
                cv2.circle(vis, j, 5, (0, 0, 255), -1)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "graph":
            # Graph visualization requires manifest
            if file_id not in manifests_db:
                raise HTTPException(404, "Run processing first")
            manifest = manifests_db[file_id]
            
            vis = image.copy()
            # Draw components
            for comp in manifest.components:
                x1, y1, x2, y2 = comp.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, comp.designator, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                for pin in comp.pins:
                    cv2.circle(vis, tuple(pin.coord), 4, (255, 0, 0), -1)
            
            # Draw nets
            for net in manifest.nets:
                pts = net.path_points
                for i in range(len(pts)-1):
                    cv2.line(vis, pts[i], pts[i+1], (0, 0, 255), 1)
            
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        else:
            raise HTTPException(400, f"Unknown stage: {stage}")
        
        return FileResponse(str(vis_path))
    
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")


@router.put("/components/{file_id}/{comp_id}")
async def update_component(file_id: str, comp_id: str, edit: ComponentEditRequest):
    """Update component (human correction)."""
    if file_id not in manifests_db:
        raise HTTPException(404, "Manifest not found")
    
    manifest = manifests_db[file_id]
    
    for comp in manifest.components:
        if comp.id == comp_id:
            if edit.designator is not None:
                comp.designator = edit.designator
            if edit.bbox is not None:
                comp.bbox = edit.bbox
            if edit.part_number is not None:
                comp.part_number = edit.part_number
            if edit.confidence is not None:
                comp.confidence = edit.confidence
            # TODO: pins update
            
            # Save
            result_path = RESULTS_DIR / f"{file_id}.json"
            manifest.save(result_path)
            
            return comp.model_dump()
    
    raise HTTPException(404, "Component not found")


@router.post("/issues/{file_id}/{issue_idx}/resolve")
async def resolve_issue(file_id: str, issue_idx: int, req: IssueResolutionRequest):
    """Resolve human review issue."""
    if file_id not in manifests_db:
        raise HTTPException(404, "Manifest not found")
    
    manifest = manifests_db[file_id]
    
    if issue_idx < 0 or issue_idx >= len(manifest.human_review_required):
        raise HTTPException(404, "Issue not found")
    
    issue = manifest.human_review_required[issue_idx]
    issue.resolved = req.resolved
    issue.resolution = req.resolution
    
    if req.corrected_text:
        issue.resolution = f"corrected_text={req.corrected_text}"
    if req.connected is not None:
        issue.resolution = f"connected={req.connected}"
    
    result_path = RESULTS_DIR / f"{file_id}.json"
    manifest.save(result_path)
    
    return issue.model_dump()


@router.get("/export/{file_id}")
async def export_manifest(
    file_id: str,
    format: str = Query("json", enum=["json", "xml"])
):
    """Export manifest."""
    if file_id not in manifests_db:
        result_path = RESULTS_DIR / f"{file_id}.json"
        if not result_path.exists():
            raise HTTPException(404, "Result not found")
        with open(result_path, "r") as f:
            data = json.load(f)
        # Reconstruct manifest
        from avers.core.types import AVERSManifest
        manifest = AVERSManifest(**data)
    else:
        manifest = manifests_db[file_id]
    
    export_path = RESULTS_DIR / f"{file_id}_export.{format}"
    manifest.save(export_path, format=format)
    
    return FileResponse(
        str(export_path),
        filename=f"{files_db[file_id]['filename']}_avers.{format}",
        media_type="application/json" if format == "json" else "application/xml"
    )


# ==================== Dataset & RAG APIs ====================

@router.get("/dataset/classes")
async def get_detection_classes():
    """Get detection classes for annotation."""
    from avers.core.types import DETECTION_CLASSES
    return DETECTION_CLASSES


# RAG endpoints will be added via separate router
rag_router = APIRouter(prefix="/api/rag")

# Simple in-memory RAG store
rag_store: List[dict] = []


@rag_router.post("/index")
async def rag_index(req: RAGIndexRequest):
    """Add example to RAG store."""
    item_id = _generate_id()
    
    # Decode image
    try:
        img_data = base64.b64decode(req.image_base64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    
    # Save crop
    x1, y1, x2, y2 = req.bbox
    crop = img[y1:y2, x1:x2] if img is not None else None
    
    # Simple embedding: use histogram or CLIP later
    # For now, store metadata
    
    entry = {
        "id": item_id,
        "label": req.label,
        "bbox": req.bbox,
        "description": req.description,
        "metadata": req.metadata,
        "created_at": datetime.now().isoformat(),
    }
    
    rag_store.append(entry)
    
    return {"id": item_id, "status": "indexed"}


@rag_router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """Query RAG store."""
    start = time.time()
    
    # Simple text matching for now
    # In production: use CLIP embeddings + vector DB
    
    results = []
    query_lower = (req.text_query or "").lower()
    
    for item in rag_store:
        score = 0.0
        if query_lower:
            if query_lower in item["label"].lower():
                score = 0.9
            elif query_lower in (item["description"] or "").lower():
                score = 0.7
            else:
                score = 0.1
        else:
            score = 0.5
        
        if score > 0.2:
            results.append(RAGResult(
                id=item["id"],
                label=item["label"],
                score=score,
                bbox=item["bbox"],
                description=item["description"],
                metadata=item["metadata"],
            ))
    
    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[:req.top_k]
    
    # VLM answer mock
    vlm_answer = None
    if req.use_vlm and results:
        vlm_answer = {
            "connected": True if "junction" in query_lower else False,
            "confidence": 0.75,
            "reasoning": f"Based on {len(results)} similar examples, this appears to be {results[0].label}",
            "examples_used": len(results)
        }
    
    return RAGQueryResponse(
        query=req.text_query or "",
        results=results,
        vlm_answer=vlm_answer,
        processing_time_ms=(time.time() - start) * 1000
    )


@rag_router.get("/stats")
async def rag_stats():
    """Get RAG store stats."""
    return {
        "total_items": len(rag_store),
        "labels": list(set(item["label"] for item in rag_store)),
    }


@rag_router.delete("/clear")
async def rag_clear():
    """Clear RAG store."""
    rag_store.clear()
    return {"status": "cleared"}
