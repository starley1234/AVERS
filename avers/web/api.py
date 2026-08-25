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
    """Resolve human review issue with Active Learning + RAG integration."""
    if file_id not in manifests_db:
        # Try load from disk
        result_path = RESULTS_DIR / f"{file_id}.json"
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from avers.core.types import AVERSManifest
            manifest = AVERSManifest(**data)
            manifests_db[file_id] = manifest
        else:
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
    
    # Active Learning + RAG integration
    try:
        from avers.active_learning.loop import get_active_learning_loop
        from avers.rag import get_rag
        
        loop = get_active_learning_loop()
        
        # Extract ROI for feedback
        file_info = files_db.get(file_id)
        if file_info and Path(file_info["path"]).exists():
            pil_img = Image.open(file_info["path"])
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image = np.array(pil_img)
            
            x1, y1, x2, y2 = issue.bbox
            # Expand bbox for context
            h, w = image.shape[:2]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            roi_size = 256
            x1_roi = max(0, cx - roi_size//2)
            y1_roi = max(0, cy - roi_size//2)
            x2_roi = min(w, cx + roi_size//2)
            y2_roi = min(h, cy + roi_size//2)
            roi = image[y1_roi:y2_roi, x1_roi:x2_roi]
            
            if roi.size > 0:
                # Determine corrected label
                corrected_label = "junction_dot_connected" if req.connected else "junction_dot_none"
                if req.corrected_text:
                    corrected_label = f"text_{req.corrected_text}"
                elif req.resolution:
                    corrected_label = req.resolution[:50]
                
                # Add to active learning loop (auto-indexes in RAG)
                loop.add_feedback(
                    file_id=file_id,
                    bbox=issue.bbox,
                    original_label=issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type),
                    corrected_label=corrected_label,
                    issue_type=issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type),
                    original_bbox=issue.bbox,
                    corrected_bbox=req.corrected_bbox or issue.bbox,
                    image_crop=roi,
                    user_id="validator_ui",
                    comment=req.resolution or f"Resolved: connected={req.connected}, text={req.corrected_text}"
                )
                
                # Also add to RAG store directly for backward compat
                try:
                    rag = get_rag()
                    rag.add_example(
                        image=roi,
                        label=corrected_label,
                        bbox=req.corrected_bbox or issue.bbox,
                        description=req.resolution or issue.description,
                        metadata={"feedback": True, "file_id": file_id, "issue_idx": issue_idx}
                    )
                except Exception:
                    pass
                
                # Check if should retrain
                if loop.should_retrain():
                    # Don't auto-trigger heavy training, just log
                    import logging
                    logging.getLogger("avers.web").info(f"Active learning threshold reached: {len(loop.feedback_entries)} samples, consider retraining")
    
    except Exception as e:
        # Don't fail the request if active learning fails
        import logging
        logging.getLogger("avers.web").warning(f"Active learning feedback failed: {e}")
    
    return issue.model_dump()


@router.get("/active-learning/stats")
async def get_active_learning_stats():
    """Get active learning stats."""
    try:
        from avers.active_learning.loop import get_active_learning_loop
        loop = get_active_learning_loop()
        stats = loop.get_correction_stats()
        return stats
    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {e}")


@router.post("/active-learning/retrain")
async def trigger_active_learning_retrain(
    background_tasks: BackgroundTasks,
    model_type: str = Query("rtdetr-l"),
    epochs: int = Query(20),
):
    """Trigger retraining from feedback."""
    try:
        from avers.active_learning.loop import get_active_learning_loop
        loop = get_active_learning_loop()
        
        if not loop.should_retrain(threshold=1):
            return {"status": "no_data", "message": "Not enough feedback samples"}
        
        # Run in background
        def retrain_task():
            loop.trigger_retraining(model_type=model_type, epochs=epochs)
        
        background_tasks.add_task(retrain_task)
        
        return {
            "status": "started",
            "feedback_samples": len(loop.feedback_entries),
            "model_type": model_type,
            "epochs": epochs,
        }
    except Exception as e:
        raise HTTPException(500, f"Retrain failed: {e}")


@router.delete("/active-learning/clear")
async def clear_active_learning():
    """Clear feedback."""
    try:
        from avers.active_learning.loop import get_active_learning_loop
        loop = get_active_learning_loop()
        count = len(loop.feedback_entries)
        loop.clear()
        return {"status": "cleared", "cleared_count": count}
    except Exception as e:
        raise HTTPException(500, f"Clear failed: {e}")


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


# RAG endpoints - now using real VisionRAG
rag_router = APIRouter(prefix="/api/rag")

# Simple in-memory RAG store for backward compat
rag_store: List[dict] = []

def _get_vision_rag():
    """Get VisionRAG instance."""
    try:
        from avers.rag import get_rag
        return get_rag()
    except Exception:
        return None


@rag_router.post("/index")
async def rag_index(req: RAGIndexRequest):
    """Add example to RAG store (real VisionRAG + backward compat)."""
    item_id = _generate_id()
    
    # Decode image
    try:
        img_data = base64.b64decode(req.image_base64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    
    # Try real VisionRAG
    rag = _get_vision_rag()
    if rag and img is not None:
        try:
            real_id = rag.add_example(
                image=img,
                label=req.label,
                bbox=req.bbox,
                description=req.description,
                metadata=req.metadata,
            )
            # Also keep in simple store
            entry = {
                "id": real_id,
                "label": req.label,
                "bbox": req.bbox,
                "description": req.description,
                "metadata": req.metadata,
                "created_at": datetime.now().isoformat(),
            }
            rag_store.append(entry)
            return {"id": real_id, "status": "indexed", "backend": "vision_rag"}
        except Exception as e:
            # Fallback to simple
            pass
    
    # Fallback simple store
    entry = {
        "id": item_id,
        "label": req.label,
        "bbox": req.bbox,
        "description": req.description,
        "metadata": req.metadata,
        "created_at": datetime.now().isoformat(),
    }
    rag_store.append(entry)
    
    return {"id": item_id, "status": "indexed", "backend": "simple"}


@rag_router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """Query RAG store - tries real VisionRAG first."""
    start = time.time()
    
    # Try real VisionRAG
    rag = _get_vision_rag()
    if rag:
        try:
            # Decode image if provided
            query_image = None
            if req.image_base64:
                try:
                    img_data = base64.b64decode(req.image_base64.split(",")[-1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        query_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
            
            response = rag.query(
                image=query_image,
                text=req.text_query,
                top_k=req.top_k,
                use_vlm=req.use_vlm,
            )
            
            # Convert to API model
            results = []
            for r in response.results:
                results.append(RAGResult(
                    id=r.entry.id,
                    label=r.entry.label,
                    score=r.score,
                    bbox=r.entry.bbox,
                    description=r.entry.description,
                    metadata=r.entry.metadata,
                ))
            
            return RAGQueryResponse(
                query=req.text_query or "",
                results=results,
                vlm_answer=response.vlm_answer,
                processing_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            # Fallback to simple
            pass
    
    # Fallback simple text matching
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
    
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[:req.top_k]
    
    vlm_answer = None
    if req.use_vlm and results:
        vlm_answer = {
            "connected": True if "junction" in query_lower else False,
            "confidence": 0.75,
            "reasoning": f"Based on {len(results)} similar examples, this appears to be {results[0].label}",
            "examples_used": len(results),
            "backend": "simple"
        }
    
    return RAGQueryResponse(
        query=req.text_query or "",
        results=results,
        vlm_answer=vlm_answer,
        processing_time_ms=(time.time() - start) * 1000
    )


@rag_router.get("/stats")
async def rag_stats():
    """Get RAG store stats - real VisionRAG if available."""
    rag = _get_vision_rag()
    if rag:
        try:
            return rag.stats()
        except Exception:
            pass
    
    return {
        "total_items": len(rag_store),
        "labels": list(set(item["label"] for item in rag_store)),
        "backend": "simple",
    }


@rag_router.delete("/clear")
async def rag_clear():
    """Clear RAG store."""
    rag = _get_vision_rag()
    if rag:
        try:
            rag.clear()
        except Exception:
            pass
    
    rag_store.clear()
    return {"status": "cleared"}
