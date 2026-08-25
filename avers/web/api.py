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

PDF_CACHE_DIR = Path("/tmp/avers_pdf_cache")
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global state
files_db: Dict[str, dict] = {}
jobs_db: Dict[str, dict] = {}
manifests_db: Dict[str, AVERSManifest] = {}
pdf_pages_db: Dict[str, List[Path]] = {}  # file_id -> list of page image paths

router = APIRouter(prefix="/api")


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _is_pdf_file(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload schematic image or PDF."""
    file_id = _generate_id()
    ext = Path(file.filename).suffix.lower() or ".png"
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    
    # Save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Handle PDF - extract pages
    if _is_pdf_file(file.filename):
        try:
            from avers.utils.pdf_loader import PDFLoader, is_pdf
            from avers.utils.image_helpers import load_image_auto
            
            loader = PDFLoader()
            info = loader.get_info(save_path)
            num_pages = info.get("num_pages", 1)
            if isinstance(num_pages, str):
                num_pages = 1
            
            # Load pages as images
            pages = load_image_auto(save_path, dpi=300, max_pages=20)  # Limit to 20 pages for safety
            
            # Save each page as separate image for preview
            page_paths = []
            for i, page_img in enumerate(pages):
                page_path = PDF_CACHE_DIR / f"{file_id}_page_{i:04d}.jpg"
                # Convert RGB to BGR for cv2
                import cv2
                bgr = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(page_path), bgr)
                page_paths.append(page_path)
            
            pdf_pages_db[file_id] = page_paths
            
            # Use first page for dimensions
            if pages:
                height, width = pages[0].shape[:2]
            else:
                width, height = 0, 0
            
            files_db[file_id] = {
                "id": file_id,
                "filename": file.filename,
                "path": str(save_path),
                "width": width,
                "height": height,
                "size": len(content),
                "created_at": datetime.now(),
                "is_pdf": True,
                "num_pages": len(pages),
                "page_paths": [str(p) for p in page_paths],
            }
            
            return UploadResponse(
                file_id=file_id,
                filename=file.filename,
                width=width,
                height=height,
                size_bytes=len(content),
                preview_url=f"/api/image/{file_id}?page=0"
            )
        
        except Exception as e:
            # Fallback - treat as regular file
            import logging
            logging.getLogger("avers.web").warning(f"PDF processing failed: {e}, treating as single file")
    
    # Regular image handling
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
        "is_pdf": False,
        "num_pages": 1,
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
async def get_image(file_id: str, page: int = Query(0, ge=0)):
    """Get original image or PDF page."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    file_info = files_db[file_id]
    
    # PDF page handling
    if file_info.get("is_pdf") and file_id in pdf_pages_db:
        pages = pdf_pages_db[file_id]
        if page < len(pages):
            return FileResponse(str(pages[page]))
        else:
            raise HTTPException(404, f"Page {page} not found, PDF has {len(pages)} pages")
    
    path = file_info["path"]
    return FileResponse(path)


@router.get("/pdf/{file_id}/info")
async def get_pdf_info(file_id: str):
    """Get PDF info and pages."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    file_info = files_db[file_id]
    if not file_info.get("is_pdf"):
        raise HTTPException(400, "File is not PDF")
    
    pages = pdf_pages_db.get(file_id, [])
    
    return {
        "file_id": file_id,
        "filename": file_info["filename"],
        "num_pages": file_info.get("num_pages", len(pages)),
        "pages": [
            {
                "page_number": i,
                "preview_url": f"/api/image/{file_id}?page={i}",
                "path": str(p),
            }
            for i, p in enumerate(pages)
        ]
    }


@router.get("/pdf/{file_id}/page/{page_number}")
async def get_pdf_page(file_id: str, page_number: int):
    """Get specific PDF page as image."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    if file_id not in pdf_pages_db:
        raise HTTPException(404, "PDF pages not found, re-upload")
    
    pages = pdf_pages_db[file_id]
    if page_number < 0 or page_number >= len(pages):
        raise HTTPException(404, f"Page {page_number} not found")
    
    return FileResponse(str(pages[page_number]))


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
    """Background pipeline execution - supports PDF multi-page."""
    job = jobs_db[job_id]
    job["status"] = JobStatus.RUNNING
    job["updated_at"] = datetime.now()
    
    try:
        file_info = files_db[file_id]
        image_path = Path(file_info["path"])
        
        # Load images - handle PDF multi-page via unified loader
        from avers.utils.image_helpers import load_image_auto
        
        # Config
        config = AVERSConfig()
        if request.config_overrides:
            for key, value in request.config_overrides.items():
                if hasattr(config, key):
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            if hasattr(getattr(config, key), subkey):
                                setattr(getattr(config, key), subkey, subval)
                    else:
                        setattr(config, key, value)
        
        pipeline = ProductionPipeline(config)
        
        # Check if PDF
        is_pdf = file_info.get("is_pdf", False)
        page_to_process = request.config_overrides.get("pdf_page", 0) if request.config_overrides else 0
        
        if is_pdf and file_id in pdf_pages_db:
            # For PDF, process specified page or all pages
            process_all_pages = request.config_overrides.get("pdf_process_all", False) if request.config_overrides else False
            
            if process_all_pages:
                # Process all pages and merge manifests
                all_pages = load_image_auto(image_path, dpi=300)
                job["warnings"] = [f"PDF with {len(all_pages)} pages, processing all"]
                
                merged_components = []
                merged_nets = []
                merged_issues = []
                total_timings = {}
                
                for page_idx, page_image in enumerate(all_pages):
                    job["current_stage"] = f"detection (page {page_idx+1}/{len(all_pages)})"
                    job["progress"] = int(10 + 80 * page_idx / len(all_pages))
                    job["updated_at"] = datetime.now()
                    
                    result = pipeline.run(page_image, f"{file_info['filename']}_page_{page_idx}", dpi=300)
                    
                    # Offset bboxes by page? For now keep separate with page prefix
                    for comp in result.manifest.components:
                        comp.id = f"p{page_idx}_{comp.id}"
                        # Store page info in text_associations
                        comp.text_associations["pdf_page"] = str(page_idx)
                    
                    for net in result.manifest.nets:
                        net.net_id = f"p{page_idx}_{net.net_id}"
                    
                    for issue in result.manifest.human_review_required:
                        # Add page info to description
                        issue.description = f"[Page {page_idx}] {issue.description}"
                    
                    merged_components.extend(result.manifest.components)
                    merged_nets.extend(result.manifest.nets)
                    merged_issues.extend(result.manifest.human_review_required)
                    
                    for k, v in result.stage_timings.items():
                        total_timings[k] = total_timings.get(k, 0) + v
                
                # Create merged manifest
                from avers.core.types import AVERSManifest, SchemaMetadata
                first_page = all_pages[0]
                merged_manifest = AVERSManifest(
                    schema_metadata=SchemaMetadata(
                        source_file=file_info["filename"],
                        resolution_dpi=300,
                        width=first_page.shape[1],
                        height=first_page.shape[0],
                        format="PDF",
                    ),
                    components=merged_components,
                    nets=merged_nets,
                    human_review_required=merged_issues,
                )
                
                # Create PipelineResult
                from avers.core.validators import PipelineResult
                result = PipelineResult(
                    manifest=merged_manifest,
                    errors=[],
                    warnings=[f"Processed {len(all_pages)} PDF pages"],
                    stage_timings=total_timings,
                    success=True,
                )
            else:
                # Process single page (specified or first)
                if file_id in pdf_pages_db and len(pdf_pages_db[file_id]) > page_to_process:
                    # Load from cached page image
                    cached_path = pdf_pages_db[file_id][page_to_process]
                    pil_img = Image.open(cached_path)
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
                    image = np.array(pil_img)
                else:
                    pages = load_image_auto(image_path, dpi=300, page_numbers=[page_to_process])
                    image = pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
                
                job["current_stage"] = "detection"
                job["progress"] = 10
                job["updated_at"] = datetime.now()
                
                result = pipeline.run(image, f"{file_info['filename']}_page_{page_to_process}", dpi=300)
        else:
            # Regular single image
            pages = load_image_auto(image_path, dpi=300, max_pages=1)
            image = pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
            
            job["current_stage"] = "detection"
            job["progress"] = 10
            job["updated_at"] = datetime.now()
            
            result = pipeline.run(image, file_info["filename"], dpi=300)
        
        # Update job
        job["stage_timings"] = result.stage_timings
        job["errors"] = result.errors
        job["warnings"] = result.warnings
        job["progress"] = 100
        job["current_stage"] = "completed"
        job["status"] = JobStatus.COMPLETED
        job["updated_at"] = datetime.now()
        
        manifests_db[file_id] = result.manifest
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
async def get_visualization(file_id: str, stage: str, page: int = Query(0, ge=0)):
    """Get visualization for specific stage - enhanced with PDF support."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    file_info = files_db[file_id]
    
    # Load image - handle PDF
    try:
        if file_info.get("is_pdf") and file_id in pdf_pages_db:
            pages = pdf_pages_db[file_id]
            if page < len(pages):
                pil_img = Image.open(pages[page])
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                image = np.array(pil_img)
            else:
                raise HTTPException(404, f"Page {page} not found")
        else:
            from avers.utils.image_helpers import load_image_auto
            pages = load_image_auto(Path(file_info["path"]), dpi=300, max_pages=1)
            image = pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {e}")
    
    vis_path = RESULTS_DIR / f"{file_id}_vis_{stage}_p{page}.png"
    
    try:
        if stage == "tiles":
            from avers.web.visualization import visualize_sahi_tiles
            vis = visualize_sahi_tiles(image, tile_size=1024, overlap_ratio=0.2)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "detection":
            from avers.core.validators import SlicedDetector
            from avers.web.visualization import visualize_detections_enhanced
            detector = SlicedDetector(device="cpu")
            detections = detector.detect(image)
            vis = visualize_detections_enhanced(image, detections)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "ocr":
            from avers.core.validators import SchematicOCR
            from avers.web.visualization import visualize_ocr_enhanced
            ocr = SchematicOCR()
            texts = ocr.recognize(image)
            vis = visualize_ocr_enhanced(image, texts)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "vectorization":
            from avers.web.visualization import visualize_vectorization_steps
            # Get exclusion bboxes from manifest if available
            exclusion = []
            if file_id in manifests_db:
                manifest = manifests_db[file_id]
                exclusion = [c.bbox for c in manifest.components]
            
            steps = visualize_vectorization_steps(image, exclusion if exclusion else None)
            # For main endpoint, return final
            vis = steps.get("final", image)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "graph":
            if file_id not in manifests_db:
                # Try to generate from current image
                from avers.core.validators import SlicedDetector, WireVectorizer
                from avers.stages.stage5_graph_synthesis import GraphBuilder, PinReference
                from avers.web.visualization import visualize_detections_enhanced
                
                detector = SlicedDetector(device="cpu")
                detections = detector.detect(image)
                
                vis = image.copy()
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            else:
                manifest = manifests_db[file_id]
                vis = image.copy()
                for comp in manifest.components:
                    x1, y1, x2, y2 = comp.bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, comp.designator, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    for pin in comp.pins:
                        cv2.circle(vis, tuple(pin.coord), 4, (255, 0, 0), -1)
                for net in manifest.nets:
                    pts = net.path_points
                    for i in range(len(pts)-1):
                        cv2.line(vis, pts[i], pts[i+1], (0, 0, 255), 1)
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        elif stage == "vlm":
            # VLM ROI visualization - show issue crops
            if file_id not in manifests_db:
                raise HTTPException(404, "Run processing first")
            manifest = manifests_db[file_id]
            
            # Create grid of issue ROIs
            issues = manifest.human_review_required[:6]
            if not issues:
                # No issues - create placeholder
                vis = np.ones((256, 512, 3), dtype=np.uint8) * 255
                cv2.putText(vis, "No issues for VLM arbitration", (20, 128),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
            else:
                from avers.web.visualization import visualize_vlm_arbitration
                # For demo, show first issue
                first_issue = issues[0]
                viz_dict = visualize_vlm_arbitration(image, first_issue.bbox, roi_size=256)
                vis = viz_dict.get("context", image)
                # Resize for display
                if vis.shape[0] > 1000 or vis.shape[1] > 1000:
                    vis = cv2.resize(vis, (512, 512))
            
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        else:
            raise HTTPException(400, f"Unknown stage: {stage}. Known: tiles, detection, ocr, vectorization, graph, vlm")
        
        return FileResponse(str(vis_path))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")


@router.get("/visualization/{file_id}/vectorization/{substep}")
async def get_vectorization_substep(file_id: str, substep: str, page: int = Query(0, ge=0)):
    """Get specific vectorization substep: binary, masked, skeleton, segments, junctions, final."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    file_info = files_db[file_id]
    
    try:
        if file_info.get("is_pdf") and file_id in pdf_pages_db:
            pages = pdf_pages_db[file_id]
            pil_img = Image.open(pages[page])
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image = np.array(pil_img)
        else:
            from avers.utils.image_helpers import load_image_auto
            pages = load_image_auto(Path(file_info["path"]), dpi=300, max_pages=1)
            image = pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {e}")
    
    vis_path = RESULTS_DIR / f"{file_id}_vec_{substep}_p{page}.png"
    
    try:
        from avers.web.visualization import visualize_vectorization_steps
        
        exclusion = []
        if file_id in manifests_db:
            manifest = manifests_db[file_id]
            exclusion = [c.bbox for c in manifest.components]
        
        steps = visualize_vectorization_steps(image, exclusion if exclusion else None)
        
        if substep not in steps:
            raise HTTPException(400, f"Unknown substep: {substep}. Known: {list(steps.keys())}")
        
        vis = steps[substep]
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        return FileResponse(str(vis_path))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")


@router.get("/graph/{file_id}/interactive")
async def get_interactive_graph(file_id: str):
    """Get interactive graph JSON for D3/vis-network."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    if file_id not in manifests_db:
        result_path = RESULTS_DIR / f"{file_id}.json"
        if not result_path.exists():
            raise HTTPException(404, "Run processing first")
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from avers.core.types import AVERSManifest
        manifest = AVERSManifest(**data)
    else:
        manifest = manifests_db[file_id]
    
    try:
        from avers.web.visualization import visualize_graph_interactive
        graph_data = visualize_graph_interactive(manifest.components, manifest.nets)
        return graph_data
    except Exception as e:
        raise HTTPException(500, f"Graph generation failed: {e}")


@router.get("/report/{file_id}")
async def generate_report(file_id: str, page: int = Query(0, ge=0)):
    """Generate HTML report with all stages."""
    if file_id not in files_db:
        raise HTTPException(404, "File not found")
    
    file_info = files_db[file_id]
    
    try:
        if file_info.get("is_pdf") and file_id in pdf_pages_db:
            pages = pdf_pages_db[file_id]
            pil_img = Image.open(pages[page])
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image = np.array(pil_img)
        else:
            from avers.utils.image_helpers import load_image_auto
            pages = load_image_auto(Path(file_info["path"]), dpi=300, max_pages=1)
            image = pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {e}")
    
    if file_id not in manifests_db:
        result_path = RESULTS_DIR / f"{file_id}.json"
        if not result_path.exists():
            raise HTTPException(404, "Run processing first")
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from avers.core.types import AVERSManifest
        manifest = AVERSManifest(**data)
    else:
        manifest = manifests_db[file_id]
    
    # Get job timings
    job_timings = {}
    for job in jobs_db.values():
        if job["file_id"] == file_id:
            job_timings = job.get("stage_timings", {})
            break
    
    try:
        from avers.web.visualization import create_pipeline_report
        
        report_path = RESULTS_DIR / f"{file_id}_report_p{page}.html"
        create_pipeline_report(image, manifest, job_timings, report_path)
        
        return FileResponse(str(report_path), media_type="text/html")
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


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
