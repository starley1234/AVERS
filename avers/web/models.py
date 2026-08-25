"""Pydantic models for Web API."""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    width: int
    height: int
    size_bytes: int
    preview_url: str
    is_pdf: bool = False
    num_pages: int = 1


class ProcessRequest(BaseModel):
    config_overrides: Optional[Dict[str, Any]] = None
    stages: Optional[List[str]] = None  # which stages to run
    pdf_page: int = 0  # for PDF: which page to process
    pdf_process_all: bool = False  # for PDF: process all pages and merge


class JobStatusResponse(BaseModel):
    job_id: str
    file_id: str
    status: JobStatus
    progress: int = 0  # 0-100
    current_stage: Optional[str] = None
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ComponentEditRequest(BaseModel):
    designator: Optional[str] = None
    type: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    pins: Optional[List[Dict[str, Any]]] = None
    part_number: Optional[str] = None
    confidence: Optional[float] = None


class IssueResolutionRequest(BaseModel):
    resolved: bool
    resolution: Optional[str] = None
    corrected_bbox: Optional[Tuple[int, int, int, int]] = None
    corrected_text: Optional[str] = None
    connected: Optional[bool] = None  # for junction issues


class DatasetItem(BaseModel):
    id: str
    image_path: str
    width: int
    height: int
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class RAGQueryRequest(BaseModel):
    text_query: Optional[str] = None
    image_base64: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    top_k: int = 5
    use_vlm: bool = True


class RAGIndexRequest(BaseModel):
    image_base64: str
    bbox: Tuple[int, int, int, int]
    label: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    id: str
    label: str
    score: float
    bbox: Tuple[int, int, int, int]
    description: Optional[str] = None
    image_preview: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    query: str
    results: List[RAGResult]
    vlm_answer: Optional[Dict[str, Any]] = None
    processing_time_ms: float
