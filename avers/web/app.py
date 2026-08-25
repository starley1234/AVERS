"""AVERS Web App - FastAPI application."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from avers.web.api import router, rag_router
from avers.web.annotator_api import router as annotator_router


def create_app() -> FastAPI:
    """Create FastAPI app."""
    app = FastAPI(
        title="АВЕРС - Автоматическая Векторизация и Распознавание Схем",
        description="Web UI валидатор для системы АВЕРС + Датасет инструменты + Vision RAG",
        version="0.2.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API routes
    app.include_router(router)
    app.include_router(rag_router)
    app.include_router(annotator_router)
    
    # Static files
    static_dir = Path(__file__).parent / "static"
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Create dirs if not exist
    static_dir.mkdir(exist_ok=True)
    frontend_dir.mkdir(exist_ok=True)
    
    # Mount static if exists
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Frontend - serve index.html at root
    @app.get("/")
    async def serve_frontend():
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {
            "message": "AVERS API is running",
            "docs": "/docs",
            "frontend": "Frontend not built yet - check /docs for API"
        }
    
    @app.get("/annotator")
    async def serve_annotator():
        annotator_path = frontend_dir / "annotator.html"
        if annotator_path.exists():
            return FileResponse(str(annotator_path))
        return {"message": "Annotator not found"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "0.2.0"}
    
    # Mount frontend assets
    if frontend_dir.exists():
        # Serve frontend static files via /assets if needed
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    
    return app


# For uvicorn
app = create_app()
