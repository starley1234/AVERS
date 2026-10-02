"""AVERS Web App - FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from avers.web.api import router, rag_router
from avers.web.annotator_api import router as annotator_router
from avers.web.active_learning_api import router as active_learning_router


def _maybe_start_active_learning_scheduler() -> None:
    """Auto-start the retrain scheduler if enabled via
    AVERS_AL_SCHEDULER_ENABLED env var (off by default; also controllable
    via config.yaml `active_learning.scheduler_enabled` + CLI)."""
    import os

    enabled = os.environ.get("AVERS_AL_SCHEDULER_ENABLED", "").lower() in ("1", "true", "yes")
    if not enabled:
        return
    try:
        from avers.active_learning.loop import get_active_learning_loop
        from avers.active_learning.registry import get_model_registry
        from avers.active_learning.notify import get_notifier
        from avers.active_learning.scheduler import get_scheduler

        scheduler = get_scheduler(
            loop=get_active_learning_loop(),
            registry=get_model_registry(),
            notifier=get_notifier(),
        )
        scheduler.start()
    except Exception as e:  # pragma: no cover - defensive, startup must not crash
        import logging

        logging.getLogger("avers.web").warning(f"Could not start AL scheduler: {e}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _maybe_start_active_learning_scheduler()
    yield
    try:
        from avers.active_learning.scheduler import get_scheduler

        get_scheduler().stop(timeout=1.0)
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create FastAPI app."""
    app = FastAPI(
        title="АВЕРС - Автоматическая Векторизация и Распознавание Схем",
        description="Web UI валидатор для системы АВЕРС + Датасет инструменты + Vision RAG",
        version="0.2.0",
        lifespan=_lifespan,
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
    app.include_router(active_learning_router)
    
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
    
    @app.get("/visualize")
    async def serve_visualize():
        vis_path = frontend_dir / "visualize.html"
        if vis_path.exists():
            return FileResponse(str(vis_path))
        return {"message": "Visualize page not found"}
    
    @app.get("/annotator")
    async def serve_annotator():
        annotator_path = frontend_dir / "annotator.html"
        if annotator_path.exists():
            return FileResponse(str(annotator_path))
        return {"message": "Annotator not found"}

    @app.get("/dashboard")
    async def serve_dashboard():
        dashboard_path = frontend_dir / "dashboard.html"
        if dashboard_path.exists():
            return FileResponse(str(dashboard_path))
        return {"message": "Dashboard not found"}
    
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
