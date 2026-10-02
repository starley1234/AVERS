"""
Active Learning v0.3 API — scheduler control, model registry, metrics history.

Complements the existing feedback endpoints in ``avers/web/api.py``
(``/api/active-learning/stats|retrain|clear``) with the pieces needed to
close the loop end-to-end:

  - GET  /api/active-learning/history         -> time-series metrics for a dashboard
  - GET  /api/active-learning/scheduler/status
  - POST /api/active-learning/scheduler/start
  - POST /api/active-learning/scheduler/stop
  - GET  /api/active-learning/registry
  - POST /api/active-learning/registry/{version_id}/promote
  - POST /api/active-learning/registry/rollback
  - POST /api/active-learning/registry/{version_id}/ab-test?traffic_ratio=0.2
  - POST /api/active-learning/registry/ab-test/stop
  - GET  /api/active-learning/notifications

These are deliberately thin wrappers around the pure-Python modules in
``avers/active_learning/`` (registry.py, scheduler.py, notify.py) so they
work without GPU/ML dependencies — only the actual retraining call
(delegated to ActiveLearningLoop.trigger_retraining) needs ultralytics/torch.
"""

from fastapi import APIRouter, HTTPException, Query

from avers.core.logger import get_logger

logger = get_logger("avers.web.active_learning_api")

router = APIRouter(prefix="/api/active-learning")


def _loop():
    from avers.active_learning.loop import get_active_learning_loop

    return get_active_learning_loop()


def _registry():
    from avers.active_learning.registry import get_model_registry

    return get_model_registry()


def _notifier():
    from avers.active_learning.notify import get_notifier

    return get_notifier()


def _scheduler():
    from avers.active_learning.scheduler import get_scheduler

    return get_scheduler(loop=_loop(), registry=_registry(), notifier=_notifier())


@router.get("/history")
async def get_active_learning_history():
    """Time-series-ish metrics for a dashboard (feedback/day, top corrections)."""
    try:
        from avers.active_learning.metrics import ActiveLearningMetrics

        loop = _loop()
        metrics = ActiveLearningMetrics.from_feedback(loop.feedback_entries)
        return metrics.to_dict()
    except Exception as e:
        raise HTTPException(500, f"Failed to compute history: {e}")


@router.get("/scheduler/status")
async def scheduler_status():
    return _scheduler().status()


@router.post("/scheduler/start")
async def scheduler_start():
    scheduler = _scheduler()
    scheduler.start()
    return scheduler.status()


@router.post("/scheduler/stop")
async def scheduler_stop():
    scheduler = _scheduler()
    scheduler.stop()
    return scheduler.status()


@router.get("/registry")
async def registry_status():
    return _registry().status()


@router.post("/registry/{version_id}/promote")
async def registry_promote(version_id: str):
    try:
        version = _registry().promote(version_id)
        return version.to_dict()
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/registry/rollback")
async def registry_rollback():
    version = _registry().rollback()
    if version is None:
        raise HTTPException(400, "No previous version to roll back to")
    return version.to_dict()


@router.post("/registry/{version_id}/ab-test")
async def registry_start_ab_test(version_id: str, traffic_ratio: float = Query(0.1, gt=0.0, le=1.0)):
    try:
        return _registry().start_ab_test(version_id, traffic_ratio=traffic_ratio)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/registry/ab-test/stop")
async def registry_stop_ab_test():
    _registry().stop_ab_test()
    return {"status": "stopped"}


@router.get("/registry/ab-test/summary")
async def registry_ab_summary():
    return _registry().ab_summary()


@router.get("/notifications")
async def get_notifications(limit: int = Query(50, ge=1, le=1000)):
    return {"notifications": _notifier().get_history(limit=limit)}
