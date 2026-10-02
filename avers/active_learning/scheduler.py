"""
RetrainScheduler — фоновый планировщик дообучения для v0.3 Active Learning Loop.

Периодически проверяет ``ActiveLearningLoop.should_retrain()`` и, когда порог
накопленного feedback достигнут, запускает дообучение, опционально
регистрирует результат в ``ModelRegistry`` и шлёт уведомления через
``Notifier``.

Два режима использования:

  1. Blocking/standalone — для cron/systemd или ручного запуска на GPU-машине::

        from avers.active_learning.loop import get_active_learning_loop
        from avers.active_learning.scheduler import RetrainScheduler

        scheduler = RetrainScheduler(loop=get_active_learning_loop())
        scheduler.run_forever()   # блокирует текущий процесс

  2. Background thread — встраивается в FastAPI-приложение (Web UI), чтобы
     можно было стартовать/останавливать планировщик через API, не поднимая
     отдельный процесс::

        scheduler.start()
        ...
        scheduler.stop()

Вся тяжёлая работа (загрузка ultralytics, обучение на GPU) делегируется в
``ActiveLearningLoop.trigger_retraining`` (которая уже умеет работать без
ultralytics — просто вернёт ``status=failed``), поэтому сам планировщик
полностью тестируется на CPU без GPU/torch — см. tests/test_active_learning.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from avers.core.logger import get_logger

logger = get_logger("avers.active_learning.scheduler")

RetrainFn = Callable[[], Dict[str, Any]]


@dataclass
class SchedulerState:
    running: bool = False
    last_check_at: Optional[str] = None
    last_retrain_at: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    total_retrains: int = 0
    total_checks: int = 0


class RetrainScheduler:
    """Периодически проверяет и (при необходимости) запускает дообучение."""

    def __init__(
        self,
        loop=None,
        registry=None,
        notifier=None,
        check_interval_sec: int = 3600,
        min_feedback_for_retrain: Optional[int] = None,
        model_type: str = "rtdetr-l",
        epochs: int = 20,
        retrain_fn: Optional[RetrainFn] = None,
        on_retrain_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        # Lazy imports to keep this module importable standalone/in tests.
        if loop is None:
            from avers.active_learning.loop import get_active_learning_loop
            loop = get_active_learning_loop()
        self.loop = loop
        self.registry = registry
        self.notifier = notifier
        self.check_interval_sec = check_interval_sec
        self.min_feedback_for_retrain = min_feedback_for_retrain
        self.model_type = model_type
        self.epochs = epochs
        # Allow dependency injection of the (possibly expensive) retrain call,
        # so unit tests don't need ultralytics/torch/a GPU.
        self._retrain_fn = retrain_fn or (
            lambda: self.loop.trigger_retraining(model_type=self.model_type, epochs=self.epochs)
        )
        self._on_retrain_complete = on_retrain_complete

        self.state = SchedulerState()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Core logic (single check), fully unit-testable
    # ------------------------------------------------------------------ #
    def tick(self) -> Dict[str, Any]:
        """Выполнить одну проверку/запуск. Возвращает результат проверки."""
        with self._lock:
            self.state.total_checks += 1
            self.state.last_check_at = datetime.now().isoformat()

            due = self.loop.should_retrain(threshold=self.min_feedback_for_retrain)
            if not due:
                return {"status": "skipped", "reason": "threshold_not_reached"}

            feedback_count = len(self.loop.feedback_entries)
            if self.notifier:
                self.notifier.notify(
                    "retrain_due",
                    {"feedback_count": feedback_count},
                    message=f"Retrain threshold reached ({feedback_count} samples)",
                )

            if self.notifier:
                self.notifier.notify("retrain_started", {"feedback_count": feedback_count})

            try:
                result = self._retrain_fn()
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("Retrain function raised an exception")
                result = {"status": "failed", "error": str(e)}

            self.state.last_retrain_at = datetime.now().isoformat()
            self.state.last_result = result
            self.state.total_retrains += 1

            if result.get("status") == "success":
                if self.notifier:
                    self.notifier.notify("retrain_done", result, message="Retraining finished")
                if self.registry is not None:
                    self._maybe_register(result)
            else:
                if self.notifier:
                    self.notifier.notify("retrain_failed", result, message="Retraining failed")

            if self._on_retrain_complete:
                try:
                    self._on_retrain_complete(result)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("on_retrain_complete callback failed")

            return {"status": "retrained", "result": result}

    def _maybe_register(self, result: Dict[str, Any]) -> None:
        """Попытаться найти обученные веса и зарегистрировать версию."""
        weights_path = result.get("weights_path")
        if not weights_path:
            # Best-effort guess at ultralytics' default output layout.
            output_dir = result.get("output_dir")
            if output_dir:
                candidates = list(Path(output_dir).glob("runs/**/weights/best.pt"))
                if candidates:
                    weights_path = str(candidates[0])
        if not weights_path:
            logger.info("Retrain succeeded but no weights path found; skipping registry")
            return
        try:
            version = self.registry.register(
                weights_path=weights_path,
                model_type=result.get("model_type", self.model_type),
                metrics=result.get("metrics", {}),
                dataset_info={
                    "feedback_samples": result.get("feedback_samples"),
                    "epochs": result.get("epochs"),
                    "trigger": "scheduler",
                },
                notes="Auto-registered by RetrainScheduler",
            )
            logger.info(f"Registered new model version from scheduler: {version.version_id}")
        except Exception as e:
            logger.warning(f"Failed to register model version: {e}")

    # ------------------------------------------------------------------ #
    # Blocking loop (for standalone/cron usage)
    # ------------------------------------------------------------------ #
    def run_forever(self) -> None:
        logger.info(
            f"RetrainScheduler running (interval={self.check_interval_sec}s, "
            f"threshold={self.min_feedback_for_retrain or self.loop.min_feedback_for_retrain})"
        )
        self.state.running = True
        self._stop_event.clear()
        try:
            while not self._stop_event.is_set():
                self.tick()
                self._stop_event.wait(self.check_interval_sec)
        finally:
            self.state.running = False

    # ------------------------------------------------------------------ #
    # Background thread (for embedding in a long-running web server)
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True, name="avers-retrain-scheduler")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.state.running = False

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "check_interval_sec": self.check_interval_sec,
            "min_feedback_for_retrain": self.min_feedback_for_retrain or self.loop.min_feedback_for_retrain,
            "total_checks": self.state.total_checks,
            "total_retrains": self.state.total_retrains,
            "last_check_at": self.state.last_check_at,
            "last_retrain_at": self.state.last_retrain_at,
            "last_result": self.state.last_result,
        }


_global_scheduler: Optional[RetrainScheduler] = None


def get_scheduler(**kwargs) -> RetrainScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = RetrainScheduler(**kwargs)
    return _global_scheduler
