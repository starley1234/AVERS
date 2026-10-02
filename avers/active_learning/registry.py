"""
Model Registry — версионирование моделей и A/B тестирование для АВЕРС.

Часть v0.3 Active Learning Loop (см. docs/ACTIVE_LEARNING_V03.md).

Каждый цикл дообучения (ActiveLearningLoop.trigger_retraining) производит
новые веса модели. Чтобы не терять историю и иметь возможность откатиться
на предыдущую версию (или сравнить две версии в проде через A/B), веса и
метаданные регистрируются здесь.

Дизайн специально НЕ зависит от torch/ultralytics — это чистая работа с
файлами и JSON, поэтому модуль полностью тестируется на CPU без GPU и ML
зависимостей.

Структура на диске:

    registry_root/
      registry.json              <- индекс версий, current/challenger, A/B лог
      versions/
        v000001/
          weights<.pt|.onnx|...> <- копия/симлинк на файл весов
          metadata.json          <- метрики, dataset info, timestamps
        v000002/
          ...
      ab_log.jsonl                <- лог результатов A/B тестирования

Использование:

    from avers.active_learning.registry import get_model_registry

    registry = get_model_registry()
    version = registry.register(
        weights_path="/tmp/avers_runs/avers_rtdetr/weights/best.pt",
        model_type="rtdetr-l",
        metrics={"mAP50": 0.81},
        dataset_info={"feedback_samples": 62, "source": "feedback+synthetic"},
    )
    registry.promote(version.version_id)
    ...
    # A/B test новой версии против текущей на 20% трафика
    registry.start_ab_test(challenger_id="v000002", traffic_ratio=0.2)
    chosen = registry.route(request_id="file_abc123")  # -> ModelVersion
    registry.log_ab_result(chosen.version_id, {"confidence": 0.9, "accepted": True})
    print(registry.ab_summary())
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from avers.core.logger import get_logger

logger = get_logger("avers.active_learning.registry")


@dataclass
class ModelVersion:
    """Одна зарегистрированная версия модели."""

    version_id: str
    model_type: str
    weights_path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    status: str = "candidate"  # candidate | current | challenger | retired | rejected
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(**data)


class ModelRegistry:
    """Файловый реестр версий моделей + простое A/B тестирование."""

    def __init__(self, root: Path = Path("/tmp/avers_model_registry")):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "registry.json"
        self.ab_log_file = self.root / "ab_log.jsonl"

        self._versions: Dict[str, ModelVersion] = {}
        self._current: Optional[str] = None
        self._ab_test: Optional[Dict[str, Any]] = None  # {challenger, traffic_ratio}
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.index_file.exists():
            return
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load registry index: {e}")
            return
        self._versions = {
            vid: ModelVersion.from_dict(v) for vid, v in data.get("versions", {}).items()
        }
        self._current = data.get("current")
        self._ab_test = data.get("ab_test")

    def _save(self) -> None:
        data = {
            "versions": {vid: v.to_dict() for vid, v in self._versions.items()},
            "current": self._current,
            "ab_test": self._ab_test,
        }
        self.index_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # Registration / promotion
    # ------------------------------------------------------------------ #
    def _next_version_id(self) -> str:
        n = len(self._versions) + 1
        vid = f"v{n:06d}"
        while vid in self._versions:
            n += 1
            vid = f"v{n:06d}"
        return vid

    def register(
        self,
        weights_path: Path | str,
        model_type: str,
        metrics: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
        notes: str = "",
        copy_weights: bool = True,
        auto_promote_if_first: bool = True,
    ) -> ModelVersion:
        """Зарегистрировать новую версию модели.

        Если ``weights_path`` существует, файл копируется в
        ``versions/<id>/`` (или сохраняется путь как есть, если
        ``copy_weights=False`` — удобно в тестах/при больших файлах на
        сетевом хранилище).
        """
        weights_path = Path(weights_path)
        version_id = self._next_version_id()
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        stored_path = str(weights_path)
        if copy_weights and weights_path.exists() and weights_path.is_file():
            dest = version_dir / weights_path.name
            try:
                shutil.copy2(weights_path, dest)
                stored_path = str(dest)
            except Exception as e:
                logger.warning(f"Could not copy weights into registry: {e}")
        elif not weights_path.exists():
            logger.warning(
                f"Weights path {weights_path} does not exist yet; "
                "registering metadata only (useful for dry-runs/tests)."
            )

        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            weights_path=stored_path,
            metrics=metrics or {},
            dataset_info=dataset_info or {},
            parent_version=parent_version or self._current,
            status="candidate",
            notes=notes,
        )
        (version_dir / "metadata.json").write_text(
            json.dumps(version.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._versions[version_id] = version

        if auto_promote_if_first and self._current is None:
            self.promote(version_id)
        else:
            self._save()

        logger.info(f"Registered model version {version_id} ({model_type})")
        return version

    def promote(self, version_id: str) -> ModelVersion:
        """Сделать версию текущей (production), предыдущую пометить retired."""
        if version_id not in self._versions:
            raise KeyError(f"Unknown model version: {version_id}")

        if self._current and self._current in self._versions and self._current != version_id:
            self._versions[self._current].status = "retired"

        self._versions[version_id].status = "current"
        self._current = version_id
        # Promoting ends any running A/B test for cleanliness.
        self._ab_test = None
        self._save()
        logger.info(f"Promoted model version {version_id} to current")
        return self._versions[version_id]

    def rollback(self) -> Optional[ModelVersion]:
        """Откатиться на родителя текущей версии (если есть)."""
        if not self._current:
            return None
        current = self._versions.get(self._current)
        if not current or not current.parent_version:
            logger.warning("No parent version to roll back to")
            return None
        return self.promote(current.parent_version)

    def reject(self, version_id: str, reason: str = "") -> ModelVersion:
        if version_id not in self._versions:
            raise KeyError(f"Unknown model version: {version_id}")
        v = self._versions[version_id]
        v.status = "rejected"
        v.notes = (v.notes + f" | rejected: {reason}").strip(" |")
        self._save()
        return v

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get(self, version_id: str) -> Optional[ModelVersion]:
        return self._versions.get(version_id)

    def get_current(self) -> Optional[ModelVersion]:
        return self._versions.get(self._current) if self._current else None

    def list_versions(self) -> List[ModelVersion]:
        return sorted(self._versions.values(), key=lambda v: v.version_id)

    def compare(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Сравнить метрики двух версий."""
        a, b = self._versions.get(version_a), self._versions.get(version_b)
        if not a or not b:
            raise KeyError("Unknown version(s) for comparison")
        keys = set(a.metrics) | set(b.metrics)
        diff = {}
        for k in keys:
            va, vb = a.metrics.get(k), b.metrics.get(k)
            delta = None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
            diff[k] = {"a": va, "b": vb, "delta": delta}
        return {"a": version_a, "b": version_b, "metrics": diff}

    # ------------------------------------------------------------------ #
    # A/B testing
    # ------------------------------------------------------------------ #
    def start_ab_test(self, challenger_id: str, traffic_ratio: float = 0.1) -> Dict[str, Any]:
        if challenger_id not in self._versions:
            raise KeyError(f"Unknown model version: {challenger_id}")
        if not (0.0 < traffic_ratio <= 1.0):
            raise ValueError("traffic_ratio must be in (0, 1]")
        self._versions[challenger_id].status = "challenger"
        self._ab_test = {"challenger": challenger_id, "traffic_ratio": traffic_ratio}
        self._save()
        logger.info(f"Started A/B test: challenger={challenger_id} ratio={traffic_ratio}")
        return self._ab_test

    def stop_ab_test(self) -> None:
        if self._ab_test:
            challenger_id = self._ab_test.get("challenger")
            if challenger_id in self._versions:
                self._versions[challenger_id].status = "retired"
        self._ab_test = None
        self._save()

    def route(self, request_id: str) -> Optional[ModelVersion]:
        """Выбрать версию модели для данного запроса (детерминированно).

        Используется стабильный хэш ``request_id``, чтобы один и тот же
        файл/пользователь всегда получал одну и ту же версию в рамках
        активного A/B теста (не "мигает" между моделями).
        """
        current = self.get_current()
        if not self._ab_test:
            return current

        challenger = self._versions.get(self._ab_test["challenger"])
        if not challenger:
            return current

        ratio = self._ab_test["traffic_ratio"]
        h = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest(), 16)
        bucket = (h % 10_000) / 10_000.0
        return challenger if bucket < ratio else current

    def log_ab_result(self, version_id: str, outcome: Dict[str, Any]) -> None:
        entry = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            **outcome,
        }
        with open(self.ab_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def ab_summary(self) -> Dict[str, Any]:
        if not self.ab_log_file.exists():
            return {"total": 0, "by_version": {}}
        by_version: Dict[str, Dict[str, Any]] = {}
        total = 0
        with open(self.ab_log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                total += 1
                vid = entry.get("version_id", "unknown")
                by_version.setdefault(vid, {"count": 0, "accepted": 0})
                by_version[vid]["count"] += 1
                if entry.get("accepted"):
                    by_version[vid]["accepted"] += 1
        return {"total": total, "by_version": by_version, "ab_test": self._ab_test}

    def status(self) -> Dict[str, Any]:
        return {
            "current": self._current,
            "ab_test": self._ab_test,
            "total_versions": len(self._versions),
            "versions": [v.to_dict() for v in self.list_versions()],
        }


_global_registry: Optional[ModelRegistry] = None


def get_model_registry(root: Path = Path("/tmp/avers_model_registry")) -> ModelRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry(root=root)
    return _global_registry
