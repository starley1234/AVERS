"""Tests for the v0.3 Active Learning Loop: feedback, metrics, model
registry, scheduler and dataset merge.

These tests are intentionally CPU-only / dependency-light (no torch,
ultralytics, GPU) so they run in any sandbox, including the one used to
*prepare* v0.3 before handing the GPU-heavy training work to another
machine. See docs/ACTIVE_LEARNING_V03.md.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


# --------------------------------------------------------------------- #
# FeedbackEntry / ActiveLearningLoop
# --------------------------------------------------------------------- #

class TestFeedbackEntry:
    def test_roundtrip_dict(self):
        from avers.active_learning.loop import FeedbackEntry

        entry = FeedbackEntry(
            file_id="f1",
            bbox=(1, 2, 3, 4),
            original_label="diode",
            corrected_label="resistor",
            issue_type="low_confidence_detection",
            comment="test",
        )
        data = entry.to_dict()
        restored = FeedbackEntry.from_dict(data)
        assert restored.id == entry.id
        assert restored.original_label == "diode"
        assert restored.corrected_label == "resistor"
        assert tuple(restored.bbox) == (1, 2, 3, 4)


class TestActiveLearningLoop:
    def _make_loop(self, tmp_path, **kwargs):
        from avers.active_learning.loop import ActiveLearningLoop

        return ActiveLearningLoop(feedback_dir=tmp_path / "feedback", rag_enabled=False, **kwargs)

    def test_add_feedback_persists_and_loads(self, tmp_path):
        loop = self._make_loop(tmp_path, min_feedback_for_retrain=5)
        for i in range(3):
            loop.add_feedback(
                file_id=f"file{i}",
                bbox=(0, 0, 10, 10),
                original_label="diode",
                corrected_label="resistor",
                issue_type="low_confidence_detection",
            )
        assert len(loop.feedback_entries) == 3
        assert not loop.should_retrain()

        # Reload from disk in a fresh instance
        from avers.active_learning.loop import ActiveLearningLoop

        loop2 = ActiveLearningLoop(feedback_dir=tmp_path / "feedback", rag_enabled=False)
        assert len(loop2.feedback_entries) == 3

    def test_should_retrain_threshold(self, tmp_path):
        loop = self._make_loop(tmp_path, min_feedback_for_retrain=2)
        assert not loop.should_retrain()
        loop.add_feedback("f1", (0, 0, 1, 1), "a", "b")
        loop.add_feedback("f2", (0, 0, 1, 1), "a", "b")
        assert loop.should_retrain()
        # explicit threshold overrides default
        assert not loop.should_retrain(threshold=10)

    def test_correction_stats(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop.add_feedback("f1", (0, 0, 1, 1), "diode", "resistor", issue_type="t1")
        loop.add_feedback("f2", (0, 0, 1, 1), "diode", "capacitor", issue_type="t1")
        stats = loop.get_correction_stats()
        assert stats["total"] == 2
        assert stats["by_original_label"]["diode"] == 2
        assert stats["unique_files"] == 2

    def test_export_for_training_creates_yolo_layout(self, tmp_path):
        loop = self._make_loop(tmp_path)
        crop = np.ones((64, 64, 3), dtype=np.uint8) * 200
        loop.add_feedback(
            "f1", (0, 0, 10, 10), "diode", "resistor",
            issue_type="t1", image_crop=crop,
        )
        out_dir = tmp_path / "export"
        yaml_path = loop.export_for_training(out_dir)
        assert yaml_path.exists()
        images = list((out_dir / "images" / "train").glob("*.jpg"))
        labels = list((out_dir / "labels" / "train").glob("*.txt"))
        assert len(images) == 1
        assert len(labels) == 1

    def test_clear_removes_entries(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop.add_feedback("f1", (0, 0, 1, 1), "a", "b")
        assert len(loop.feedback_entries) == 1
        loop.clear()
        assert len(loop.feedback_entries) == 0
        assert not loop.meta_file.exists()

    def test_trigger_retraining_without_ultralytics_is_graceful(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop.add_feedback("f1", (0, 0, 1, 1), "a", "b")
        result = loop.trigger_retraining(output_dir=tmp_path / "runs", epochs=1)
        # Either ultralytics is missing (status failed with a message) or,
        # if present in this environment, training itself may fail fast on a
        # tiny fake dataset. Either way it must not raise and must return a
        # status field.
        assert "status" in result
        assert "feedback_samples" in result


# --------------------------------------------------------------------- #
# ActiveLearningMetrics
# --------------------------------------------------------------------- #

class TestActiveLearningMetrics:
    def test_from_feedback_empty(self):
        from avers.active_learning.metrics import ActiveLearningMetrics

        m = ActiveLearningMetrics.from_feedback([])
        assert m.total_feedback == 0

    def test_from_feedback_aggregates(self):
        from avers.active_learning.loop import FeedbackEntry
        from avers.active_learning.metrics import ActiveLearningMetrics

        entries = [
            FeedbackEntry(original_label="diode", corrected_label="resistor"),
            FeedbackEntry(original_label="diode", corrected_label="resistor"),
            FeedbackEntry(original_label="ground", corrected_label="shield"),
        ]
        m = ActiveLearningMetrics.from_feedback(entries)
        assert m.total_feedback == 3
        assert m.most_corrected_classes[0] == ("diode->resistor", 2)
        d = m.to_dict()
        assert d["total_feedback"] == 3


# --------------------------------------------------------------------- #
# ModelRegistry
# --------------------------------------------------------------------- #

class TestModelRegistry:
    def _registry(self, tmp_path):
        from avers.active_learning.registry import ModelRegistry

        return ModelRegistry(root=tmp_path / "registry")

    def test_register_first_version_becomes_current(self, tmp_path):
        reg = self._registry(tmp_path)
        v = reg.register(
            weights_path=tmp_path / "nonexistent.pt",
            model_type="rtdetr-l",
            metrics={"mAP50": 0.5},
            copy_weights=False,
        )
        assert v.status == "current"
        assert reg.get_current().version_id == v.version_id

    def test_register_second_version_is_candidate(self, tmp_path):
        reg = self._registry(tmp_path)
        v1 = reg.register(tmp_path / "w1.pt", "rtdetr-l", copy_weights=False)
        v2 = reg.register(tmp_path / "w2.pt", "rtdetr-l", copy_weights=False)
        assert v1.status == "current"
        assert v2.status == "candidate"
        assert v2.parent_version == v1.version_id

    def test_promote_and_rollback(self, tmp_path):
        reg = self._registry(tmp_path)
        v1 = reg.register(tmp_path / "w1.pt", "rtdetr-l", copy_weights=False)
        v2 = reg.register(tmp_path / "w2.pt", "rtdetr-l", copy_weights=False)
        reg.promote(v2.version_id)
        assert reg.get_current().version_id == v2.version_id
        assert reg.get(v1.version_id).status == "retired"

        rolled_back = reg.rollback()
        assert rolled_back.version_id == v1.version_id
        assert reg.get_current().version_id == v1.version_id

    def test_registry_persists_across_instances(self, tmp_path):
        from avers.active_learning.registry import ModelRegistry

        reg = ModelRegistry(root=tmp_path / "registry")
        v1 = reg.register(tmp_path / "w1.pt", "rtdetr-l", copy_weights=False)

        reg2 = ModelRegistry(root=tmp_path / "registry")
        assert reg2.get_current().version_id == v1.version_id
        assert len(reg2.list_versions()) == 1

    def test_copy_weights_when_file_exists(self, tmp_path):
        reg = self._registry(tmp_path)
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake-weights")
        v = reg.register(weights, "rtdetr-l")
        stored = Path(v.weights_path)
        assert stored.exists()
        assert stored.read_bytes() == b"fake-weights"
        assert stored != weights

    def test_compare_versions(self, tmp_path):
        reg = self._registry(tmp_path)
        v1 = reg.register(tmp_path / "w1.pt", "rtdetr-l", metrics={"mAP50": 0.5}, copy_weights=False)
        v2 = reg.register(tmp_path / "w2.pt", "rtdetr-l", metrics={"mAP50": 0.6}, copy_weights=False)
        diff = reg.compare(v1.version_id, v2.version_id)
        assert diff["metrics"]["mAP50"]["delta"] == pytest.approx(0.1)

    def test_ab_test_routing_is_deterministic(self, tmp_path):
        reg = self._registry(tmp_path)
        v1 = reg.register(tmp_path / "w1.pt", "rtdetr-l", copy_weights=False)
        v2 = reg.register(tmp_path / "w2.pt", "rtdetr-l", copy_weights=False)
        reg.start_ab_test(v2.version_id, traffic_ratio=1.0)  # route everything to challenger

        chosen1 = reg.route("request-abc")
        chosen2 = reg.route("request-abc")
        assert chosen1.version_id == chosen2.version_id == v2.version_id

        reg.stop_ab_test()
        assert reg.route("request-abc").version_id == v1.version_id

    def test_ab_summary(self, tmp_path):
        reg = self._registry(tmp_path)
        v1 = reg.register(tmp_path / "w1.pt", "rtdetr-l", copy_weights=False)
        reg.log_ab_result(v1.version_id, {"accepted": True})
        reg.log_ab_result(v1.version_id, {"accepted": False})
        summary = reg.ab_summary()
        assert summary["total"] == 2
        assert summary["by_version"][v1.version_id]["count"] == 2
        assert summary["by_version"][v1.version_id]["accepted"] == 1


# --------------------------------------------------------------------- #
# Notifier
# --------------------------------------------------------------------- #

class TestNotifier:
    def test_notify_writes_log_file(self, tmp_path):
        from avers.active_learning.notify import Notifier

        log_file = tmp_path / "notif.jsonl"
        notifier = Notifier(webhook_url=None, log_file=log_file)
        notifier.notify("retrain_due", {"count": 5}, message="hello")

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "retrain_due"
        assert entry["payload"]["count"] == 5

    def test_get_history_respects_limit(self, tmp_path):
        from avers.active_learning.notify import Notifier

        notifier = Notifier(webhook_url=None, log_file=tmp_path / "notif.jsonl")
        for i in range(5):
            notifier.notify("event", {"i": i})
        history = notifier.get_history(limit=2)
        assert len(history) == 2
        assert history[-1]["payload"]["i"] == 4

    def test_bad_webhook_does_not_raise(self, tmp_path):
        from avers.active_learning.notify import Notifier

        notifier = Notifier(webhook_url="http://127.0.0.1:1/nope", log_file=tmp_path / "notif.jsonl")
        # Should not raise even though the webhook is unreachable.
        notifier.notify("event", {}, message="should not crash")


# --------------------------------------------------------------------- #
# RetrainScheduler
# --------------------------------------------------------------------- #

class TestRetrainScheduler:
    def _loop(self, tmp_path, threshold=2):
        from avers.active_learning.loop import ActiveLearningLoop

        return ActiveLearningLoop(
            feedback_dir=tmp_path / "feedback", rag_enabled=False, min_feedback_for_retrain=threshold
        )

    def test_tick_skips_when_not_due(self, tmp_path):
        from avers.active_learning.scheduler import RetrainScheduler

        loop = self._loop(tmp_path, threshold=5)
        scheduler = RetrainScheduler(loop=loop, retrain_fn=lambda: {"status": "success"})
        result = scheduler.tick()
        assert result["status"] == "skipped"
        assert scheduler.state.total_retrains == 0

    def test_tick_retrains_when_due_and_registers_model(self, tmp_path):
        from avers.active_learning.scheduler import RetrainScheduler
        from avers.active_learning.registry import ModelRegistry

        loop = self._loop(tmp_path, threshold=1)
        loop.add_feedback("f1", (0, 0, 1, 1), "a", "b")

        weights = tmp_path / "best.pt"
        weights.write_bytes(b"fake")

        registry = ModelRegistry(root=tmp_path / "registry")
        calls = []

        def fake_retrain():
            calls.append(1)
            return {"status": "success", "weights_path": str(weights), "feedback_samples": 1}

        scheduler = RetrainScheduler(loop=loop, registry=registry, retrain_fn=fake_retrain)
        result = scheduler.tick()

        assert result["status"] == "retrained"
        assert len(calls) == 1
        assert scheduler.state.total_retrains == 1
        assert registry.get_current() is not None

    def test_tick_handles_failed_retrain(self, tmp_path):
        from avers.active_learning.scheduler import RetrainScheduler

        loop = self._loop(tmp_path, threshold=1)
        loop.add_feedback("f1", (0, 0, 1, 1), "a", "b")

        scheduler = RetrainScheduler(loop=loop, retrain_fn=lambda: {"status": "failed", "error": "no gpu"})
        result = scheduler.tick()
        assert result["status"] == "retrained"
        assert result["result"]["status"] == "failed"

    def test_start_stop_background_thread(self, tmp_path):
        from avers.active_learning.scheduler import RetrainScheduler

        loop = self._loop(tmp_path, threshold=10_000)  # never due, keeps tick() cheap
        scheduler = RetrainScheduler(
            loop=loop, check_interval_sec=0.05, retrain_fn=lambda: {"status": "success"}
        )
        scheduler.start()
        time.sleep(0.2)
        status = scheduler.status()
        assert status["running"] is True
        assert status["total_checks"] >= 1
        scheduler.stop()
        assert scheduler.status()["running"] is False

    def test_notifier_called_on_retrain(self, tmp_path):
        from avers.active_learning.scheduler import RetrainScheduler
        from avers.active_learning.notify import Notifier

        loop = self._loop(tmp_path, threshold=1)
        loop.add_feedback("f1", (0, 0, 1, 1), "a", "b")
        notifier = Notifier(webhook_url=None, log_file=tmp_path / "notif.jsonl")

        scheduler = RetrainScheduler(
            loop=loop, notifier=notifier, retrain_fn=lambda: {"status": "success"}
        )
        scheduler.tick()
        history = notifier.get_history()
        event_types = [h["event_type"] for h in history]
        assert "retrain_due" in event_types
        assert "retrain_started" in event_types
        assert "retrain_done" in event_types


# --------------------------------------------------------------------- #
# Dataset merge
# --------------------------------------------------------------------- #

class TestDatasetMerge:
    def _make_yolo_dataset(self, root: Path, names, samples):
        """Create a minimal YOLO-format dataset.

        samples: list of (split, class_id, filename_stem)
        """
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for split, class_id, stem in samples:
            img_path = root / "images" / split / f"{stem}.jpg"
            img_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")  # not a real jpg, fine for this test
            label_path = root / "labels" / split / f"{stem}.txt"
            label_path.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = root / "dataset.yaml"
        names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        yaml_path.write_text(
            f"path: {root}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames:\n{names_block}\n",
            encoding="utf-8",
        )
        return yaml_path

    def test_merge_two_sources_remaps_classes(self, tmp_path):
        from avers.dataset.merge import DatasetSource, merge_datasets

        ds_a = tmp_path / "a"
        ds_b = tmp_path / "b"
        yaml_a = self._make_yolo_dataset(ds_a, ["resistor", "diode"], [("train", 0, "img1"), ("val", 1, "img2")])
        yaml_b = self._make_yolo_dataset(ds_b, ["diode", "capacitor"], [("train", 0, "img3")])

        out_dir = tmp_path / "merged"
        yaml_path = merge_datasets(
            [DatasetSource("a", yaml_a), DatasetSource("b", yaml_b)],
            out_dir,
            canonical_names={0: "resistor", 1: "diode", 2: "capacitor"},
        )

        assert yaml_path.exists()
        content = yaml_path.read_text(encoding="utf-8")
        assert "nc: 3" in content

        # img3 from source b used local class_id 0 ("diode"), which must be
        # remapped to canonical id 1.
        label_b = out_dir / "labels" / "train" / "b_img3.txt"
        assert label_b.exists()
        assert label_b.read_text().split()[0] == "1"

        # img1 from source a used local class_id 0 ("resistor" in its own
        # namespace), canonical id for resistor is 0 too.
        label_a = out_dir / "labels" / "train" / "a_img1.txt"
        assert label_a.read_text().split()[0] == "0"

    def test_merge_respects_oversample_weight(self, tmp_path):
        from avers.dataset.merge import DatasetSource, merge_datasets

        ds_a = tmp_path / "a"
        yaml_a = self._make_yolo_dataset(ds_a, ["resistor"], [("train", 0, "img1")])

        out_dir = tmp_path / "merged"
        merge_datasets([DatasetSource("a", yaml_a, weight=3)], out_dir)

        images = list((out_dir / "images" / "train").glob("a_img1*"))
        assert len(images) == 3

    def test_merge_falls_back_val_from_train_when_empty(self, tmp_path):
        from avers.dataset.merge import DatasetSource, merge_datasets

        ds_a = tmp_path / "a"
        yaml_a = self._make_yolo_dataset(ds_a, ["resistor"], [("train", 0, "img1")])
        out_dir = tmp_path / "merged"
        merge_datasets([DatasetSource("a", yaml_a)], out_dir)

        val_images = list((out_dir / "images" / "val").glob("*"))
        assert len(val_images) >= 1

    def test_merge_missing_source_is_skipped_gracefully(self, tmp_path):
        from avers.dataset.merge import DatasetSource, merge_datasets

        out_dir = tmp_path / "merged"
        yaml_path = merge_datasets(
            [DatasetSource("missing", tmp_path / "does_not_exist.yaml")], out_dir
        )
        assert yaml_path.exists()
