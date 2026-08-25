"""Tests for Web UI module."""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

# Test that web module imports
def test_web_imports():
    from avers.web.app import create_app
    from avers.web.models import JobStatus, UploadResponse
    from avers.web.api import router
    
    assert create_app is not None
    assert JobStatus.PENDING == "pending"


def test_create_app():
    from avers.web.app import create_app
    app = create_app()
    assert app is not None
    assert app.title.startswith("АВЕРС")


def test_annotator_imports():
    from avers.web.annotator_api import router
    from avers.dataset.annotator import AnnotationStore, get_store
    
    assert router is not None
    assert AnnotationStore is not None


def test_active_learning_imports():
    from avers.active_learning import ActiveLearningLoop, FeedbackEntry
    from avers.active_learning.loop import get_active_learning_loop
    
    assert ActiveLearningLoop is not None
    assert FeedbackEntry is not None


def test_feedback_entry():
    from avers.active_learning.loop import FeedbackEntry
    
    entry = FeedbackEntry(
        file_id="test123",
        bbox=(10, 20, 100, 200),
        original_label="diode",
        corrected_label="resistor",
        issue_type="low_confidence_detection",
        comment="This is resistor"
    )
    
    d = entry.to_dict()
    assert d["file_id"] == "test123"
    assert d["original_label"] == "diode"
    assert d["corrected_label"] == "resistor"
    
    # From dict
    entry2 = FeedbackEntry.from_dict(d)
    assert entry2.file_id == "test123"
    assert entry2.corrected_label == "resistor"


def test_active_learning_loop():
    from avers.active_learning.loop import ActiveLearningLoop
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        loop = ActiveLearningLoop(feedback_dir=Path(tmpdir), rag_enabled=False)
        assert len(loop.feedback_entries) == 0
        
        # Add feedback
        entry = loop.add_feedback(
            file_id="file1",
            bbox=(0, 0, 10, 10),
            original_label="diode",
            corrected_label="resistor",
            issue_type="test",
            comment="correction"
        )
        
        assert len(loop.feedback_entries) == 1
        assert entry.corrected_label == "resistor"
        
        # Stats
        stats = loop.get_correction_stats()
        assert stats["total"] == 1
        assert "diode->resistor" in stats["by_correction"]
        
        # Should retrain check
        assert loop.should_retrain(threshold=1) is True
        assert loop.should_retrain(threshold=10) is False


def test_web_models():
    from avers.web.models import UploadResponse, JobStatusResponse, JobStatus
    from datetime import datetime
    
    # Test enums
    assert JobStatus.PENDING == "pending"
    assert JobStatus.RUNNING == "running"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"
