"""Metrics for active learning."""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

from avers.active_learning.loop import FeedbackEntry


@dataclass
class ActiveLearningMetrics:
    """Metrics for active learning loop."""
    
    total_feedback: int = 0
    feedback_per_day: Dict[str, int] = field(default_factory=dict)
    accuracy_improvement: float = 0.0
    most_corrected_classes: List[tuple] = field(default_factory=list)
    avg_time_to_correction: float = 0.0
    
    @classmethod
    def from_feedback(cls, entries: List[FeedbackEntry]) -> "ActiveLearningMetrics":
        """Calculate metrics from feedback entries."""
        if not entries:
            return cls()
        
        # Per day
        per_day = {}
        for entry in entries:
            day = entry.created_at.date().isoformat()
            per_day[day] = per_day.get(day, 0) + 1
        
        # Most corrected
        corrections = {}
        for entry in entries:
            key = f"{entry.original_label}->{entry.corrected_label}"
            corrections[key] = corrections.get(key, 0) + 1
        
        most_corrected = sorted(corrections.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return cls(
            total_feedback=len(entries),
            feedback_per_day=per_day,
            most_corrected_classes=most_corrected,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_feedback": self.total_feedback,
            "feedback_per_day": self.feedback_per_day,
            "accuracy_improvement": self.accuracy_improvement,
            "most_corrected_classes": self.most_corrected_classes,
            "avg_time_to_correction": self.avg_time_to_correction,
        }
