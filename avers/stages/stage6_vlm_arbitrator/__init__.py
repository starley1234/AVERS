"""Stage 6: VLM Arbitration module."""

from avers.stages.stage6_vlm_arbitrator.arbitrator import (
    VLMWrapper,
    VLMConfig,
    ArbitrationEngine,
    ArbitrationResult,
    JunctionIssue,
    SYSTEM_PROMPT,
    JUNCTION_PROMPT,
    CROSSING_PROMPT,
    TEXT_PROMPT,
    create_issues_from_detections,
)

__all__ = [
    "VLMWrapper",
    "VLMConfig",
    "ArbitrationEngine",
    "ArbitrationResult",
    "JunctionIssue",
    "SYSTEM_PROMPT",
    "JUNCTION_PROMPT",
    "CROSSING_PROMPT",
    "TEXT_PROMPT",
    "create_issues_from_detections",
]
