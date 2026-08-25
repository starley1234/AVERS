"""
АВЕРС — Автоматическая Векторизация и Распознавание Схем
Automated Vectorization and Recognition of Schematics ( бортовых кабельных сетей )
"""

__version__ = "0.1.0"
__author__ = "AVERS Development Team"

from avers.core.pipeline import aversPipeline
from avers.core.types import (
    Component,
    Pin,
    Net,
    WireConnection,
    SchemaMetadata,
    AVERSManifest,
    HumanReviewIssue,
)

__all__ = [
    "aversPipeline",
    "Component",
    "Pin",
    "Net",
    "WireConnection",
    "SchemaMetadata",
    "AVERSManifest",
    "HumanReviewIssue",
]
