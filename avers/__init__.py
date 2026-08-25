"""
АВЕРС — Автоматическая Векторизация и Распознавание Схем
Automated Vectorization and Recognition of Schematics (бортовых кабельных сетей)

Модули:
  - core: базовые типы и пайплайн
  - stages: 6 стадий обработки (SAHI, RT-DETR, OCR, Vectorization, Graph, VLM)
  - web: Web UI валидатор + API
  - dataset: синтетическая генерация ГОСТ УГО + обучение
  - rag: Vision RAG для VLM-арбитража
"""

__version__ = "0.2.0"
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
from avers.pipeline import ProductionPipeline, PipelineResult

__all__ = [
    "aversPipeline",
    "ProductionPipeline",
    "PipelineResult",
    "Component",
    "Pin",
    "Net",
    "WireConnection",
    "SchemaMetadata",
    "AVERSManifest",
    "HumanReviewIssue",
]

