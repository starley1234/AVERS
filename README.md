# АВЕРС — Автоматическая Векторизация и Распознавание Схем

**AVERS** (Automated Vectorization and Recognition of Schematics) — production-ready система автоматической векторизации и семантической оцифровки схем бортовых кабельных сетей (БКС).

## 🎯 Назначение

Преобразование растровых сканов принципиальных электрических схем в **математический граф связей** (Netlist / JSON / XML), пригодный для прямой загрузки в САПР («Макс-САПР», «КОМПАС-Электрик»).

### Ключевые возможности

- ✅ **Production-ready pipeline** — stateless API, error handling, validation
- ✅ **SAHI интеграция** — нарезка на тайлы 1024×1024 с перекрытием
- ✅ **ML-ready** — RT-DETR/YOLO, PaddleOCR, Qwen-VL с fallback
- ✅ **Обработка больших форматов** — А2х6 (до 15000×4000 px)
- ✅ **Graph synthesis** — NetworkX, wire snapping, net extraction
- ✅ **Экспорт в САПР** — JSON/XML формат

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│  Исходный скан (А2х6, TIF/PNG/PDF)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  SAHI SlicedDetector │
              │  (RT-DETR / YOLO)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    SchematicOCR      │
              │  (PaddleOCR/EasyOCR) │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   WireVectorizer     │
              │   (OpenCV + RDP)     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  ProductionPipeline  │
              │  GraphBuilder + k-d  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  VLMWrapper          │
              │  (Qwen/Gemma-VL)     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  NETLIST JSON/XML    │
              └─────────────────────┘
```

## 📦 Установка

```bash
git clone https://github.com/starley1234/AVERS.git
cd AVERS

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Быстрый старт

```python
from avers.pipeline import load_and_process

# Load and process image
result = load_and_process("input.tif", "output.json")

print(f"Components: {len(result.manifest.components)}")
print(f"Nets: {len(result.manifest.nets)}")
print(f"Errors: {result.errors}")
print(f"Timings: {result.stage_timings}")
```

```bash
# CLI
python -m avers.main input.tif --output result.json

# Demo
python demo_production.py
```

## 📁 API

### Production Pipeline

```python
from avers.pipeline import ProductionPipeline, PipelineResult
from PIL import Image
import numpy as np

# Method 1: Load and process
result = load_and_process("image.tif", "output.json")

# Method 2: Direct pipeline
pipeline = ProductionPipeline()
image = np.array(Image.open("image.tif"))
result = pipeline.run(image, "image.tif", dpi=300)

# PipelineResult
result.manifest      # AVERSManifest with results
result.errors        # List of errors
result.warnings      # List of warnings
result.stage_timings  # Dict of stage -> time
result.success       # True if no errors
```

### Legacy Pipeline (backwards compatible)

```python
from avers import process_schematic

manifest = process_schematic("input.tif", "output.json", production=False)
```

## 🔧 Конфигурация

```yaml
# config.yaml
slicing:
  tile_size: 1024
  overlap_ratio: 0.2

detection:
  model_type: yolo      # 'yolo' or 'rtdetr'
  confidence_threshold: 0.25
  device: cuda          # 'cuda', 'cpu', 'mps'

ocr:
  lang: ru
  text_confidence_threshold: 0.65

vectorization:
  rdp_epsilon: 2.0

graph_synthesis:
  snap_enabled: true
  snap_radius: 15

vlm_arbitrator:
  enabled: true
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
```

## 📊 Выходной формат

```json
{
  "schema_metadata": {
    "source_file": "schema.tif",
    "resolution_dpi": 300,
    "width": 14200,
    "height": 3800
  },
  "components": [
    {
      "id": "comp_001",
      "designator": "X1",
      "type": "connector",
      "bbox": [1240, 500, 1480, 890],
      "pins": [
        {"pin_number": "1", "coord": [1480, 520]}
      ]
    }
  ],
  "nets": [
    {
      "net_id": "NET_PWR_27V",
      "connections": [
        {"component_id": "comp_001", "pin": "1"}
      ],
      "path_points": [[1480, 520], [3200, 520]],
      "confidence": 0.98
    }
  ],
  "human_review_required": []
}
```

## 🧪 Тестирование

```bash
# All tests (57 passing)
python -m pytest tests/ -v

# Production tests only
python -m pytest tests/test_production.py -v

# With coverage
python -m pytest tests/ --cov=avers --cov-report=html
```

## 🏭 Production Features

| Feature | Description |
|---------|-------------|
| **Stateless API** | No side effects, run multiple images in parallel |
| **Error Handling** | Each stage wrapped, pipeline continues on failure |
| **Validation** | Config validation at init, manifest validation |
| **Stage Timings** | Per-stage performance tracking |
| **Fallback** | SAHI → Mock, PaddleOCR → EasyOCR → Mock |
| **Type Safety** | Pydantic models, full type hints |

## 📋 Требования

- Python 3.11+
- 8+ GB RAM
- CUDA GPU (optional, for ML models)

## 📦 Dependencies

```txt
# Core
numpy scipy networkx pydantic opencv-python scikit-image Pillow lxml

# Optional ML (install separately)
ultralytics      # YOLO/RT-DETR
sahi             # SAHI
paddleocr        # OCR
transformers     # VLM
```

## 📄 Лицензия

MIT

## 👥 Авторы

AVERS Development Team
