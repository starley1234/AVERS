# АВЕРС — Автоматическая Векторизация и Распознавание Схем

**AVERS** (Automated Vectorization and Recognition of Schematics) — production-ready система автоматической векторизации и семантической оцифровки схем бортовых кабельных сетей (БКС).

**v0.2** — Web UI валидатор + Synthetic Dataset ГОСТ УГО + Vision RAG

> 🚀 **Быстрый старт:** `bash scripts/quickstart.sh` — окружение, демо и первый запуск одним скриптом.
> Краткая инструкция и проверка работоспособности (`bash scripts/smoke_test.sh`) — в [QUICKSTART.md](QUICKSTART.md).

## 🎯 Назначение

Преобразование растровых сканов принципиальных электрических схем (включая нестандартные форматы А2х6 до 15000×4000px) в **математический граф связей** (Netlist / JSON / XML), пригодный для прямой загрузки в САПР («Макс-САПР», «КОМПАС-Электрик»).

### Ключевые возможности v0.2

- ✅ **Production pipeline** — 6 стадий, stateless API, error handling, SAHI
- ✅ **Web UI валидатор** — минималистичный UI, drag&drop, zoom/pan canvas, редактирование, экспорт
- ✅ **Synthetic Dataset ГОСТ** — 10 классов УГО по ГОСТ 2.721, 2.728, 2.730, 2.755, 2.756, генерация A2x6
- ✅ **Ручная разметка** — встроенный аннотатор (/annotator), YOLO/COCO экспорт
- ✅ **RT-DETR/YOLO обучение** — YOLOv11x/m, RT-DETRv2-l/x, ГОСТ-аугментации, ONNX экспорт
- ✅ **Vision RAG** — CLIP embeddings + FAISS + few-shot VLM prompting для арбитража коллизий
- ✅ **Graph synthesis** — NetworkX, wire snapping, net extraction, k-d tree текст

## 🏗️ Архитектура

```
┌──────────────────────────────────────────────────────────────────────┐
│  Исходный скан (А2х6, TIF/PNG/PDF, 15000×4000)                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│ [ВЕТКА 1: УГО]         [ВЕТКА 2: ТЕКСТ]         [ВЕТКА 3: ЛИНИИ]      │
│ SAHI + RT-DETR/YOLO    PaddleOCR DBNet+CRNN     OpenCV скелетизация   │
│ 1024×1024 tiles 0.2    0°/90°/270° + ГОСТ regex  Guo-Hall + RDP        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Графовый синтез       │
              │   NetworkX + k-d tree   │
              │   Snapping + Nets       │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Vision RAG + VLM       │
              │  CLIP + FAISS + Qwen-VL │
              │  Few-shot арбитраж      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  NETLIST JSON/XML       │
              │  + Web UI валидатор     │
              └─────────────────────────┘
```

## 📦 Установка

```bash
git clone https://github.com/starley1234/AVERS.git
cd AVERS

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Для Web UI
pip install fastapi uvicorn python-multipart

# Для ML (опционально)
pip install ultralytics sahi paddleocr transformers torch faiss-cpu

# Или всё сразу
pip install -e ".[all]"
```

## 🚀 Быстрый старт

### CLI - обработка схемы (изображения + PDF)

```bash
# Новый CLI v0.2 - поддерживает изображения и PDF
python -m avers process input.tif --output result.json
python -m avers process board.png -o nets.xml --format xml --tile-size 1024

# PDF поддержка (NEW)
python -m avers process schema.pdf --output result.json  # Первая страница
python -m avers process schema.pdf --pdf-page 2 -o result.json  # Конкретная страница
python -m avers process schema.pdf --pdf-all -o result.json  # Все страницы, merge

# Legacy (совместимость)
python -m avers.main input.tif --output result.json
```

### Python API

```python
from avers.pipeline import load_and_process

result = load_and_process("input.tif", "output.json")
print(f"Components: {len(result.manifest.components)}")
print(f"Nets: {len(result.manifest.nets)}")
print(f"Timings: {result.stage_timings}")

# Прямой pipeline
from avers.pipeline import ProductionPipeline
from PIL import Image
import numpy as np

pipeline = ProductionPipeline()
image = np.array(Image.open("input.tif"))
result = pipeline.run(image, "input.tif", dpi=300)
```

### Web UI валидатор (NEW)

```bash
# Запуск
python -m avers web --port 8000 --host 0.0.0.0

# Откройте в браузере
# http://localhost:8000          - Валидатор
# http://localhost:8000/annotator - Аннотатор ГОСТ УГО
# http://localhost:8000/docs      - API docs
```

**Фичи Web UI:**
- Drag & drop загрузка TIF/PNG до 50MB
- Визуализация всех стадий (детекция, OCR, векторизация, граф)
- Zoom/pan canvas с overlay слоями
- Редактирование компонентов, разрешение проблем
- Экспорт JSON/XML
- Vision RAG поиск
- Минималистичный дизайн (Inter + JetBrains Mono, темная тема)

### Synthetic Dataset ГОСТ УГО + Public Datasets (NEW)

```bash
# Генерация синтетического датасета
python -m avers dataset generate --output /tmp/avers_dataset --num-train 1000 --num-val 200 --image-size 1024

# Превью
python -m avers dataset preview --output /tmp/preview --num 20

# Публичные датасеты для обучения (NEW - 8 датасетов найдено)
python -m avers dataset public list
python -m avers dataset public list --gost-only  # Только GOST-совместимые
python -m avers dataset public info --dataset masala-chai
python -m avers dataset public download --dataset masala-chai
python -m avers dataset public strategy  # Рекомендуемая стратегия обучения

# Смешанный датасет: синтетика + публичные
python -m avers dataset public mix --synthetic /tmp/gost/dataset.yaml --public masala-chai:/tmp/masala-chai,circuit-diagram-roboflow:/tmp/roboflow --output /tmp/mixed

# Или через Python
from avers.dataset.synthetic import GOSTGenerator, SyntheticConfig

config = SyntheticConfig(image_size=1024, enable_wires=True, enable_scan_effects=True)
gen = GOSTGenerator(config)
img, annotations = gen.generate_realistic_schematic()

# Большие схемы A2x6
from avers.dataset.generator import SchematicComposer
composer = SchematicComposer()
large_img, anns = composer.compose_a2x6(width=14000, height=3500)

# Public datasets
from avers.dataset.public_datasets import PublicDatasetLoader
loader = PublicDatasetLoader()
datasets = loader.list_datasets(gost_compatible_only=True)  # Masala-CHAI 4300, ElectroNet 3500
```

**10 классов ГОСТ:**
- `connector_body` — корпус разъема (ГОСТ 2.755-87)
- `pin` — контакт разъема
- `junction_dot` — точка соединения (ГОСТ 2.721-74)
- `ground` — заземление
- `shield` — экран кабеля
- `offpage_connector` — переход на другой лист
- `diode` — диод (ГОСТ 2.730-73)
- `relay` — реле (ГОСТ 2.756-76)
- `resistor` — резистор (ГОСТ 2.728-74)
- `capacitor` — конденсатор

### Ручная разметка (NEW)

```bash
# Запустите Web UI и откройте /annotator
python -m avers web --port 8000
# http://localhost:8000/annotator

# Горячие клавиши:
# 1-9 - выбор класса
# Drag - рисование бокса
# Del - удалить
# Ctrl+S - сохранить
```

Или API:

```python
from avers.dataset.annotator import get_store

store = get_store()
# После разметки в UI
store.export_yolo(Path("/tmp/export"), split_ratio=0.8)
```

### Обучение RT-DETR / YOLO (NEW)

```bash
# Обучение RT-DETR
python -m avers dataset train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l --epochs 100 --batch 8

# YOLOv11
python -m avers dataset train --data /tmp/avers_dataset/dataset.yaml --model yolo11x --epochs 100

# Через ultralytics напрямую
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
model.train(data='/tmp/avers_dataset/dataset.yaml', epochs=100, imgsz=640)
model.export(format='onnx')

# Использование в АВЕРС
# config.yaml:
# detection:
#   model_path: /tmp/avers_runs/rtdetr/weights/best.pt
#   model_type: rtdetr
```

### Vision RAG (NEW)

**Идея:** вместо чистого VLM для разрешения коллизий — retrieval augmented generation.

```
ROI (256×256) -> CLIP ViT-B/32 (512d) -> FAISS search Top-K -> few-shot prompt -> VLM
```

```python
from avers.rag import VisionRAG
import cv2

rag = VisionRAG()

# Индексация примеров из валидатора
roi = cv2.imread("junction_example.jpg")
rag.add_example(roi, label="junction_dot_connected", description="Точка соединения есть контакт")

# Запрос при коллизии
query_roi = cv2.imread("ambiguous_junction.jpg")
response = rag.query(image=query_roi, text="Is there a junction dot?", top_k=5)

print(f"Похожие: {response.results}")
print(f"VLM ответ: {response.vlm_answer}")
# {"connected": true, "confidence": 0.85, "reasoning": "Based on 5 examples..."}

# Интеграция в Stage 6 - автоматически использует RAG если enabled
```

CLI:

```bash
python -m avers rag index --image junction.jpg --label junction_dot --description "connected"
python -m avers rag query --text "junction dot" --image query.jpg --top-k 5
python -m avers rag stats
```

## 📁 Структура проекта

```
avers/
├── core/               # Базовые типы, pipeline, логгер
│   ├── types.py        # Pydantic модели (Component, Net, Manifest)
│   ├── pipeline.py     # Legacy pipeline
│   └── validators.py   # ProductionPipeline + SAHI, OCR, Vectorizer
├── stages/
│   ├── stage1_slicing/ # SAHI нарезка
│   ├── stage2_detection/ # RT-DETR/YOLO детекция УГО
│   ├── stage3_ocr/     # PaddleOCR + ГОСТ regex
│   ├── stage4_vectorization/ # Скелетизация + RDP
│   ├── stage5_graph_synthesis/ # NetworkX граф
│   └── stage6_vlm_arbitrator/ # VLM арбитраж
├── web/                # NEW: Web UI
│   ├── app.py          # FastAPI app
│   ├── api.py          # Основные API + RAG
│   ├── annotator_api.py # Аннотатор API
│   └── frontend/
│       ├── index.html  # Валидатор UI
│       └── annotator.html # Аннотатор UI
├── dataset/            # NEW: Датасет инструменты
│   ├── gost_symbols.py # Определения ГОСТ УГО
│   ├── synthetic.py    # SyntheticGenerator
│   ├── generator.py    # SchematicComposer A2x6
│   ├── annotator.py    # AnnotationStore
│   ├── export.py       # YOLO/COCO экспорт
│   ├── train.py        # Обучение RT-DETR/YOLO
│   └── cli.py          # CLI
├── rag/                # NEW: Vision RAG
│   ├── embeddings.py   # CLIP + HOG fallback
│   ├── store.py        # FAISS + brute-force
│   └── vision_rag.py   # VisionRAG + few-shot VLM
├── pipeline.py         # ProductionPipeline entry
├── config.py           # Конфигурация (YAML + Pydantic)
└── main.py             # CLI (process, web, dataset, rag)
```

## 🔧 Конфигурация

```yaml
# config.yaml v0.2
slicing:
  tile_size: 1024
  overlap_ratio: 0.2

detection:
  model_type: rtdetr    # yolo or rtdetr
  model_path: /tmp/best.pt
  confidence_threshold: 0.25
  device: cuda

ocr:
  lang: ru
  text_confidence_threshold: 0.65

vectorization:
  rdp_epsilon: 2.0

graph_synthesis:
  snap_radius: 15

vlm_arbitrator:
  enabled: true
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  max_vlm_calls: 50

vision_rag:  # NEW
  enabled: true
  embedding_model: openai/clip-vit-base-patch32
  top_k: 5
  use_few_shot: true

dataset:  # NEW
  image_size: 1024
  num_train: 1000
  classes: [connector_body, pin, junction_dot, ...]

web:  # NEW
  host: 0.0.0.0
  port: 8000
```

## 📊 Выходной формат

```json
{
  "schema_metadata": {
    "source_file": "schema.tif",
    "resolution_dpi": 300,
    "width": 14200,
    "height": 3800,
    "processing_time_seconds": 12.34
  },
  "components": [
    {
      "id": "comp_001",
      "designator": "X1",
      "type": "connector",
      "part_number": "СНЦ144-6/10РО11",
      "bbox": [1240, 500, 1480, 890],
      "pins": [
        {"pin_number": "1", "coord": [1480, 520], "confidence": 0.95}
      ],
      "confidence": 0.98
    }
  ],
  "nets": [
    {
      "net_id": "NET_PWR_27V",
      "wire_type": "БПВЛ-0.35",
      "wire_color": "К",
      "connections": [
        {"component_id": "comp_001", "pin": "1"},
        {"component_id": "comp_002", "pin": "1"}
      ],
      "path_points": [[1480, 520], [3200, 520], [8500, 520]],
      "confidence": 0.98
    }
  ],
  "human_review_required": [
    {
      "issue_type": "low_confidence_text",
      "bbox": [4320, 1100, 4450, 1180],
      "description": "Не удалось прочитать маркировку",
      "confidence": 0.45,
      "suggestions": ["3", "8"]
    }
  ]
}
```

## 🧪 Тестирование

```bash
# Все тесты
python -m pytest tests/ -v

# Только production
python -m pytest tests/test_production.py -v

# Демо
python demo_production.py
python demo_v02.py  # NEW: демо всех фич v0.2
```

## 🏭 Production Features

| Feature | Описание |
|---------|----------|
| **Stateless API** | Параллельная обработка, без сайд-эффектов |
| **Error Handling** | Каждая стадия обернута, пайплайн не падает |
| **Validation** | Pydantic валидация конфига и манифеста |
| **Stage Timings** | Трекинг времени по стадиям |
| **Fallback** | SAHI → Mock, PaddleOCR → EasyOCR → Mock, CLIP → HOG |
| **Type Safety** | Полные type hints |
| **Web UI** | FastAPI + минималистичный frontend |
| **Vision RAG** | CLIP + FAISS + few-shot VLM |
| **Synthetic Data** | ГОСТ УГО генератор для обучения |

## 📋 Требования

- Python 3.11+
- 8+ GB RAM
- CUDA GPU (опционально, для ML моделей)

## 📦 Dependencies

```txt
# Core
numpy scipy networkx pydantic opencv-python scikit-image Pillow lxml tqdm

# Web UI
fastapi uvicorn python-multipart

# ML (опционально)
ultralytics      # YOLO/RT-DETR
sahi             # SAHI
paddleocr        # OCR
transformers     # VLM + CLIP
faiss-cpu        # Vector DB
torch            # PyTorch
```

## 🚀 Roadmap

- [x] v0.1 — Production pipeline (6 стадий)
- [x] v0.2 — Web UI + Synthetic Dataset + Vision RAG
- [ ] v0.3 — Active learning loop (валидатор → дообучение)
- [ ] v0.4 — Экспорт в Макс-САПР / КОМПАС-Электрик (нативный)
- [ ] v0.5 — Multi-page схемы (A2x6 склейка)

## 📄 Лицензия

MIT

## 👥 Авторы

AVERS Development Team
