# АВЕРС — Автоматическая Векторизация и Распознавание Схем

**AVERS** (Automated Vectorization and Recognition of Schematics) — production-ready система автоматической векторизации и семантической оцифровки схем бортовых кабельных сетей (БКС).

**v0.2** — Web UI валидатор + Synthetic Dataset ГОСТ УГО + Vision RAG

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
# Запуск (порт по умолчанию — 8030)
python -m avers web

# Откройте в браузере
# http://localhost:8030          - Валидатор
# http://localhost:8030/annotator - Аннотатор ГОСТ УГО
# http://localhost:8030/docs      - API docs
```

В Windows из корня проекта можно запустить `run_web.cmd` — он использует
локальное окружение `venv\Scripts\python.exe` и порт 8030.

**Фичи Web UI:**

Без обученных весов детектора (`detection.model_path`) и OCR-бэкенда интерфейс
показывает предупреждение: он не присваивает контурам вымышленные классы УГО
или маркировки. Ограничения также сохраняются в `processing_warnings` при
экспорте JSON/XML. Для распознавания реальных схем нужны обученная модель и
установленный OCR (см. раздел «Установка»). Web UI читает `config.yaml`
(или путь из `AVERS_CONFIG`); укажите там `detection.model_path` к весам
модели, обученной на соответствующих классах УГО.

- Drag & drop загрузка TIF/PNG до 50MB
- Визуализация всех стадий (детекция, OCR, векторизация, граф)
- Zoom/pan canvas с overlay слоями
- Редактирование компонентов, разрешение проблем
- Экспорт JSON/XML
- Vision RAG поиск
- Минималистичный дизайн (Inter + JetBrains Mono, темная тема)

Для проверки геометрии проводов можно загрузить `data/demo_connections.png`
([ожидаемая топология](data/demo_connections.md)): верхняя Т-образная ветка
и нижняя Г-образная ветка должны образовать две отдельные цепи. Это тест **проводов**, не автоматического распознавания УГО:
без обученной модели контакты компонентов не определяются. Измерения на
реальных сканах и известные ограничения: [docs/REAL_SCAN_SMOKE_TEST.md](docs/REAL_SCAN_SMOKE_TEST.md).

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
python -m avers web --port 8030
# http://localhost:8030/annotator

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

### Active Learning Loop v0.3 (NEW) — валидатор → дообучение

Замыкает цикл: исправления в валидаторе накапливаются → при достижении
порога сливаются с синтетическим ГОСТ-датасетом → дообучают модель →
новая версия регистрируется с возможностью promote/rollback/A-B теста.
Полное описание и план доработки на GPU: [`docs/ACTIVE_LEARNING_V03.md`](docs/ACTIVE_LEARNING_V03.md).

```bash
# Статистика накопленного feedback
python -m avers active-learning stats

# Смержить synthetic + feedback в один dataset.yaml (с ремапом классов и oversampling)
python -m avers active-learning merge-datasets --synthetic /tmp/gost/dataset.yaml \
    --feedback /tmp/fb/dataset_feedback.yaml --feedback-weight 3 -o /tmp/merged

# Дообучить на merged датасете и зарегистрировать версию
python -m avers active-learning retrain --synthetic /tmp/gost/dataset.yaml \
    --model rtdetr-l --epochs 20 --register

# Фоновый планировщик автодообучения (blocking, для cron/systemd)
python -m avers active-learning scheduler run --interval 3600

# Model Registry: версии, promote/rollback, A/B
python -m avers active-learning registry list
python -m avers active-learning registry promote v000002
python -m avers active-learning registry ab-test v000002 --ratio 0.2

# Web UI: валидатор автоматически собирает feedback, дашборд мониторинга
AVERS_AL_SCHEDULER_ENABLED=true python -m avers web --port 8030
# http://localhost:8030/dashboard
```

Каталог реальных (не синтетических) советских/российских схем для обучения
(от простых к сложным): [`docs/REAL_SCHEMATICS_SOURCES.md`](docs/REAL_SCHEMATICS_SOURCES.md),
куратированный пример уже в [`data/reference_schematics/`](data/reference_schematics/README.md).

```bash
python -m avers dataset real-schematics list --tier tier1_simple_car
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
├── web/                # Web UI
│   ├── app.py          # FastAPI app (+ lifespan: auto-start AL scheduler)
│   ├── api.py          # Основные API + RAG + active-learning stats/retrain/clear
│   ├── active_learning_api.py # NEW v0.3: scheduler/registry/history/notifications
│   ├── annotator_api.py # Аннотатор API
│   └── frontend/
│       ├── index.html  # Валидатор UI
│       ├── annotator.html # Аннотатор UI
│       └── dashboard.html # NEW v0.3: Active Learning dashboard (/dashboard)
├── dataset/            # Датасет инструменты
│   ├── gost_symbols.py # Определения ГОСТ УГО
│   ├── synthetic.py    # SyntheticGenerator
│   ├── generator.py    # SchematicComposer A2x6
│   ├── annotator.py    # AnnotationStore
│   ├── export.py       # YOLO/COCO экспорт
│   ├── train.py        # Обучение RT-DETR/YOLO
│   ├── public_datasets.py # Публичные датасеты (pre-training)
│   ├── merge.py        # NEW v0.3: merge synthetic+feedback(+public) датасетов
│   ├── real_schematics.py # NEW v0.3: каталог реальных RU/СССР источников
│   └── cli.py          # CLI
├── active_learning/    # v0.3: цикл валидатор -> дообучение
│   ├── loop.py          # ActiveLearningLoop, FeedbackEntry
│   ├── metrics.py        # ActiveLearningMetrics
│   ├── registry.py       # NEW: ModelRegistry (версии, promote/rollback, A/B)
│   ├── scheduler.py       # NEW: RetrainScheduler (фон/cron)
│   └── notify.py          # NEW: Notifier (webhook/log/JSONL)
├── rag/                # Vision RAG
│   ├── embeddings.py   # CLIP + HOG fallback
│   ├── store.py        # FAISS + brute-force
│   └── vision_rag.py   # VisionRAG + few-shot VLM
├── pipeline.py         # ProductionPipeline entry
├── config.py           # Конфигурация (YAML + Pydantic, + active_learning section)
└── main.py             # CLI (process, web, dataset, rag, active-learning)

docs/
├── ACTIVE_LEARNING_V03.md      # NEW: статус v0.3 + план для GPU-машины
└── REAL_SCHEMATICS_SOURCES.md  # NEW: каталог реальных RU/СССР схем (simple->complex)

data/reference_schematics/      # NEW: куратированный пример реальных схем (12 файлов)
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
  port: 8030

active_learning:  # NEW v0.3
  enabled: true
  min_feedback_for_retrain: 50
  scheduler_enabled: false       # true -> auto-start RetrainScheduler with `avers web`
  registry_dir: /tmp/avers_model_registry
  base_synthetic_dataset_yaml: /tmp/avers_gost/dataset.yaml
  notify_webhook_url: null
```
Полный список опций (scheduler, registry, merge, notifications) — в `config.yaml` и `docs/ACTIVE_LEARNING_V03.md`.

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
- [~] v0.3 — Active learning loop (валидатор → дообучение) — **вся CPU-логика готова и протестирована** (scheduler, model registry/A-B, notifications, dataset merge, dashboard); обучение на реальном GPU — следующий шаг. План: [`docs/ACTIVE_LEARNING_V03.md`](docs/ACTIVE_LEARNING_V03.md)
- [ ] v0.4 — Экспорт в Макс-САПР / КОМПАС-Электрик (нативный)
- [ ] v0.5 — Multi-page схемы (A2x6 склейка)

Полная дорожная карта с чеклистами: [`ROADMAP.md`](ROADMAP.md).

## 🧑‍💻 Для агентов/разработчиков

- **Начните с [`AGENTS.md`](AGENTS.md)** — как быстро поднять окружение, какие
  есть известные ловушки (opencv, отсутствие GPU в песочнице), как не тратить
  токены впустую.
- Работа над v0.3 Active Learning Loop: [`docs/ACTIVE_LEARNING_V03.md`](docs/ACTIVE_LEARNING_V03.md)
  (что готово, что осталось сделать на GPU-машине, пошаговый план).
- Материал для обучения на реальных (не синтетических) схемах:
  [`docs/REAL_SCHEMATICS_SOURCES.md`](docs/REAL_SCHEMATICS_SOURCES.md) +
  куратированный пример в [`data/reference_schematics/`](data/reference_schematics/README.md).

## 📄 Лицензия

MIT

## 👥 Авторы

AVERS Development Team
