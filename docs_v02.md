# АВЕРС v0.2 — Документация по новым модулям

## Обзор

v0.2 добавляет три ключевых компонента к базовому пайплайну v0.1:

1. **Web UI валидатор** — минималистичный интерфейс для проверки и редактирования результатов
2. **Synthetic Dataset + Обучение** — генерация ГОСТ УГО и обучение RT-DETR/YOLO
3. **Vision RAG** — retrieval augmented generation для VLM-арбитража

---

## 1. Web UI валидатор

### Архитектура

```
Frontend (vanilla JS + Canvas)
  ├── Загрузка (drag&drop, 50MB)
  ├── Canvas (zoom/pan, SVG overlay)
  ├── Слои (detections, text, wires, pins, issues)
  ├── Pipeline (5 стадий, прогресс, тайминги)
  └── Tabs (components, nets, issues, JSON)

Backend (FastAPI)
  ├── /api/upload - загрузка
  ├── /api/process/{id} - запуск пайплайна (background)
  ├── /api/status/{job_id} - статус
  ├── /api/result/{id} - манифест
  ├── /api/visualization/{id}/{stage} - визуализации
  ├── /api/components/{id}/{comp_id} - редактирование
  ├── /api/issues/{id}/{idx}/resolve - разрешение проблем
  ├── /api/export/{id}?format=json|xml
  ├── /api/rag/* - Vision RAG
  └── /api/annotator/* - Аннотатор
```

### Принципы UI (минимализм)

- **Темная тема**: --bg #0a0a0b, --bg2 #141416, акцент #6c5cff
- **Типографика**: Inter для UI, JetBrains Mono для кода/координат
- **Иерархия**: header 56px, sidebar 320px, canvas flex, statusbar 28px
- **Интерактив**: hover states, active stages, progress bar 2px
- **Canvas**: grab cursor, wheel zoom, SVG overlay, layers toggle
- **Empty states**: иконка + текст подсказки
- **Модалки**: backdrop blur, 16px radius, тень 24px

### Запуск

```bash
python -m avers web --host 0.0.0.0 --port 8000 --reload
# http://localhost:8000
# http://localhost:8000/annotator
# http://localhost:8000/docs
```

### API примеры

```bash
# Upload
curl -X POST -F "file=@schema.tif" http://localhost:8000/api/upload

# Process
curl -X POST http://localhost:8000/api/process/{file_id} -H "Content-Type: application/json" -d '{"config_overrides": {"slicing": {"tile_size": 1024}}}'

# Status
curl http://localhost:8000/api/status/{job_id}

# Result
curl http://localhost:8000/api/result/{file_id}

# Export
curl http://localhost:8000/api/export/{file_id}?format=json -o result.json
```

---

## 2. Synthetic Dataset ГОСТ УГО

### ГОСТ символы

Реализованы по ГОСТ 2.721-74, 2.728-74, 2.730-73, 2.755-87, 2.756-76:

| ID | Класс | ГОСТ | Описание | Размер |
|----|-------|------|----------|--------|
| 0 | connector_body | 2.755-87 | Корпус разъема | 40-120×60-200 |
| 1 | pin | 2.755-87 | Контакт | 6-16×6-16 |
| 2 | junction_dot | 2.721-74 | Точка соединения | 4-12×4-12 |
| 3 | ground | 2.721-74 | Земля | 20-40×20-40 |
| 4 | shield | 2.721-74 | Экран | 30-80×20-50 |
| 5 | offpage_connector | 2.721-74 | Переход листа | 20-50×15-30 |
| 6 | diode | 2.730-73 | Диод | 20-40×12-24 |
| 7 | relay | 2.756-76 | Реле | 40-80×20-40 |
| 8 | resistor | 2.728-74 | Резистор | 30-60×10-20 |
| 9 | capacitor | 2.728-74 | Конденсатор | 20-30×10-20 |

### Генерация

```python
from avers.dataset.synthetic import GOSTGenerator, SyntheticConfig

config = SyntheticConfig(
    image_size=1024,
    min_objects=8,
    max_objects=30,
    classes=["connector_body", "pin", "junction_dot", "ground", "diode", "resistor"],
    enable_wires=True,
    enable_noise=True,
    enable_scan_effects=True,
    enable_blur=True,
)

gen = GOSTGenerator(config)

# Простое изображение
img, anns = gen.generate_image()

# Реалистичная схема БКС (разъемы слева/справа, провода)
img, anns = gen.generate_realistic_schematic()

# Датасет YOLO
gen.generate_dataset(num_images=1000, output_dir="/tmp/avers_dataset", split="train")
```

### Большие схемы A2x6

```python
from avers.dataset.generator import SchematicComposer

composer = SchematicComposer()
img, anns = composer.compose_a2x6(width=14000, height=3500)  # А2х6
# Или батч
composer.generate_batch(num_images=100, output_dir="/tmp/batch", image_size=(1024,1024))
```

### Аугментации для реализма

- **Шум**: Gaussian noise 0.05
- **Размытие**: Gaussian blur 3×3 (имитация скана)
- **Скан эффекты**: градиент освещения, виньетка
- **Поворот**: ±2° (схемы обычно прямые)
- **Провода**: ортогональная трассировка, шины, T-junctions

### Экспорт

```python
from avers.dataset.export import YOLOExporter, COCOExporter

# YOLO (создается автоматически)
# images/train/*.jpg
# labels/train/*.txt
# dataset.yaml

# COCO
COCOExporter.from_yolo_dataset(Path("/tmp/avers_dataset"), Path("/tmp/coco.json"))
```

### CLI

```bash
python -m avers dataset generate --output /tmp/avers_dataset --num-train 1000 --num-val 200 --num-test 100 --image-size 1024
python -m avers dataset preview --output /tmp/preview --num 20 --size 1024
python -m avers dataset export --input /tmp/avers_dataset --output /tmp/coco.json --format coco
```

---

## 3. Инструмент ручной разметки

### Web аннотатор

Откройте `/annotator` после запуска `avers web`:

- **Загрузка**: кнопка или drag&drop
- **Рисование**: выберите класс (1-9) → drag по изображению
- **Редактирование**: клик по боксу → Del для удаления
- **Сохранение**: Ctrl+S или кнопка
- **Экспорт**: кнопка "Экспорт YOLO" → `/tmp/avers_dataset_export/`

### API

```bash
# Upload для разметки
curl -X POST -F "file=@img.jpg" http://localhost:8000/api/annotator/upload

# List
curl http://localhost:8000/api/annotator/images

# Update annotations
curl -X PUT http://localhost:8000/api/annotator/images/{id}/annotations -H "Content-Type: application/json" -d '{"annotations": [{"class_id": 0, "class_name": "connector_body", "bbox": [10,20,100,200]}]}'

# Export
curl -X POST http://localhost:8000/api/annotator/export -H "Content-Type: application/json" -d '{"output_dir": "/tmp/export", "split_ratio": 0.8}'
```

### Python API

```python
from avers.dataset.annotator import get_store
from pathlib import Path

store = get_store(Path("/tmp/avers_dataset"))

# List
images = store.list_images()

# Update
store.update_annotations(image_id, [{"class_id": 0, "bbox": (10,20,100,200)}])

# Export YOLO with train/val split
yaml_path = store.export_yolo(Path("/tmp/export"), split_ratio=0.8)
```

---

## 4. Обучение RT-DETR / YOLO

### Поддерживаемые модели

- **YOLOv11x** — максимальная точность, медленнее
- **YOLOv11m** — баланс
- **RT-DETRv2-l** — рекомендован для ГОСТ (трансформер, лучше на мелких объектах)
- **RT-DETRv2-x** — максимальная точность

### ГОСТ-специфичные аугментации

```python
{
    "hsv_h": 0.015,      # Малый цветовой сдвиг (схемы ЧБ)
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "degrees": 2.0,      # Малый поворот
    "translate": 0.05,
    "scale": 0.1,
    "shear": 1.0,
    "perspective": 0.0,  # Нет перспективы
    "flipud": 0.0,       # Не флипаем (текст)
    "fliplr": 0.0,
    "mosaic": 0.5,       # Mosaic аугментация
}
```

### Обучение

```bash
# RT-DETR
python -m avers dataset train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l --epochs 100 --batch 8 --imgsz 640 --device cuda

# YOLO
python -m avers dataset train --data /tmp/avers_dataset/dataset.yaml --model yolo11x --epochs 100 --batch 8

# Через Python
from avers.dataset.train import train_rtdetr, train_yolo
from pathlib import Path

train_rtdetr(
    data_yaml=Path("/tmp/avers_dataset/dataset.yaml"),
    model_name="rtdetr-l.pt",
    epochs=100,
    batch=8,
    imgsz=640,
    device="cuda",
    project="/tmp/avers_runs"
)
```

### Валидация и экспорт

```python
from avers.dataset.train import validate_model

metrics = validate_model(
    model_path=Path("/tmp/avers_runs/rtdetr/weights/best.pt"),
    data_yaml=Path("/tmp/avers_dataset/dataset.yaml"),
    model_type="rtdetr"
)
print(metrics)

# Экспорт в ONNX для продакшена
from ultralytics import RTDETR
model = RTDETR("/tmp/avers_runs/rtdetr/weights/best.pt")
model.export(format="onnx", dynamic=True)
```

### Использование в АВЕРС

```yaml
# config.yaml
detection:
  model_path: /tmp/avers_runs/rtdetr/weights/best.pt
  model_type: rtdetr
  confidence_threshold: 0.25
  device: cuda
```

---

## 5. Vision RAG

### Архитектура

```
Индексация:
  ROI (256×256) ──> CLIP ViT-B/32 (512d) ──> FAISS Index ──> Vector DB
  + label, bbox, description, metadata

Запрос (при коллизии в Stage 6):
  ROI + текст ──> CLIP embedding ──> FAISS search Top-K (5) ──> few-shot prompt ──> Qwen-VL ──> JSON ответ
  {"connected": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
```

### Компоненты

1. **Embeddings** (`avers/rag/embeddings.py`):
   - `CLIPEmbedding` — CLIP ViT-B/32, 512d, normalized
   - Fallback: HOG + color histogram (если transformers нет)
   - `MultiModalEmbedding` — объединяет image + text (weight 0.7)

2. **Vector Store** (`avers/rag/store.py`):
   - `FAISSStore` — FAISS IndexFlatIP (cosine), fallback brute-force
   - Save/load на диск (embeddings.npy + meta.json)
   - Stats, delete, clear

3. **VisionRAG** (`avers/rag/vision_rag.py`):
   - `add_example(image, label, bbox, description, metadata)`
   - `query(image, text, top_k, use_vlm)`
   - Few-shot prompt building
   - Mock VLM voting (если нет реального VLM)
   - Save/load

### Использование

```python
from avers.rag import VisionRAG
import cv2

rag = VisionRAG(storage_path="/tmp/avers_rag")

# Индексация из валидатора (human-in-the-loop)
roi = cv2.imread("junction.jpg")
rag.add_example(
    image=roi,
    label="junction_dot_connected",
    bbox=(100,100,200,200),
    description="Точка соединения - есть контакт, черный круг 6px",
    metadata={"source": "validator", "user": "engineer1"}
)

# Запрос при коллизии
query_roi = cv2.imread("ambiguous.jpg")
response = rag.query(
    image=query_roi,
    text="Is there a junction dot at the center? Connected or crossing?",
    top_k=5,
    use_vlm=True
)

print(f"Найдено {len(response.results)} примеров:")
for r in response.results:
    print(f"  {r.entry.label} score={r.score:.3f}")

print(f"VLM ответ: {response.vlm_answer}")
# {
#   "connected": True,
#   "confidence": 0.85,
#   "reasoning": "Based on 5 examples, 4 indicate connection with dot",
#   "examples_used": 5
# }

# Сохранение
rag.save()
stats = rag.stats()
# {"total": 100, "labels": {"junction_dot_connected": 40, ...}}
```

### Интеграция в Stage 6

```python
# avers/stages/stage6_vlm_arbitrator/arbitrator.py (идея)

from avers.rag import get_rag

rag = get_rag()

# При коллизии
issue_roi = extract_roi(image, bbox, size=256)

# RAG query
rag_response = rag.query(
    image=issue_roi,
    text="Is this junction connected? Look for dot at center",
    top_k=5,
    use_vlm=True
)

# Используем VLM ответ с учетом примеров
if rag_response.vlm_answer["confidence"] > 0.7:
    connected = rag_response.vlm_answer["connected"]
else:
    # Fallback to pure VLM
    connected = vlm.query(issue_roi, prompt)["connected"]

# Human-in-the-loop: если пользователь исправил в валидаторе, добавляем в RAG
if user_corrected:
    rag.add_example(issue_roi, label="junction_dot_connected" if connected else "junction_dot_none", description=user_comment)
```

### CLI

```bash
python -m avers rag index --image junction.jpg --label junction_dot --description "connected junction with dot"
python -m avers rag query --text "junction dot" --image query.jpg --top-k 5
python -m avers rag stats

# Или через API
curl -X POST http://localhost:8000/api/rag/index -H "Content-Type: application/json" -d '{"image_base64": "...", "bbox": [0,0,256,256], "label": "junction_dot", "description": "connected"}'
curl -X POST http://localhost:8000/api/rag/query -H "Content-Type: application/json" -d '{"text_query": "junction dot", "top_k": 5, "use_vlm": true}'
```

### Преимущества RAG для АВЕРС

1. **Точность**: VLM видит похожие примеры → меньше галлюцинаций
2. **Адаптивность**: база пополняется из валидатора (human feedback)
3. **Объяснимость**: можно показать пользователю, на основе каких примеров принято решение
4. **Эффективность**: Top-K поиск быстрее, чем полный перебор, и дешевле, чем большой VLM
5. **Домейн-специфика**: база ГОСТ символов специфична для БКС, не generic

---

## 6. Полный пайплайн v0.2

```python
from avers.pipeline import ProductionPipeline
from avers.config import AVERSConfig
from avers.rag import VisionRAG

# Конфиг с RAG
config = AVERSConfig()
config.vision_rag.enabled = True
config.vision_rag.top_k = 5
config.vision_rag.use_few_shot = True

# RAG с предзаполненной базой
rag = VisionRAG()
rag.load("/tmp/avers_rag")  # Загрузить базу

# Pipeline
pipeline = ProductionPipeline(config)

# Обработка
result = pipeline.run(image, "schema.tif", dpi=300)

# Результат + RAG примеры
print(f"Components: {len(result.manifest.components)}")
print(f"Nets: {len(result.manifest.nets)}")
print(f"Issues: {len(result.manifest.human_review_required)}")

# Если есть проблемы, показать в Web UI для валидации
# Пользователь исправляет → добавляется в RAG → улучшает будущие результаты
```

---

## 7. Демо

```bash
python demo_v02.py  # Полный демо всех фич v0.2

# Вывод:
# 1. Synthetic ГОСТ dataset
# 2. Большие схемы A2x6
# 3. Vision RAG индексация и запрос
# 4. Конфиг обучения
# 5. Web UI
# 6. Полный пайплайн с RAG
```

---

## 8. Roadmap

- [x] v0.1 — Production pipeline (6 стадий)
- [x] v0.2 — Web UI + Synthetic Dataset + Vision RAG
- [ ] v0.3 — Active learning loop (валидатор → дообучение автоматически)
- [ ] v0.4 — Нативный экспорт в Макс-САПР / КОМПАС-Электрик
- [ ] v0.5 — Multi-page схемы (склейка A2x6)
- [ ] v0.6 — Qdrant / Milvus для RAG в продакшене
- [ ] v0.7 — Дистрибутивное обучение на кластере

---

## 9. Зависимости

```bash
# Core (обязательно)
pip install numpy scipy networkx pydantic PyYAML opencv-python scikit-image Pillow lxml tqdm

# Web UI
pip install fastapi uvicorn python-multipart

# ML (опционально)
pip install ultralytics sahi paddleocr transformers torch faiss-cpu

# Все
pip install -e ".[all]"
```
