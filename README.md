# АВЕРС — Автоматическая Векторизация и Распознавание Схем

**AVERS** (Automated Vectorization and Recognition of Schematics) — система автоматической векторизации и семантической оцифровки схем бортовых кабельных сетей (БКС).

## 🎯 Назначение

Преобразование растровых сканов принципиальных электрических схем в **математический граф связей** (Netlist / JSON / XML), пригодный для прямой загрузки в САПР («Макс-САПР», «КОМПАС-Электрик»).

### Ключевые возможности

- ✅ **Многостадийный гибридный пайплайн**: Детекция УГО + OCR + Computer Vision + Графовый синтез
- ✅ **Обработка больших форматов**: А2х6 (до 15000×4000 px) при 300+ DPI
- ✅ **SAHI-нарезка**: Сохранение детализации при параллельной обработке
- ✅ **Точечный VLM-арбитраж**: Разрешение коллизий только для неопределённых участков
- ✅ **Экспорт в САПР**: JSON/XML формат, совместимый с российскими САПР

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│  Исходный скан (А2х6, TIF/PNG/PDF)                          │
│  Разрешение: 300+ DPI, ~14000x3500 px                        │
└────────────────────────┬────────────────────────────────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
│  УГО    │      │    OCR      │     │   Линии     │
│ SAHI +  │      │  PaddleOCR  │     │  OpenCV     │
│ RT-DETR │      │  DBNet+CRNN │     │ Скелетиз.   │
└────┬────┘      └──────┬──────┘     └──────┬──────┘
     │                   │                   │
     └───────────────────┼───────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Графовый синтез     │
              │  NetworkX + k-d tree │
              │  Snapping + Nets     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  VLM Арбитраж       │
              │  (точечные ROI)     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  NETLIST JSON/XML   │
              │  Экспорт в САПР     │
              └─────────────────────┘
```

## 📦 Установка

```bash
# Клонирование репозитория
git clone https://github.com/starley1234/AVERS.git
cd AVERS

# Виртуальное окружение
python -m venv venv && source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

## 🚀 Быстрый старт

```bash
# Обработка схемы
python -m avers.main input.tif --output result.json

# С XML экспортом
python -m avers.main board.png -o nets.xml --format xml

# С визуализацией
python -m avers.main schematic.tif --visualize

# Демо без реальных данных
python demo.py
```

## 📁 Использование в Python

```python
from avers import process_schematic, aversPipeline
from avers.config import AVERSConfig

# Быстрый запуск
manifest = process_schematic("input.tif", "output.json")

# Или через pipeline с контролем
pipeline = aversPipeline()
manifest = pipeline.process("input.tif", "output.json")

# Доступ к данным
for component in manifest.components:
    print(f"{component.designator}: {component.type}")

for net in manifest.nets:
    print(f"{net.net_id}: {len(net.connections)} connections")

# Сохранение вручную
manifest.save("output.xml", format="xml")
```

## 🔧 Стадии обработки

| Стадия | Модуль | Описание | Статус |
|--------|--------|----------|--------|
| 1 | SAHI Slicing | Нарезка на тайлы 1024×1024 с перекрытием | ✅ |
| 2 | УГО Detection | RT-DETR / YOLO детекция компонентов | ✅ |
| 3 | OCR | PaddleOCR распознавание текста | ✅ |
| 4 | Vectorization | OpenCV скелетизация и векторизация линий | ✅ |
| 5 | Graph Synthesis | NetworkX сборка графа, выделение цепей | ✅ |
| 6 | VLM Arbitration | Разрешение коллизий через Gemma/Qwen-VL | ✅ |

### Детали реализации

#### Stage 1: SAHI Slicing
- Нарезка изображений на перекрывающиеся тайлы
- Проекция координат детекций обратно в глобальное пространство
- NMS-дедупликация

#### Stage 2: УГО Detection
- YOLODetector с поддержкой RT-DETR/YOLO
- SlicingDetector для параллельного инференса
- Группировка детекций в семантические компоненты
- Mock-режим для тестирования

#### Stage 3: OCR
- PaddleOCR с fallback на EasyOCR
- Классификация текста (коннекторы, пины, типы проводов, напряжения)
- Regex-валидация по ГОСТ
- TextAssociationEngine с k-d tree

#### Stage 4: Wire Vectorization
- OpenCV скелетизация (Zhang-Suen)
- Hough Transform + contour tracing
- Ramer-Douglas-Peucker упрощение
- Детекция T/X junction points

#### Stage 5: Graph Synthesis
- NetworkX MultiGraph
- Wire-to-pin snapping (k-d tree, R=15px)
- Объединение коллинеарных сегментов
- Net extraction из connected components

#### Stage 6: VLM Arbitration
- VLMWrapper для Qwen2.5-VL / Gemma-4-VIT
- ROI extraction (256×256)
- Mock-режим без GPU

## ⚙️ Конфигурация

```yaml
# config.yaml
project_name: AVERS

slicing:
  tile_size: 1024
  overlap_ratio: 0.2

detection:
  model_type: yolo
  confidence_threshold: 0.25
  device: cuda

ocr:
  enabled: true
  lang: ru
  text_confidence_threshold: 0.65

vectorization:
  enabled: true
  skeletonize_method: zhang_suen
  rdp_epsilon: 2.0

graph_synthesis:
  snap_enabled: true
  snap_radius: 15

vlm_arbitrator:
  enabled: true
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  roi_size: 256
```

## 📊 Формат выходных данных

```json
{
  "schema_metadata": {
    "source_file": "schema_board_A2x6.tif",
    "resolution_dpi": 300,
    "width": 14200,
    "height": 3800
  },
  "components": [
    {
      "id": "comp_001",
      "designator": "X1",
      "type": "connector",
      "part_number": "СНЦ144-6/10РО11",
      "bbox": [1240, 500, 1480, 890],
      "pins": [
        {"pin_number": "1", "coord": [1480, 520], "confidence": 1.0},
        {"pin_number": "2", "coord": [1480, 560], "confidence": 1.0}
      ]
    }
  ],
  "nets": [
    {
      "net_id": "NET_PWR_27V",
      "wire_type": "БПВЛ-0.35",
      "connections": [
        {"component_id": "comp_001", "pin": "1"},
        {"component_id": "comp_002", "pin": "1"}
      ],
      "path_points": [[1480, 520], [3200, 520], [8500, 520]],
      "confidence": 0.98
    }
  ],
  "human_review_required": []
}
```

## 🧪 Тестирование

```bash
# Все тесты
python -m pytest tests/ -v

# Только core тесты
python -m pytest tests/test_core.py -v

# Только stage тесты
python -m pytest tests/test_full_pipeline.py -v

# С покрытием
python -m pytest tests/ --cov=avers --cov-report=html
```

## 📋 Требования

- Python 3.11+
- CUDA-совместимая видеокарта (для нейросетей)
- 16+ GB RAM для обработки больших схем

## 📦 Dependencies

```
numpy>=1.24.0
scipy>=1.11.0
networkx>=3.2.0
pydantic>=2.5.0
opencv-python>=4.8.0
scikit-image>=0.22.0
Pillow>=10.0.0
lxml>=4.9.0
```

### Optional (для ML)
```
ultralytics>=8.0.0  # YOLO/RT-DETR
paddleocr>=2.7.0    # OCR
transformers>=4.35.0  # VLM
```

## 📄 Лицензия

MIT

## 👥 Авторы

AVERS Development Team
