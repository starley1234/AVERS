# АВЕРС — Автоматическая Векторизация и Распознавание Схем

**AVERS** (Automated Vectorization and Recognition of Schematics) — система автоматической векторизации и семантической оцифровки схем бортовых кабельных сетей (БКС).

## Назначение

Преобразование растровых сканов принципиальных электрических схем в **математический граф связей** (Netlist / JSON / XML), пригодный для прямой загрузки в САПР («Макс-САПР», «КОМПАС-Электрик»).

### Ключевые возможности

- **Многостадийный гибридный пайплайн**: Детекция УГО + OCR + Computer Vision + Графовый синтез
- **Обработка больших форматов**: А2х6 (до 15000×4000 px) при 300+ DPI
- **SAHI-нарезка**: Сохранение детализации при параллельной обработке
- **Точечный VLM-арбитраж**: Разрешение коллизий только для неопределённых участков
- **Экспорт в САПР**: JSON/XML формат, совместимый с российскими САПР

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│  Исходный скан (А2х6, TIF/PNG/PDF)                          │
│  Разрешение: 300+ DPI, ~14000x3500 px                       │
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

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/starley1234/AVERS.git
cd AVERS

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Быстрый старт

```bash
# Обработка схемы
python -m avers.main input.tif --output result.json

# С XML экспортом
python -m avers.main board.png -o nets.xml --format xml

# С пользовательской конфигурацией
python -m avers.main schematic.tif -c config.yaml --visualize
```

## Использование в Python

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

## Стадии обработки

| Стадия | Модуль | Описание |
|--------|--------|----------|
| 1 | SAHI Slicing | Нарезка на тайлы 1024×1024 с перекрытием |
| 2 | УГО Detection | RT-DETR / YOLO детекция компонентов |
| 3 | OCR | PaddleOCR распознавание текста |
| 4 | Vectorization | OpenCV скелетизация и векторизация линий |
| 5 | Graph Synthesis | NetworkX сборка графа, выделение цепей |
| 6 | VLM Arbitration | Разрешение коллизий через Gemma/Qwen-VL |

## Конфигурация

```yaml
# config.yaml
project_name: AVERS
version: "0.1.0"

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
  skeletonize_method: guo_hall
  rdp_epsilon: 2.0
  snap_radius: 15

graph_synthesis:
  enabled: true
  snap_enabled: true
  merge_collinear_segments: true

vlm_arbitrator:
  enabled: true
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  roi_size: 256
  confidence_threshold: 0.65
```

## Формат выходных данных

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
        {"pin_number": "1", "coord": [1480, 520]},
        {"pin_number": "2", "coord": [1480, 560]}
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

## Требования

- Python 3.11+
- CUDA-совместимая видеокарта (для нейросетей)
- 16+ GB RAM для обработки больших схем

## Текущий статус

⚠️ **В разработке**: Ядро пайплайна (Stages 1, 5) + интерфейсы заглушки для остальных модулей

### Реализовано
- ✅ Stage 1: SAHI-нарезка изображений
- ✅ Stage 5: Графовый синтез (NetworkX)
- ✅ Конфигурация и типы данных (Pydantic)
- ✅ CLI интерфейс
- ✅ JSON/XML экспорт

### В разработке
- 🔄 Stage 2: Интеграция RT-DETR/YOLO
- 🔄 Stage 3: PaddleOCR
- 🔄 Stage 4: OpenCV векторизация
- 🔄 Stage 6: VLM арбитраж

## Лицензия

MIT

## Авторы

AVERS Development Team
