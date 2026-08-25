# Публичные датасеты для обучения АВЕРС

## Обзор

Найдено 8 готовых датасетов для обучения детекции компонентов электрических схем. Все они могут использоваться для pre-training моделей RT-DETR/YOLO перед fine-tuning на ГОСТ УГО.

---

## Список датасетов

### 1. Masala-CHAI ⭐ Рекомендуется
- **Описание:** Large-Scale SPICE Netlist Dataset из 10 учебников, автоматическая детекция YOLOv8
- **Размер:** 4300 изображений, 12 классов
- **Классы:** AC Source, BJT, Battery, Capacitor, DC Source, Diode, Ground, Inductor, MOSFET, Resistor, Current Source, Voltage Source
- **Лицензия:** Research
- **Формат:** YOLO
- **GOST-совместимость:** ✅ Да (близко к ГОСТ)
- **URL:** https://arxiv.org/html/2411.14299v5
- **Особенности:** Включает SPICE netlists, самый близкий к АВЕРС use case

**Скачивание:**
```bash
# Статья содержит ссылки на датасет
# Обычно доступен через авторов
python -m avers dataset public download --dataset masala-chai
```

### 2. ElectroNet ⭐ Рекомендуется
- **Описание:** Small-Scale Object Detection in Electrical Schematic Diagrams
- **Размер:** 3500 схем, 23 класса
- **Классы:** resistor, capacitor, inductor, diode, transistor, mosfet, bjt, op_amp, battery, ground, current_source, voltage_source, switch, transformer, integrated_circuit, led, potentiometer, fuse, relay, connector, antenna, crystal, display + текст с единицами
- **Лицензия:** Research
- **Формат:** YOLO
- **GOST-совместимость:** ✅ Да
- **URL:** https://www.researchgate.net/publication/372298462_ElectroNet
- **Особенности:** Включает текстовые элементы с единицами измерения

### 3. Circuit Diagram (Roboflow) - 2073 изображений
- **Описание:** 2073 изображений с аннотациями компонентов
- **Размер:** 2073 изображений, 12 классов
- **Классы:** capacitor, resistor, diode, transistor, inductor, ground, battery, switch, transformer, integrated_circuit, led, voltage_source
- **Лицензия:** CC BY 4.0 ✅
- **Формат:** YOLO, COCO, etc.
- **GOST-совместимость:** ❌ Частично
- **URL:** https://universe.roboflow.com/deeplearning-l1bq5/circuit-diagram-eo4kn
- **Скачивание:**
```bash
pip install roboflow
# Или через UI: Download -> YOLOv8
python -m avers dataset public download --dataset circuit-diagram-roboflow
```

### 4. Handwritten Circuit Diagram (Roboflow) - 200 изображений
- **Описание:** 200 open source изображений с bounding boxes
- **Размер:** 200 изображений, 12 классов
- **Лицензия:** CC BY 4.0 ✅
- **Формат:** YOLO
- **URL:** https://universe.roboflow.com/deep-learning-in-computer-vision/handwritten-circuit-diagram
- **Особенности:** Готов для YOLO training, хорош для pre-training

### 5. Digitize-HCD
- **Описание:** Digitization of Handwritten Circuit Diagrams, 17 классов + heatmaps для портов
- **Размер:** ~1000 изображений, 17 классов
- **Классы:** resistor, capacitor, inductor, diode, transistor, battery, ground, switch, transformer, op_amp, current_source, voltage_source, ac_source, bulb, antenna, motor, integrated_circuit
- **Лицензия:** CC BY 4.0 ✅
- **Формат:** Custom + heatmaps для портов
- **URL:** https://data.mendeley.com/datasets/rngcz5wtv8/2
- **Особенности:** Включает heatmaps для локализации пинов - полезно для pin detection!

**Скачивание:**
```bash
# Mendeley Data
# https://data.mendeley.com/datasets/rngcz5wtv8/2
python -m avers dataset public download --dataset digitize-hcd
```

### 6. JUHCCR-v1
- **Описание:** Hand-drawn electrical and electronics circuit component recognition
- **Размер:** 20 классов, 150 samples/class original (3000 total), 1500 samples/class augmented (30000 total)
- **Классы:** ammeter, voltmeter, transformer, resistor, ac_source, capacitor, diode, transistor, inductor, battery, ground, switch, bulb, buzzer, antenna, motor, speaker, microphone, integrated_circuit, potentiometer
- **Лицензия:** MIT
- **Формат:** Classification (изолированные компоненты)
- **GOST-совместимость:** ❌
- **URL:** https://github.com/AyushRoy2001/Circuit-Component-Analysis
- **Paper:** https://www.nature.com/articles/s41598-025-22404-5
- **Особенности:** Изолированные компоненты, нужно конвертировать в detection. Хорош для pre-training классификатора

**Скачивание:**
```bash
git clone https://github.com/AyushRoy2001/Circuit-Component-Analysis
```

### 7. CircuitNet
- **Описание:** Hand-drawn schematic sketch recognizer
- **Размер:** 3191 изображений (включая аугментацию), 5 классов
- **Классы:** resistor, capacitor, inductor, diode, voltage_source
- **Лицензия:** MIT
- **URL:** https://github.com/aaanthonyyy/CircuitNet
- **Особенности:** Оригинал из mahmut-aksakalli/circuit_recognizer

### 8. Razavi Custom
- **Описание:** Из книги Razavi Analog CMOS Integrated Circuit Design, 1200 detection + 3552 graph
- **Размер:** 1200 + 3552
- **Классы:** mosfet, resistor, capacitor, current_source, voltage_source, ground, bjt, diode, inductor, op_amp, switch, transformer
- **Лицензия:** Research
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10781286/
- **Особенности:** Включает алгоритм локализации портов

---

## Маппинг в ГОСТ

Публичные датасеты используют англоязычные названия, ГОСТ - русские. Маппинг:

```python
PUBLIC_TO_GOST_MAPPING = {
    "resistor": "resistor",           # Резистор
    "capacitor": "capacitor",         # Конденсатор
    "diode": "diode",                 # Диод
    "ground": "ground",               # Земля
    "relay": "relay",                 # Реле
    "connector": "connector_body",    # Корпус разъема
    "inductor": "resistor",           # Ближайший ГОСТ
    "transformer": "relay",
    "switch": "offpage_connector",    # Переход
    "battery": "ground",              # Питание
    # ...
}
```

---

## Рекомендуемая стратегия обучения

### Этап 1: Pre-training на публичных датасетах
```bash
# Скачайте Masala-CHAI и ElectroNet (самые большие и GOST-совместимые)
python -m avers dataset public list --gost-only
python -m avers dataset public info --dataset masala-chai
python -m avers dataset public download --dataset masala-chai

# Обучение базовой детекции
python -m avers dataset train --data /tmp/public/masala-chai/dataset.yaml --model rtdetr-l --epochs 50
```

### Этап 2: Fine-tuning на синтетическом ГОСТ
```bash
# Генерация ГОСТ датасета
python -m avers dataset generate --output /tmp/avers_gost --num-train 5000 --num-val 500 --image-size 1024

# Дообучение
python -m avers dataset train --data /tmp/avers_gost/dataset.yaml --model /tmp/runs/masala-chai/weights/best.pt --epochs 50
```

### Этап 3: Mixed training
```bash
# Смешанный датасет: синтетика + публичные
python -m avers dataset public mix --synthetic /tmp/avers_gost/dataset.yaml --public masala-chai:/tmp/public/masala-chai,circuit-diagram-roboflow:/tmp/public/circuit-roboflow --output /tmp/mixed

python -m avers dataset train --data /tmp/mixed/dataset_mixed.yaml --model rtdetr-l --epochs 100
```

### Этап 4: Active Learning на реальных БКС
```bash
# Запустите Web UI, исправляйте ошибки
python -m avers web --port 8000

# Feedback автоматически идет в /tmp/avers_feedback/ и RAG
curl http://localhost:8000/api/active-learning/stats

# Дообучение на feedback
curl -X POST http://localhost:8000/api/active-learning/retrain?model_type=rtdetr-l&epochs=20
```

---

## Ожидаемые метрики

- **Pre-training (Masala-CHAI):** mAP@0.5 ~0.75-0.85
- **Fine-tuning (GOST synthetic):** mAP@0.5 >0.85 на синтетике
- **Mixed:** mAP@0.5 >0.80 на публичных + синтетике
- **Active Learning (реальные БКС):** mAP@0.5 >0.70 после 50+ feedback

---

## Использование в АВЕРС

```yaml
# config.yaml
detection:
  model_path: /tmp/mixed/runs/train/weights/best.pt  # Обученная на mixed
  model_type: rtdetr
  confidence_threshold: 0.25
  device: cuda
```

```python
from avers.dataset.public_datasets import PublicDatasetLoader

loader = PublicDatasetLoader()

# List
datasets = loader.list_datasets(gost_compatible_only=True)
for ds in datasets:
    print(f"{ds.name}: {ds.num_images} images")

# Convert public to GOST
loader.convert_to_gost("masala-chai", Path("/tmp/masala-chai"), Path("/tmp/gost_converted"))

# Mixed dataset
loader.create_mixed_dataset_config(
    synthetic_dataset_yaml=Path("/tmp/avers_gost/dataset.yaml"),
    public_datasets=[("masala-chai", Path("/tmp/public/masala-chai"))],
    output_path=Path("/tmp/mixed")
)
```

---

## Лицензии

- ✅ CC BY 4.0: circuit-diagram-roboflow, handwritten-circuit-roboflow, digitize-hcd — можно использовать свободно
- ⚠️ MIT: juhccr-v1, circuitnet — можно с указанием авторства
- ⚠️ Research: masala-chai, electronet, razavi-custom — для исследований, уточните у авторов для коммерческого использования

Для коммерческого БКС проекта: используйте CC BY 4.0 и MIT датасеты для pre-training, затем синтетический ГОСТ и active learning на своих данных.

---

## CLI Reference

```bash
python -m avers dataset public list
python -m avers dataset public list --gost-only
python -m avers dataset public info --dataset masala-chai
python -m avers dataset public download --dataset masala-chai
python -m avers dataset public convert --dataset masala-chai --input /tmp/masala-chai --output /tmp/gost
python -m avers dataset public mix --synthetic /tmp/gost/dataset.yaml --public masala-chai:/tmp/masala-chai --output /tmp/mixed
python -m avers dataset public strategy
```
