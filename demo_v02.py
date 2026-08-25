#!/usr/bin/env python3
"""
AVERS v0.2 Demo - Web UI + Dataset + Vision RAG

Демонстрация всех новых фич:
  - Synthetic ГОСТ dataset generation
  - Vision RAG
  - Web API
"""

import numpy as np
import cv2
from pathlib import Path

print("="*60)
print("AVERS v0.2 - Full Feature Demo")
print("="*60)

# 1. Synthetic Dataset
print("\n1. Synthetic ГОСТ УГО Dataset Generation")
print("-"*60)
from avers.dataset.gost_symbols import GOST_SYMBOLS
from avers.dataset.synthetic import GOSTGenerator, SyntheticConfig

print(f"Доступные классы ГОСТ: {len(GOST_SYMBOLS)}")
for id, sym in GOST_SYMBOLS.items():
    print(f"  {id}: {sym.class_name:20s} - {sym.gost_standard:20s} {sym.description}")

config = SyntheticConfig(
    image_size=1024,
    min_objects=8,
    max_objects=20,
    enable_wires=True,
    enable_noise=True,
    enable_scan_effects=True,
)

gen = GOSTGenerator(config)
img, anns = gen.generate_image()
print(f"\n✓ Сгенерировано изображение: {img.shape}")
print(f"  Аннотаций: {len(anns)}")
for ann in anns[:5]:
    print(f"    - {ann.class_name} {ann.bbox}")

# Realistic schematic
img2, anns2 = gen.generate_realistic_schematic()
print(f"\n✓ Реалистичная схема БКС: {img2.shape}, {len(anns2)} аннотаций")

# Save preview
preview_dir = Path("/tmp/avers_v02_preview")
preview_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(preview_dir / "synthetic_1.jpg"), img)
cv2.imwrite(str(preview_dir / "realistic_1.jpg"), img2)
print(f"✓ Превью сохранены в {preview_dir}")

# 2. Dataset Composer (A2x6)
print("\n2. Генерация больших схем A2x6")
print("-"*60)
from avers.dataset.generator import SchematicComposer

composer = SchematicComposer(config)
large_img, large_anns = composer.compose_a2x6(width=4000, height=1000)
print(f"✓ Большая схема: {large_img.shape}, {len(large_anns)} аннотаций")
cv2.imwrite(str(preview_dir / "large_a2x6.jpg"), large_img)

# 3. Vision RAG
print("\n3. Vision RAG - Retrieval Augmented Generation")
print("-"*60)
from avers.rag import VisionRAG
from avers.rag.embeddings import CLIPEmbedding

rag = VisionRAG(storage_path=preview_dir / "rag_db")

# Добавляем примеры
print("Индексация примеров...")
examples = [
    ("junction_dot_connected", "Точка соединения - есть контакт", np.random.randint(0,255,(256,256,3),dtype=np.uint8)),
    ("junction_dot_none", "Пересечение без точки - нет контакта", np.random.randint(0,255,(256,256,3),dtype=np.uint8)),
    ("ground", "Заземление - три линии", np.random.randint(0,255,(256,256,3),dtype=np.uint8)),
    ("diode", "Диод - треугольник + черта", np.random.randint(0,255,(256,256,3),dtype=np.uint8)),
]

for label, desc, roi in examples:
    # Draw something representative
    if "junction" in label:
        cv2.circle(roi, (128,128), 8, (0,0,0), -1)
        if "connected" in label:
            cv2.line(roi, (0,128), (256,128), (0,0,0), 2)
            cv2.line(roi, (128,0), (128,256), (0,0,0), 2)
    elif label == "ground":
        cv2.line(roi, (128,50), (128,150), (0,0,0), 2)
        cv2.line(roi, (80,150), (176,150), (0,0,0), 2)
    
    entry_id = rag.add_example(roi, label=label, description=desc)
    print(f"  ✓ {entry_id}: {label}")

# Query
print("\nЗапрос к RAG:")
query_roi = np.ones((256,256,3), dtype=np.uint8)*255
cv2.circle(query_roi, (128,128), 6, (0,0,0), -1)
cv2.line(query_roi, (0,128), (256,128), (0,0,0), 2)

response = rag.query(image=query_roi, text="Is there a junction dot at center?", top_k=3)
print(f"  Найдено {len(response.results)} похожих примеров:")
for r in response.results:
    print(f"    - {r.entry.label} (score: {r.score:.3f}): {r.entry.description}")

print(f"  VLM ответ: {response.vlm_answer}")

rag.save()
print(f"✓ RAG база сохранена: {rag.stats()}")

# 4. Training config
print("\n4. Конфигурация обучения RT-DETR / YOLO")
print("-"*60)
from avers.dataset.train import get_training_config

yolo_cfg = get_training_config("yolo")
rtdetr_cfg = get_training_config("rtdetr")

print("YOLO config:")
for k, v in list(yolo_cfg.items())[:8]:
    print(f"  {k}: {v}")

print("\nRT-DETR config:")
for k, v in list(rtdetr_cfg.items())[:8]:
    print(f"  {k}: {v}")

print("\nКоманда обучения:")
print("  python -m avers.dataset.cli train --data /tmp/avers_dataset/dataset.yaml --model rtdetr-l --epochs 100")

# 5. Web UI
print("\n5. Web UI Валидатор")
print("-"*60)
from avers.web.app import create_app

app = create_app()
print("✓ FastAPI app создан")
print("  Роуты:")
for route in app.routes:
    if hasattr(route, 'path'):
        print(f"    {route.path}")

print("\nЗапуск Web UI:")
print("  python -m avers web --port 8000")
print("  Откройте http://localhost:8000")
print("  Аннотатор: http://localhost:8000/annotator")
print("  API docs: http://localhost:8000/docs")

# 6. Full pipeline with RAG integration
print("\n6. Полный пайплайн с Vision RAG")
print("-"*60)
from avers.pipeline import ProductionPipeline
from avers.config import AVERSConfig

config = AVERSConfig()
config.vlm_arbitrator.enabled = False  # Disable real VLM for demo
config.vision_rag.enabled = True

pipeline = ProductionPipeline(config)

# Create test schematic
test_img = np.ones((1000, 2000, 3), dtype=np.uint8)*255
cv2.rectangle(test_img, (100,200), (180,400), (0,0,0), 2)
cv2.putText(test_img, "X1", (110,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
cv2.line(test_img, (180,250), (900,250), (0,0,0), 2)
cv2.circle(test_img, (500,250), 4, (0,0,0), -1)

result = pipeline.run(test_img, "demo_v02.tif", dpi=300)
print(f"✓ Пайплайн выполнен:")
print(f"  Компонентов: {len(result.manifest.components)}")
print(f"  Цепей: {len(result.manifest.nets)}")
print(f"  Проблем: {len(result.manifest.human_review_required)}")
print(f"  Время: {result.manifest.schema_metadata.processing_time_seconds:.2f}s")
for stage, t in result.stage_timings.items():
    print(f"    {stage}: {t*1000:.1f}ms")

print("\n" + "="*60)
print("Demo v0.2 Complete!")
print("="*60)
print("""
Что нового в v0.2:

1. Web UI Валидатор (минимализм + принципы UI):
   - Загрузка сканов drag&drop
   - Визуализация всех 6 стадий
   - Редактирование компонентов (human-in-the-loop)
   - Разрешение коллизий
   - Экспорт JSON/XML
   - Zoom/pan canvas с overlay

2. Synthetic Dataset ГОСТ УГО:
   - 10 классов по ГОСТ 2.721, 2.728, 2.730, 2.755, 2.756
   - Генерация реалистичных схем БКС
   - Формат A2x6 (14000x3500)
   - Аугментации: шум скана, размытие, виньетка
   - Экспорт YOLO/COCO

3. Инструмент ручной разметки:
   - Web-аннотатор (/annotator)
   - Рисование боксов
   - Горячие клавиши 1-9, Del, S
   - Экспорт в YOLO

4. Обучение RT-DETR/YOLO:
   - Поддержка YOLOv11x/m и RT-DETRv2-l/x
   - ГОСТ-специфичные аугментации
   - Экспорт в ONNX
   - CLI: avers dataset train

5. Vision RAG:
   - CLIP ViT-B/32 embeddings (512d)
   - FAISS / brute-force vector store
   - Few-shot VLM prompting
   - Интеграция в Stage 6 (VLM-арбитраж)
   - Пополнение из валидатора

Запуск:
  pip install -r requirements.txt
  pip install fastapi uvicorn  # для Web UI
  python -m avers web --port 8000
  python -m avers dataset generate --output /tmp/dataset --num-train 1000
  python demo_v02.py
""")
