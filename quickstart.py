#!/usr/bin/env python3
"""
AVERS Quickstart - быстрый старт всех компонентов.

Запускает:
  1. Генерацию синтетического датасета (10 изображений)
  2. Тест Vision RAG
  3. Тест пайплайна
  4. Инструкции для Web UI
"""

import sys
from pathlib import Path
import numpy as np
import cv2

print("""
╔════════════════════════════════════════════════════════════╗
║  АВЕРС v0.2 - Quickstart                                  ║
║  Автоматическая Векторизация и Распознавание Схем         ║
╚════════════════════════════════════════════════════════════╝
""")

# 1. Synthetic dataset
print("\n[1/4] Генерация синтетического датасета ГОСТ УГО...")
try:
    from avers.dataset.synthetic import GOSTGenerator, SyntheticConfig
    
    config = SyntheticConfig(image_size=512, min_objects=5, max_objects=10, enable_wires=True, enable_noise=False)
    gen = GOSTGenerator(config)
    
    out_dir = Path("/tmp/avers_quickstart")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        img, anns = gen.generate_realistic_schematic()
        cv2.imwrite(str(out_dir / f"quickstart_{i}.jpg"), img)
        print(f"  ✓ quickstart_{i}.jpg - {len(anns)} аннотаций")
    
    print(f"  → Сохранено в {out_dir}")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

# 2. Vision RAG
print("\n[2/4] Тест Vision RAG...")
try:
    from avers.rag import VisionRAG
    
    rag = VisionRAG(storage_path=out_dir / "rag")
    
    # Add example
    roi = np.ones((256, 256, 3), dtype=np.uint8) * 255
    cv2.circle(roi, (128, 128), 6, (0, 0, 0), -1)
    cv2.line(roi, (0, 128), (256, 128), (0, 0, 0), 2)
    cv2.line(roi, (128, 0), (128, 256), (0, 0, 0), 2)
    
    entry_id = rag.add_example(roi, label="junction_dot_connected", description="Точка соединения")
    print(f"  ✓ Добавлен пример {entry_id}")
    
    # Query
    response = rag.query(image=roi, text="junction dot connected?", top_k=2)
    print(f"  ✓ Запрос: найдено {len(response.results)} примеров")
    print(f"  → VLM ответ: {response.vlm_answer}")
    
    rag.save()
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

# 3. Pipeline
print("\n[3/4] Тест Production Pipeline...")
try:
    from avers.pipeline import ProductionPipeline
    from avers.config import AVERSConfig
    
    config = AVERSConfig()
    config.vlm_arbitrator.enabled = False
    config.vision_rag.enabled = True
    
    pipeline = ProductionPipeline(config)
    
    # Test image
    test_img = np.ones((500, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_img, (50, 100), (100, 200), (0, 0, 0), 2)
    cv2.putText(test_img, "X1", (55, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.line(test_img, (100, 150), (700, 150), (0, 0, 0), 2)
    cv2.circle(test_img, (400, 150), 4, (0, 0, 0), -1)
    
    result = pipeline.run(test_img, "quickstart.tif", dpi=300)
    print(f"  ✓ Пайплайн: {len(result.manifest.components)} компонентов, {len(result.manifest.nets)} цепей")
    print(f"  → Время: {result.manifest.schema_metadata.processing_time_seconds:.2f}s")
    for stage, t in result.stage_timings.items():
        print(f"    - {stage}: {t*1000:.0f}ms")
    
    # Save result
    result.manifest.save(out_dir / "quickstart_result.json")
    print(f"  → Результат: {out_dir / 'quickstart_result.json'}")
    
except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# 4. Web UI instructions
print("\n[4/4] Web UI Валидатор...")
try:
    from avers.web.app import create_app
    app = create_app()
    print("  ✓ FastAPI app готов")
    print("""
  Запуск Web UI:
    python -m avers web --port 8000

  Откройте:
    http://localhost:8000           - Валидатор (загрузка схем, визуализация, редактирование)
    http://localhost:8000/annotator - Аннотатор ГОСТ УГО (ручная разметка)
    http://localhost:8000/docs      - API документация

  Docker:
    docker build -t avers .
    docker run -p 8000:8000 avers
    docker-compose up

  Датасет:
    python -m avers dataset generate --output /tmp/dataset --num-train 1000
    python -m avers dataset train --data /tmp/dataset/dataset.yaml --model rtdetr-l

  RAG:
    python -m avers rag index --image junction.jpg --label junction_dot
    python -m avers rag query --text "junction dot" --top-k 5
""")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

print("""
╔════════════════════════════════════════════════════════════╗
║  Quickstart завершен!                                     ║
║  Проверьте /tmp/avers_quickstart/                         ║
╚════════════════════════════════════════════════════════════╝
""")
