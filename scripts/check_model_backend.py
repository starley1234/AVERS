"""Проверка подключения обученных весов (ultralytics) без обучения.

    venv\\Scripts\\python.exe scripts\\check_model_backend.py [weights.pt]

1. загружает веса через UltralyticsTiledDetector;
2. если классы модели не ГОСТ (например, COCO yolo*.pt) - печатает понятную
   ошибку, затем всё равно прогоняет предсказание тайла, чтобы проверить разбор
   результатов ultralytics;
3. если классы ГОСТ - прогоняет весь пайплайн на демо-схеме и сверяет с эталоном.
"""
import sys
import time
from pathlib import Path

import numpy as np

from avers.config import AVERSConfig
from avers.core.validators import ProductionPipeline
from avers.dataset.demo_schematic import build_demo, evaluate
from avers.stages.stage2_detection.model_detector import (
    ModelClassMismatch, UltralyticsTiledDetector, resolve_device,
)

weights = sys.argv[1] if len(sys.argv) > 1 else str(Path("weights") / "yolo26n.pt")
print(f"weights: {weights}, device: {resolve_device('auto')}")
image, truth = build_demo()
det = UltralyticsTiledDetector(weights, device="auto")
try:
    det.load()
    gost = True
except ModelClassMismatch as e:
    print(f"OK (ожидаемо для не-AVERS весов): {e}")
    gost = False

if not gost:
    t = time.time()
    rows = det._predict_tile(np.ascontiguousarray(image[:, :, ::-1]))
    print(f"ultralytics predict OK: {len(rows)} boxes (COCO classes) in {time.time() - t:.2f}s")
    sys.exit(0)

config = AVERSConfig.from_yaml("config.yaml")
config.detection.model_path = weights
result = ProductionPipeline(config).run(image, "demo.png")
ev = evaluate(result.manifest.model_dump(mode="json"), truth)
print("warnings:", result.warnings)
print({k: ev[k] for k in ("components_found", "components_expected", "nets_matched",
                          "nets_expected", "net_precision")})
