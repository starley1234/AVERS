"""Проверка OCR на демо-схеме: что прочитал EasyOCR и как это легло в netlist.

    venv\\Scripts\\python.exe scripts\\ocr_demo_check.py [путь_к_png]
"""
import json
import sys
import time

import cv2

from avers.core.validators import ProductionPipeline, SchematicOCR
from avers.dataset.demo_schematic import build_demo, evaluate

image, truth = build_demo()
if len(sys.argv) > 1:
    image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)

ocr = SchematicOCR()
t = time.time()
ocr.load()
print(f"backend: {ocr._backend} (load {time.time() - t:.1f}s)")
t = time.time()
texts = ocr.recognize(image)
print(f"OCR: {len(texts)} texts in {time.time() - t:.1f}s")
for x in sorted(texts, key=lambda x: (x["bbox"][1], x["bbox"][0])):
    print(f"  {x['text']!r:28} conf={x['confidence']:.2f} bbox={x['bbox']}")

t = time.time()
result = ProductionPipeline().run(image, "demo.png")
manifest = result.manifest.model_dump(mode="json")
ev = evaluate(manifest, truth)
print(f"\npipeline {time.time() - t:.1f}s, timings: "
      + ", ".join(f"{k}={v:.2f}s" for k, v in result.stage_timings.items()))
print(json.dumps({k: ev[k] for k in (
    "components_found", "components_expected", "nets_matched", "nets_expected",
    "net_precision", "designators_correct", "designators_expected", "designators_wrong",
)}, ensure_ascii=False))
for c in manifest["components"]:
    if c["type"] == "junction_dot":
        continue
    ta = c["text_associations"]
    print(f"  {c['designator']:6} {ta.get('gost_class', ''):16} src={ta.get('designator_source', '')}"
          f" ocr={ta.get('ocr_text', '')!r} pins={[p['pin_number'] for p in c['pins']]}")
