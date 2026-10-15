"""Бенчмарк пайплайна на случайных схемах генератора v2 (avers.dataset.schematic_gen).

Демо-схема одна; здесь - N разных листов с эталоном: форматы A4-A2, масштаб
4-8 px/мм, все классы УГО, таблицы разъёмов, пересечения без точки, рамки.

    venv\\Scripts\\python.exe scripts\\bench_synthetic.py                 # 20 листов, подставной OCR
    venv\\Scripts\\python.exe scripts\\bench_synthetic.py -n 50 --seed 100
    venv\\Scripts\\python.exe scripts\\bench_synthetic.py --degrade       # с эффектами скана
    venv\\Scripts\\python.exe scripts\\bench_synthetic.py --real-ocr      # EasyOCR (медленно)
    venv\\Scripts\\python.exe scripts\\bench_synthetic.py --template-only # только классы шаблонного детектора
    venv\\Scripts\\python.exe scripts\\bench_synthetic.py --json out.json --save-worst DIR

По умолчанию OCR подменяется подписями из эталона (``truth["labels"]``):
измеряются детекция, провода и граф. Итог: полнота/точность по компонентам и
цепям, полнота по классам, худшие листы.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from avers.dataset.demo_schematic import evaluate  # noqa: E402
from avers.dataset.schematic_gen import GenConfig, degrade, generate_sheet  # noqa: E402


def _fake_ocr(truth: Dict):
    from avers.core.validators import SchematicOCR

    labels = [{"text": lb["text"], "bbox": tuple(lb["bbox"]), "confidence": 0.95}
              for lb in truth.get("labels", [])]
    orig = (SchematicOCR.load, SchematicOCR.recognize)
    SchematicOCR.load = lambda self: setattr(self, "_backend", "fake") or True
    SchematicOCR.recognize = lambda self, img: [dict(t) for t in labels]

    def restore():
        SchematicOCR.load, SchematicOCR.recognize = orig
    return restore


def run_one(seed: int, cfg: GenConfig, real_ocr: bool, do_degrade: bool) -> Dict:
    from avers.core.validators import ProductionPipeline

    rgb, truth = generate_sheet(seed, cfg)
    if do_degrade:
        rgb, truth = degrade(rgb, truth, seed=seed, rotate=False)
    restore = None if real_ocr else _fake_ocr(truth)
    t = time.time()
    try:
        result = ProductionPipeline().run(rgb, f"synthetic_{seed}.png")
    finally:
        if restore:
            restore()
    elapsed = time.time() - t
    manifest = result.manifest.model_dump(mode="json")
    ev = evaluate(manifest, truth)
    found = set(ev["designator_map"].values())
    by_class: Dict[str, List[int]] = {}
    for c in truth["components"]:
        hit = by_class.setdefault(c["class_name"], [0, 0])
        hit[1] += 1
        hit[0] += c["designator"] in found
    return {
        "seed": seed, "size": f"{rgb.shape[1]}x{rgb.shape[0]}", "ppm": truth["image"]["px_per_mm"],
        "line": truth["image"]["line_px"], "time_s": round(elapsed, 1), "success": result.success,
        "components": [ev["components_found"], ev["components_expected"]],
        "detected": ev["components_detected"],
        "nets": [ev["nets_matched"], ev["nets_expected"]],
        "net_precision": round(ev["net_precision"], 3),
        "component_precision": round(ev["component_precision"], 3),
        "designators": [ev["designators_correct"], ev["designators_expected"]],
        "by_class": by_class,
        "missed": sorted(c["designator"] for c in truth["components"] if c["designator"] not in found),
        "missing_nets": ev["missing_nets"][:10], "extra_nets": ev["extra_nets"][:10],
        "_rgb": rgb, "_truth": truth,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", type=int, default=20, help="число листов")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--degrade", action="store_true", help="эффекты скана (без наклона)")
    p.add_argument("--real-ocr", action="store_true", help="настоящий OCR вместо подписей из эталона")
    p.add_argument("--template-only", action="store_true", help="только классы шаблонного детектора")
    p.add_argument("--ppm", type=float, nargs=2, metavar=("MIN", "MAX"), help="масштаб px/мм")
    p.add_argument("--json", type=Path, help="отчёт в JSON")
    p.add_argument("--save-worst", type=Path, help="сохранить 3 худших листа (png + эталон)")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    cfg = GenConfig(only_template_classes=args.template_only)
    if args.ppm:
        cfg.ppm_range = tuple(args.ppm)
    rows = []
    for i in range(args.n):
        seed = args.seed + i
        r = run_one(seed, cfg, args.real_ocr, args.degrade)
        rows.append(r)
        c, n = r["components"], r["nets"]
        print(f"seed {seed:5d} {r['size']:>10} ppm {r['ppm']:4.1f} line {r['line']}  "
              f"УГО {c[0]:3d}/{c[1]:3d} (лишних {r['detected'] - c[0]:2d})  цепи {n[0]:3d}/{n[1]:3d} "
              f"точн. {r['net_precision']:.2f}  {r['time_s']:5.1f}с", flush=True)

    comp_f = sum(r["components"][0] for r in rows)
    comp_e = sum(r["components"][1] for r in rows)
    det = sum(r["detected"] for r in rows)
    nets_m = sum(r["nets"][0] for r in rows)
    nets_e = sum(r["nets"][1] for r in rows)
    exact = sum(1 for r in rows if r["nets"][0] == r["nets"][1] and r["net_precision"] == 1.0)
    by_class: Dict[str, List[int]] = {}
    for r in rows:
        for k, (h, t) in r["by_class"].items():
            agg = by_class.setdefault(k, [0, 0])
            agg[0] += h
            agg[1] += t
    print("\nИТОГО по", len(rows), "листам:")
    print(f"  компоненты: полнота {comp_f / max(1, comp_e):.3f} ({comp_f}/{comp_e}), "
          f"точность {comp_f / max(1, det):.3f}")
    print(f"  цепи: полнота {nets_m / max(1, nets_e):.3f} ({nets_m}/{nets_e}); "
          f"листов с точным netlist: {exact}/{len(rows)}")
    print("  полнота по классам (худшие сверху):")
    for k, (h, t) in sorted(by_class.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
        print(f"    {k:18} {h / max(1, t):.2f}  ({h}/{t})")

    worst = sorted(rows, key=lambda r: r["nets"][0] / max(1, r["nets"][1]))[:3]
    if args.save_worst:
        args.save_worst.mkdir(parents=True, exist_ok=True)
        for r in worst:
            stem = args.save_worst / f"synthetic_{r['seed']}"
            cv2.imwrite(str(stem.with_suffix(".png")), cv2.cvtColor(r["_rgb"], cv2.COLOR_RGB2BGR))
            stem.with_suffix(".netlist.json").write_text(json.dumps(r["_truth"], ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
        print(f"\nхудшие листы: {', '.join(str(r['seed']) for r in worst)} -> {args.save_worst}")
    if args.json:
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        args.json.write_text(json.dumps({
            "n": len(rows), "seed": args.seed, "degrade": args.degrade, "real_ocr": args.real_ocr,
            "component_recall": comp_f / max(1, comp_e), "component_precision": comp_f / max(1, det),
            "net_recall": nets_m / max(1, nets_e), "exact_sheets": exact,
            "by_class": by_class, "sheets": clean,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"отчёт: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
