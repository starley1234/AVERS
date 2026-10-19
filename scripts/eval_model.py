"""Оценка обученной модели (YOLO / RT-DETR) на листах эталона генератора v2.

Листы берутся из ``truth/val`` датасета, собранного ``generate-v2``: по seed
лист воспроизводится бит-в-бит (``schematic_gen.dataset_sheet``), поэтому
эталон из файла соответствует картинке. Гоняются три режима пайплайна:

* ``templates`` - только шаблонный детектор по библиотеке ГОСТ (базовая линия);
* ``model``     - только обученные веса, без добора шаблонами;
* ``hybrid``    - веса + уточнение по шаблонам ГОСТ + добор пропущенного
                  (``detection.merge_templates``).

    venv\\Scripts\\python.exe scripts\\eval_model.py --weights runs/avers_yolo/weights/best.pt
    venv\\Scripts\\python.exe scripts\\eval_model.py --weights W.pt -n 40 --modes model hybrid --json out.json

Метрики те же, что у ``bench_synthetic.py`` (см. ``demo_schematic.evaluate``):
полнота/точность УГО, полнота цепей, листов с точным netlist.
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

from avers.config import AVERSConfig  # noqa: E402
from avers.dataset.demo_schematic import evaluate  # noqa: E402
from avers.dataset.schematic_gen import GenConfig, dataset_sheet  # noqa: E402

# Так собирался датасет (CLI generate-v2: --ppm-min 4 --ppm-max 12 --degrade 0.7)
PPM_RANGE = (4.0, 12.0)
DEGRADE_P = 0.7


def _fake_ocr(truth: Dict):
    """Подменный OCR: рамки подписей из эталона (изолирует детекцию от OCR)."""
    from avers.core.validators import SchematicOCR

    labels = [{"text": lb["text"], "bbox": tuple(lb["bbox"]), "confidence": 0.95}
              for lb in truth.get("labels", [])]
    orig = (SchematicOCR.load, SchematicOCR.recognize)
    SchematicOCR.load = lambda self: setattr(self, "_backend", "fake") or True
    SchematicOCR.recognize = lambda self, img: [dict(t) for t in labels]

    def restore():
        SchematicOCR.load, SchematicOCR.recognize = orig
    return restore


def mode_config(base: AVERSConfig, mode: str, weights: Optional[str], args) -> AVERSConfig:
    """Конфиг пайплайна под режим оценки."""
    cfg = base.model_copy(deep=True)
    cfg.vlm_arbitrator.enabled = False
    cfg.slicing.tile_size = args.tile
    cfg.detection.device = args.device
    cfg.detection.confidence_threshold = args.conf
    cfg.detection.use_registry = False
    if mode == "templates":
        cfg.detection.model_path = None
        cfg.detection.template_fallback = True
    elif mode == "model":
        cfg.detection.model_path = weights
        cfg.detection.template_fallback = False
        cfg.detection.merge_templates = False
    elif mode == "hybrid":
        cfg.detection.model_path = weights
        cfg.detection.template_fallback = False
        cfg.detection.merge_templates = True
    else:
        raise ValueError(f"unknown mode: {mode}")
    return cfg


def val_sheets(data_dir: Path, n: int) -> List[Path]:
    truth_dir = data_dir / "truth" / "val"
    files = sorted(truth_dir.glob("s*.json"))
    if not files:
        raise SystemExit(f"нет эталонов в {truth_dir} (сначала generate-v2)")
    return files[:n] if n > 0 else files


def run_mode(base: AVERSConfig, mode: str, weights: Optional[str], args, sheets: List[Path]) -> Dict:
    from avers.core.validators import ProductionPipeline

    cfg = mode_config(base, mode, weights, args)
    pipeline = ProductionPipeline(cfg)
    gen_cfg = GenConfig(ppm_range=PPM_RANGE)
    rows = []
    for path in sheets:
        seed = int(path.stem[1:])
        truth = json.loads(path.read_text(encoding="utf-8"))
        rgb, truth_live = dataset_sheet(seed, gen_cfg, DEGRADE_P)
        # эталон из файла и только что сгенерированный должны совпадать
        assert len(truth_live["components"]) == len(truth["components"]), seed
        restore = None if args.real_ocr else _fake_ocr(truth_live)
        t = time.time()
        try:
            result = pipeline.run(rgb, f"val_{seed}.png")
        finally:
            if restore:
                restore()
        elapsed = time.time() - t
        manifest = result.manifest.model_dump(mode="json")
        ev = evaluate(manifest, truth_live)
        found = set(ev["designator_map"].values())
        by_class: Dict[str, List[int]] = {}
        for c in truth_live["components"]:
            hit = by_class.setdefault(c["class_name"], [0, 0])
            hit[1] += 1
            hit[0] += c["designator"] in found
        rows.append({
            "seed": seed, "size": f"{rgb.shape[1]}x{rgb.shape[0]}", "time_s": round(elapsed, 1),
            "components": [ev["components_found"], ev["components_expected"]],
            "detected": ev["components_detected"],
            "nets": [ev["nets_matched"], ev["nets_expected"]],
            "net_precision": round(ev["net_precision"], 3),
            "component_precision": round(ev["component_precision"], 3),
            "by_class": by_class,
            "backend": getattr(pipeline, "_last_detector", None) and pipeline._last_detector.backend,
            "detector_stats": (getattr(pipeline, "_last_detector", None) or type("x", (), {})).__dict__.get("last_stats"),
        })
        c, nt = rows[-1]["components"], rows[-1]["nets"]
        print(f"  seed {seed:8d} {rows[-1]['size']:>10}  УГО {c[0]:3d}/{c[1]:3d}  "
              f"цепи {nt[0]:3d}/{nt[1]:3d}  точн. {rows[-1]['net_precision']:.2f}  "
              f"{rows[-1]['time_s']:5.1f}с", flush=True)

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
    return {
        "mode": mode, "weights": weights, "sheets": len(rows),
        "component_recall": comp_f / max(1, comp_e), "component_precision": comp_f / max(1, det),
        "components": [comp_f, comp_e], "detected": det,
        "net_recall": nets_m / max(1, nets_e), "nets": [nets_m, nets_e],
        "exact_sheets": exact, "by_class": by_class,
        "avg_time_s": round(sum(r["time_s"] for r in rows) / max(1, len(rows)), 1),
        "rows": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", "-w", help="веса AVERS (.pt)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "synth_v2", help="каталог generate-v2")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--modes", nargs="+", default=["templates", "model", "hybrid"],
                   choices=["templates", "model", "hybrid"])
    p.add_argument("-n", type=int, default=20, help="число листов валидации (0 = все)")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--real-ocr", action="store_true", help="настоящий EasyOCR вместо подписей из эталона")
    p.add_argument("--json", type=Path, help="отчёт в JSON")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if ("model" in args.modes or "hybrid" in args.modes) and not args.weights:
        raise SystemExit("для режимов model/hybrid нужен --weights")
    weights = args.weights
    if weights:
        weights = str(Path(weights).resolve())
        if not Path(weights).is_file():
            raise SystemExit(f"нет файла весов: {weights}")

    base = AVERSConfig.from_yaml(args.config) if args.config and Path(args.config).is_file() else AVERSConfig()
    sheets = val_sheets(args.data, args.n)
    print(f"листов эталона: {len(sheets)} ({sheets[0].stem}…{sheets[-1].stem}); режимы: {', '.join(args.modes)}")

    report = {}
    for mode in args.modes:
        print(f"\n=== режим {mode} ===")
        res = run_mode(base, mode, weights, args, sheets)
        report[mode] = res
        print(f"  ИТОГО {mode}: УГО полнота {res['component_recall']:.3f} "
              f"({res['components'][0]}/{res['components'][1]}), точность {res['component_precision']:.3f}; "
              f"цепи {res['net_recall']:.3f} ({res['nets'][0]}/{res['nets'][1]}); "
              f"точных листов {res['exact_sheets']}/{res['sheets']}; {res['avg_time_s']}с/лист")

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
