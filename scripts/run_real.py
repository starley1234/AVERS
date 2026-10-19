"""Прогон пайплайна на реальных схемах: netlist + картинки для просмотра.

Показывает, что находит детектор (обученные веса YOLO/RT-DETR, гибрид с
шаблонами ГОСТ или одни шаблоны), и сохраняет рядом с результатом:

* ``<имя>.netlist.json``  - полный manifest пайплайна;
* ``<имя>.detections.png`` - лист с рамками найденных УГО (уменьшенный для просмотра);
* ``<имя>.netlist.txt``   - читаемый список цепей «компонент:вывод».

    venv\\Scripts\\python.exe scripts\\run_real.py data\\2026_10 -o data\\out_real
    venv\\Scripts\\python.exe scripts\\run_real.py path\\to\\sheet.jpg -w runs\\avers_yolo\\weights\\best.pt
    venv\\Scripts\\python.exe scripts\\run_real.py data\\2026_10 --templates-only   # без обученных весов
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from avers.config import AVERSConfig  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}

# Имена файлов реальных схем часто кириллицей; артефакты пишем ASCII-именами,
# иначе их неудобно читать в консоли/браузере (cp1251/cp866-грабли).
_TRANSLIT = str.maketrans({
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E", "Ж": "Zh", "З": "Z",
    "И": "I", "Й": "I", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh",
    "Щ": "Shch", "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})


def ascii_stem(path: Path) -> str:
    """ASCII-имя для выходных файлов (кириллица транслитерируется)."""
    stem = path.stem.translate(_TRANSLIT)
    stem = re.sub(r"[^0-9A-Za-z._ -]+", "_", stem).strip(" ._-")
    return stem or "sheet"


def with_ext(stem: Path, ext: str) -> Path:
    """``stem`` + ``ext`` дописыванием (``with_suffix`` съел бы «.2_» в имени)."""
    return stem.parent / f"{stem.name}{ext}"



def collect_inputs(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT))
        elif p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return out


def build_config(args) -> AVERSConfig:
    cfg = AVERSConfig.from_yaml(args.config) if args.config and Path(args.config).is_file() else AVERSConfig()
    cfg.vlm_arbitrator.enabled = False
    cfg.slicing.tile_size = args.tile
    cfg.detection.device = args.device
    cfg.detection.confidence_threshold = args.conf
    cfg.detection.template_fallback = True
    cfg.detection.use_registry = args.use_registry
    if args.templates_only:
        cfg.detection.model_path = None
    else:
        cfg.detection.model_path = str(Path(args.weights).resolve()) if args.weights else None
    cfg.detection.merge_templates = args.merge_templates
    return cfg


def nice_image(image: np.ndarray, max_side: int = 2400) -> np.ndarray:
    """Уменьшенная копия листа для просмотра."""
    h, w = image.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1.0:
        return cv2.resize(image, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    return image


def zoom_crops(vis: np.ndarray, dets: List[Dict], out_stem: Path, count: int,
               box: tuple = (900, 600)) -> List[Path]:
    """Крупные (1:1) вырезки вокруг самых уверенных детекций - там, где на листе
    видно сами УГО, а не уменьшенная картинка целиком."""
    cw, ch = box
    h, w = vis.shape[:2]
    saved: List[Path] = []
    taken: List[tuple] = []

    def overlaps(x0, y0, x1, y1) -> bool:
        for a0, b0, a1, b1 in taken:
            if min(x1, a1) - max(x0, a0) > cw * 0.4 and min(y1, b1) - max(y0, b0) > ch * 0.4:
                return True
        return False

    ranked = sorted(dets, key=lambda d: -d.get("confidence", 0))
    for d in ranked:
        if len(saved) >= count:
            break
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x0 = max(0, min(w - cw, cx - cw // 2)) if w > cw else 0
        y0 = max(0, min(h - ch, cy - ch // 2)) if h > ch else 0
        x1c, y1c = min(w, x0 + cw), min(h, y0 + ch)
        if overlaps(x0, y0, x1c, y1c):
            continue
        taken.append((x0, y0, x1c, y1c))
        path = out_stem.with_name(f"{out_stem.name}.crop{len(saved) + 1}.png")
        cv2.imwrite(str(path), cv2.cvtColor(vis[y0:y1c, x0:x1c], cv2.COLOR_RGB2BGR))
        saved.append(path)
    return saved


def netlist_text(manifest: Dict) -> str:
    lines = []
    comps = {c["id"]: c for c in manifest.get("components", [])}
    for c in manifest.get("components", []):
        ta = c.get("text_associations", {}) or {}
        pins = ", ".join(f"{k}" for k in (c.get("pins") or {}))
        lines.append(f"{c.get('designator'):>8}  {ta.get('gost_class', c.get('type')):<18} "
                     f"bbox={c['bbox']} source={ta.get('designator_source', '-'):<4} pins=[{pins}]")
    lines.append("")
    lines.append(f"ЦЕПИ ({len(manifest.get('nets', []))}):")
    for net in manifest.get("nets", []):
        refs = []
        for conn in net.get("connections", []):
            c = comps.get(conn["component_id"], {})
            refs.append(f"{c.get('designator', conn['component_id'])}:{conn['pin']}")
        lines.append(f"  [{net.get('net_id')}] " + " - ".join(sorted(refs)))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", type=Path, help="файлы схем или каталог")
    p.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "out_real", help="куда писать результат")
    p.add_argument("-w", "--weights", type=str, default=None, help="веса AVERS (.pt)")
    p.add_argument("-c", "--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--templates-only", action="store_true", help="без весов: только шаблоны ГОСТ")
    p.add_argument("--use-registry", action="store_true", help="брать текущие веса из реестра active learning")
    p.add_argument("--merge-templates", action=argparse.BooleanOptionalAction, default=True,
                   help="добирать УГО, пропущенные моделью, шаблонами (по умолчанию да)")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-side", type=int, default=2400, help="сторона картинки для просмотра")
    p.add_argument("--crops", type=int, default=0, help="сделать N крупных вырезок вокруг самых уверенных УГО")
    p.add_argument("--json", type=Path, help="сводка по всем листам в JSON")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if not args.templates_only and not args.weights and not args.use_registry:
        print("веса не заданы -> работают только шаблоны ГОСТ (--templates-only)"
              " или укажите -w/--use-registry", file=sys.stderr)
        args.templates_only = True

    files = collect_inputs(args.inputs)
    if not files:
        print("не найдено входных файлов", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    from avers.core.validators import ProductionPipeline
    from avers.utils.image_helpers import load_image_auto, is_pdf
    from avers.web.visualization import visualize_detections_enhanced

    config = build_config(args)
    pipeline = ProductionPipeline(config)

    summary = []
    for path in files:
        print(f"\n=== {path.name} ===", flush=True)
        images = load_image_auto(path, dpi=300)
        for page, rgb in enumerate(images):
            suffix = "" if len(images) == 1 else f".p{page}"
            stem = args.out / f"{ascii_stem(path)}{suffix}"
            print(f"  лист {rgb.shape[1]}x{rgb.shape[0]} px, тайл {args.tile}", flush=True)
            t = time.time()
            result = pipeline.run(rgb, path.name, dpi=300)
            elapsed = time.time() - t
            manifest = result.manifest.model_dump(mode="json")
            with_ext(stem, ".netlist.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
            with_ext(stem, ".netlist.txt").write_text(netlist_text(manifest), encoding="utf-8")

            dets = [{"bbox": c["bbox"], "category": (c.get("text_associations") or {}).get("gost_class", c["type"]),
                     "confidence": (c.get("text_associations") or {}).get("confidence", 1.0)}
                    for c in manifest.get("components", [])
                    if c.get("type") != "junction_dot"]
            vis = visualize_detections_enhanced(rgb, dets)
            cv2.imwrite(str(with_ext(stem, ".detections.png")),
                        cv2.cvtColor(nice_image(vis, args.max_side), cv2.COLOR_RGB2BGR))
            crops = zoom_crops(vis, dets, stem, args.crops) if args.crops else []

            detector = getattr(pipeline, "_last_detector", None)
            counts = Counter(d["category"] for d in dets)
            print(f"  детектор: {getattr(detector, 'backend', '?')}"
                  f"  |  модель/тайлы: {getattr(detector, 'last_stats', None)}")
            print(f"  найдено УГО: {len(dets)}  цепей: {len(manifest.get('nets', []))}  "
                  f"за {elapsed:.1f}с")
            print("  по классам: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
            for w in result.warnings:
                print(f"  ! {w}")
            for e in result.errors:
                print(f"  ОШИБКА: {e}")
            summary.append({
                "file": path.name, "page": page, "size": [rgb.shape[1], rgb.shape[0]],
                "backend": getattr(detector, "backend", None),
                "detector_stats": getattr(detector, "last_stats", None),
                "components": len(dets), "nets": len(manifest.get("nets", [])),
                "by_class": dict(counts), "time_s": round(elapsed, 1),
                "warnings": result.warnings, "errors": result.errors,
                "outputs": [str(stem.with_suffix(s)) for s in (".netlist.json", ".netlist.txt", ".detections.png")]
                           + [str(p) for p in crops],
            })

    if args.json:
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nсводка: {args.json}")
    print(f"\nрезультаты: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
