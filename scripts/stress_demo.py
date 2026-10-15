"""Стресс-набор для регрессии на демо-схеме БКС (docs/HANDOFF.md, раздел 6).

Схема из ``build_demo()`` прогоняется через пайплайн в нескольких вариантах:
масштаб, шум, размытие, JPEG, низкий контраст, PDF, большой лист.
Каждый вариант сверяется с эталоном через ``evaluate()``.

    venv\\Scripts\\python.exe scripts\\stress_demo.py                 # все варианты, настоящий OCR
    venv\\Scripts\\python.exe scripts\\stress_demo.py --fake-ocr      # без EasyOCR, быстро
    venv\\Scripts\\python.exe scripts\\stress_demo.py -v noise -v pdf  # выбранные варианты
    venv\\Scripts\\python.exe scripts\\stress_demo.py --json out.json  # отчёт в файл
    venv\\Scripts\\python.exe scripts\\stress_demo.py --list

С ``--fake-ocr`` OCR подменяется подписями из эталона, как в
``tests/test_demo_schematic.py``. Так проверяются детекция, провода и граф без
EasyOCR. Колонка обозначений в этом режиме не показательна.

Код возврата 1, если какой-то вариант хуже порога (по умолчанию 19/19
компонентов и 11/11 цепей; порог задаётся у варианта в ``VARIANTS``).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from avers.dataset.demo_schematic import build_demo, evaluate  # noqa: E402

Image = np.ndarray
Truth = Dict


def _scale_truth(truth: Truth, s: float) -> Truth:
    scaled = json.loads(json.dumps(truth))
    for comp in scaled["components"]:
        comp["bbox"] = [int(round(v * s)) for v in comp["bbox"]]
    return scaled


def _resize(image: Image, s: float, interp: int) -> Image:
    return cv2.resize(image, None, fx=s, fy=s, interpolation=interp)


# --- варианты ------------------------------------------------------------------
# Каждый вариант возвращает (изображение или путь к файлу, эталон, масштаб эталона).

def v_original(img, truth):
    return img, truth


def v_scale_075(img, truth):
    return _resize(img, 0.75, cv2.INTER_AREA), _scale_truth(truth, 0.75)


def v_scale_135(img, truth):
    return _resize(img, 1.35, cv2.INTER_CUBIC), _scale_truth(truth, 1.35)


def v_noise(img, truth):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(int) + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8), truth


def v_blur(img, truth):
    return cv2.GaussianBlur(img, (3, 3), 0.8), truth


def v_jpeg(img, truth):
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60])
    return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB), truth


def v_contrast(img, truth):
    return np.clip(0.75 * img.astype(float) + 50, 0, 255).astype(np.uint8), truth


def v_pdf(img, truth, workdir: Path):
    """Картинка в PDF: страница 1600×900 px при 150 dpi, рендер пайплайна — 300 dpi (×2)."""
    import fitz  # PyMuPDF

    h, w = img.shape[:2]
    png = workdir / "stress_demo.png"
    pdf = workdir / "stress_demo.pdf"
    cv2.imwrite(str(png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    doc = fitz.open()
    page = doc.new_page(width=w * 72 / 150, height=h * 72 / 150)
    page.insert_image(page.rect, filename=str(png))
    doc.save(str(pdf))
    doc.close()
    return pdf, _scale_truth(truth, 2.0)


def v_x3_thick(img, truth):
    return _resize(img, 3, cv2.INTER_NEAREST), _scale_truth(truth, 3)


def v_x3_thin(img, truth):
    """×3 с линиями 2 px: скелет штрихов, расширенный до 2 px.

    Простая эрозия не подходит: тонкие внутренние линии (перегородки таблицы
    разъёма, текст) исчезают раньше проводов, и проверяется уже не толщина."""
    from skimage.morphology import skeletonize

    big = _resize(img, 3, cv2.INTER_NEAREST)
    dark = (cv2.cvtColor(big, cv2.COLOR_RGB2GRAY) < 128).astype(np.uint8)
    skel = skeletonize(dark > 0).astype(np.uint8)
    ink = cv2.dilate(skel, np.ones((2, 2), np.uint8)) > 0
    # Точки соединения (ГОСТ 2.721) остаются залитыми кругами диаметром ~4 толщины
    # линии, иначе скелет превращает их в пересечение без точки (другая схема).
    # Залитые области много толще линии (ядро: расстояние до фона ≥ 10 px при
    # линии 9 px; у пересечения «+» ≈ 6,4) сохраняются уменьшенными.
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    filled = cv2.dilate((dist >= 10).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ink |= (filled > 0) & (dark > 0)
    out = np.full_like(big, 255)
    out[ink] = 0
    return out, _scale_truth(truth, 3)


@dataclass
class Variant:
    name: str
    title: str
    make: Callable
    min_components: int = 19
    min_nets: int = 11
    slow: bool = False
    needs_workdir: bool = False


VARIANTS: List[Variant] = [
    Variant("original", "исходник 1600×900", v_original),
    Variant("scale075", "×0,75", v_scale_075),
    Variant("scale135", "×1,35", v_scale_135),
    Variant("noise", "шум σ=18", v_noise),
    Variant("blur", "размытие 3×3 σ=0,8", v_blur),
    Variant("jpeg", "JPEG q60", v_jpeg),
    Variant("contrast", "низкий контраст 0,75·I+50", v_contrast),
    Variant("pdf", "PDF → рендер 300 dpi", v_pdf, needs_workdir=True),
    Variant("x3", "×3, толстые линии", v_x3_thick, slow=True),
    # До 15.10.2026: 18/19 и 2/11 (реле K1 терялось между шагами масштаба,
    # выводы не доставали до проводов из-за округления, L-углы не смыкались).
    Variant("thin", "×3, линии 2 px", v_x3_thin, slow=True),
]


# --- запуск --------------------------------------------------------------------

def _fake_ocr_labels(truth: Truth, scale_hint: float) -> List[Dict]:
    labels = []
    off_hi, off_lo, width = (int(80 * scale_hint / 3), int(30 * scale_hint / 3), int(100 * scale_hint / 3))
    for comp in truth["components"]:
        if comp["designator"].startswith("_"):
            continue
        x0, y0, _, _ = comp["bbox"]
        labels.append({"text": comp["designator"], "bbox": (x0, max(0, y0 - off_hi), x0 + width, max(1, y0 - off_lo)),
                       "confidence": 0.9})
    return labels


def _install_fake_ocr(labels: List[Dict]):
    from avers.core.validators import SchematicOCR

    orig = (SchematicOCR.load, SchematicOCR.recognize)
    SchematicOCR.load = lambda self: setattr(self, "_backend", "fake") or True
    SchematicOCR.recognize = lambda self, img: [dict(t) for t in labels]

    def restore():
        SchematicOCR.load, SchematicOCR.recognize = orig
    return restore


def run_variant(v: Variant, image: Image, truth: Truth, fake_ocr: bool, workdir: Path) -> Dict:
    from avers.core.validators import ProductionPipeline, load_and_process

    data, vtruth = v.make(image, truth, workdir) if v.needs_workdir else v.make(image, truth)
    restore = None
    if fake_ocr:
        s = vtruth["components"][0]["bbox"][2] / max(1, truth["components"][0]["bbox"][2])
        restore = _install_fake_ocr(_fake_ocr_labels(vtruth, 3 * s))
    t = time.time()
    try:
        if isinstance(data, Path):
            result = load_and_process(data)
            shape = (result.manifest.schema_metadata.height, result.manifest.schema_metadata.width)
        else:
            result = ProductionPipeline().run(data, f"stress_{v.name}.png")
            shape = data.shape[:2]
    finally:
        if restore:
            restore()
    elapsed = time.time() - t
    ev = evaluate(result.manifest.model_dump(mode="json"), vtruth)
    ok = (result.success and ev["components_found"] >= v.min_components and ev["nets_matched"] >= v.min_nets)
    return {
        "variant": v.name,
        "title": v.title,
        "size": f"{shape[1]}×{shape[0]}",
        "ok": ok,
        "success": result.success,
        "errors": result.errors,
        "time_s": round(elapsed, 1),
        "components": f"{ev['components_found']}/{ev['components_expected']}",
        "nets": f"{ev['nets_matched']}/{ev['nets_expected']}",
        "designators": f"{ev['designators_correct']}/{ev['designators_expected']}",
        "thresholds": f"{v.min_components}/{v.min_nets}",
        "component_precision": round(ev["component_precision"], 3),
        "net_precision": round(ev["net_precision"], 3),
        "missing_nets": ev["missing_nets"],
        "extra_nets": ev["extra_nets"],
        "designators_wrong": ev["designators_wrong"],
    }


def _print_table(rows: List[Dict]) -> None:
    head = f"{'вариант':10} {'размер':11} {'компоненты':>10} {'цепи':>6} {'обозн.':>7} {'время':>7}  статус"
    print(head)
    print("-" * len(head))
    for r in rows:
        status = "OK" if r["ok"] else f"FAIL (порог {r['thresholds']})"
        print(f"{r['variant']:10} {r['size']:11} {r['components']:>10} {r['nets']:>6} "
              f"{r['designators']:>7} {r['time_s']:>6.1f}с  {status}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-v", "--variant", action="append", help="имя варианта (можно несколько)")
    p.add_argument("--fast", action="store_true", help="пропустить медленные варианты (×3)")
    p.add_argument("--fake-ocr", action="store_true", help="подставные подписи из эталона вместо EasyOCR")
    p.add_argument("--json", type=Path, help="сохранить отчёт в JSON")
    p.add_argument("--details", action="store_true", help="печатать потерянные/лишние цепи и ошибки OCR")
    p.add_argument("--list", action="store_true", help="список вариантов")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # консоль Windows в cp1251/cp866: не падать на ×, σ
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.list:
        for v in VARIANTS:
            print(f"{v.name:10} {v.title}  (порог {v.min_components}/{v.min_nets}{', медленный' if v.slow else ''})")
        return 0

    known = {v.name: v for v in VARIANTS}
    if args.variant:
        unknown = [n for n in args.variant if n not in known]
        if unknown:
            p.error(f"неизвестные варианты: {', '.join(unknown)}; см. --list")
        selected = [known[n] for n in args.variant]
    else:
        selected = [v for v in VARIANTS if not (args.fast and v.slow)]

    image, truth = build_demo()
    rows: List[Dict] = []
    with tempfile.TemporaryDirectory(prefix="avers_stress_") as tmp:
        for v in selected:
            print(f"-> {v.name}: {v.title}", flush=True)
            try:
                rows.append(run_variant(v, image, truth, args.fake_ocr, Path(tmp)))
            except Exception as exc:  # отчёт должен дойти до конца
                rows.append({"variant": v.name, "title": v.title, "size": "-", "ok": False,
                             "time_s": 0.0, "components": "-", "nets": "-", "designators": "-",
                             "thresholds": f"{v.min_components}/{v.min_nets}",
                             "errors": [f"{type(exc).__name__}: {exc}"]})

    print()
    _print_table(rows)
    if args.details:
        for r in rows:
            extra = {k: r.get(k) for k in ("errors", "missing_nets", "extra_nets", "designators_wrong") if r.get(k)}
            if extra:
                print(f"\n[{r['variant']}]")
                for k, val in extra.items():
                    print(f"  {k}: {val}")
    if args.json:
        args.json.write_text(json.dumps({"fake_ocr": args.fake_ocr, "results": rows},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт: {args.json}")
    failed = [r["variant"] for r in rows if not r["ok"]]
    if failed:
        print(f"\nхуже порога: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
