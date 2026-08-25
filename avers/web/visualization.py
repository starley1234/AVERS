"""
Enhanced Visualization for AVERS - наглядная визуализация процесса.

Генерирует богатые визуализации для каждой стадии:
  - Stage 1: SAHI tiles grid with overlap highlighting
  - Stage 2: Detection with class colors, confidence, grouped components
  - Stage 3: OCR with text types, orientations, associations
  - Stage 4: Vectorization steps (binary -> skeleton -> segments -> junctions)
  - Stage 5: Interactive graph (nodes, edges, nets)
  - Stage 6: VLM arbitration with ROI crops and RAG examples
  - Diff: side-by-side original vs vectorized
  - 3D: Three.js graph

Используется для Web UI и генерации отчетов.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

from avers.core.logger import get_logger

logger = get_logger("avers.visualization")

# Color palette for GOST classes (distinct, colorblind-friendly)
GOST_COLORS = {
    "connector_body": (108, 92, 255),
    "pin": (234, 179, 8),
    "junction_dot": (239, 68, 68),
    "ground": (59, 130, 246),
    "shield": (139, 92, 246),
    "offpage_connector": (236, 72, 153),
    "diode": (249, 115, 22),
    "relay": (6, 182, 212),
    "resistor": (132, 204, 22),
    "capacitor": (168, 85, 247),
    "wire": (100, 100, 100),
    "text": (59, 130, 246),
}


def visualize_sahi_tiles(
    image: np.ndarray,
    tile_size: int = 1024,
    overlap_ratio: float = 0.2,
    show_overlaps: bool = True,
    show_ids: bool = True,
) -> np.ndarray:
    """Visualize SAHI tiling - наглядно показывает нарезку."""
    from avers.stages.stage1_slicing import SlicingEngine

    engine = SlicingEngine(tile_size=tile_size, overlap_ratio=overlap_ratio)
    tiles = list(engine.generate_tiles(image))

    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    else:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 150, 50), (150, 50, 255),
    ]

    for i, tile in enumerate(tiles):
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (tile.x_min, tile.y_min), (tile.x_max, tile.y_max), color, 2)

        if show_overlaps and (tile.overlap_left or tile.overlap_top):
            overlay = vis.copy()
            if tile.overlap_left > 0:
                cv2.rectangle(overlay, (tile.x_min, tile.y_min),
                              (tile.x_min + tile.overlap_left, tile.y_max), (0, 255, 0), -1)
            if tile.overlap_top > 0:
                cv2.rectangle(overlay, (tile.x_min, tile.y_min),
                              (tile.x_max, tile.y_min + tile.overlap_top), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

        if show_ids:
            cv2.putText(vis, f"T{tile.tile_id}", (tile.x_min + 8, tile.y_min + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis, f"{tile.width}x{tile.height}", (tile.x_min + 8, tile.y_min + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    info_text = f"SAHI: {len(tiles)} tiles, {tile_size}x{tile_size}, overlap {overlap_ratio*100:.0f}%"
    cv2.putText(vis, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(vis, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def visualize_detections_enhanced(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_confidence: bool = True,
    show_class: bool = True,
    group_components: bool = True,
) -> np.ndarray:
    """Enhanced detection visualization with grouping and colors."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    by_class = {}
    for det in detections:
        cls = det.get("category", "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("confidence", 1.0)
        cls = det.get("category", "unknown")

        color_rgb = GOST_COLORS.get(cls, (128, 128, 128))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        thickness = 2 if conf > 0.7 else 1
        if conf < 0.5:
            color_bgr = tuple(int(c * 0.6) for c in color_bgr)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, thickness)

        label = ""
        if show_class:
            label += cls
        if show_confidence:
            label += f" {conf:.2f}" if label else f"{conf:.2f}"

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color_bgr, -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx, cy), 2, color_bgr, -1)

    legend_y = 20
    for cls, count in by_class.items():
        color_rgb = GOST_COLORS.get(cls, (128, 128, 128))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(vis, (vis.shape[1] - 200, legend_y),
                      (vis.shape[1] - 180, legend_y + 16), color_bgr, -1)
        cv2.putText(vis, f"{cls}: {count}", (vis.shape[1] - 170, legend_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 20

    return vis


def visualize_ocr_enhanced(
    image: np.ndarray,
    texts: List[Dict[str, Any]],
    show_types: bool = True,
) -> np.ndarray:
    """Enhanced OCR visualization with text types."""
    vis = image.copy()

    type_colors = {
        "connector": (108, 92, 255),
        "pin": (234, 179, 8),
        "wire_type": (34, 197, 94),
        "voltage": (239, 68, 68),
        "gnd": (59, 130, 246),
        "designator": (249, 115, 22),
        "other": (128, 128, 128),
    }

    for t in texts:
        x1, y1, x2, y2 = t["bbox"]
        text = t.get("text", "")
        conf = t.get("confidence", 1.0)
        text_type = t.get("type", "other")

        color_rgb = type_colors.get(text_type, (128, 128, 128))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 1)

        label = f"{text} ({text_type})" if show_types else text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if conf < 0.65:
            cv2.circle(vis, (x2 - 5, y1 + 5), 4, (239, 68, 68), -1)

    return vis


def visualize_vectorization_steps(
    image: np.ndarray,
    exclusion_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Dict[str, np.ndarray]:
    """Visualize all steps of vectorization."""
    from avers.core.validators import WireVectorizer

    vis_steps = {}

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    else:
        gray = image.copy()
        image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    vis_steps["binary"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    if exclusion_bboxes:
        mask = np.ones(gray.shape[:2], dtype=np.uint8) * 255
        for bbox in exclusion_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(mask, (x1-5, y1-5), (x2+5, y2+5), 0, -1)
        masked = cv2.bitwise_and(thresh, thresh, mask=mask)
        vis_steps["masked"] = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
        thresh_for_skeleton = masked
    else:
        vis_steps["masked"] = vis_steps["binary"]
        thresh_for_skeleton = thresh

    from skimage import morphology
    skeleton = morphology.skeletonize((thresh_for_skeleton > 0).astype(np.uint8))
    skeleton_vis = (skeleton * 255).astype(np.uint8)
    vis_steps["skeleton"] = cv2.cvtColor(skeleton_vis, cv2.COLOR_GRAY2RGB)

    vectorizer = WireVectorizer()
    segments, junctions = vectorizer.vectorize(image, exclusion_bboxes)

    seg_vis = np.ones((gray.shape[0], gray.shape[1], 3), dtype=np.uint8) * 255
    for seg in segments:
        cv2.line(seg_vis, seg.start, seg.end, (100, 100, 100), 2)
    vis_steps["segments"] = seg_vis

    junc_vis = seg_vis.copy()
    for j in junctions:
        cv2.circle(junc_vis, j, 5, (239, 68, 68), -1)
        cv2.putText(junc_vis, f"{len(junctions)}", (j[0]+8, j[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (239, 68, 68), 1)
    vis_steps["junctions"] = junc_vis

    final = image.copy()
    if len(final.shape) == 2:
        final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    for seg in segments:
        cv2.line(final, seg.start, seg.end, (100, 100, 100), 2)
    for j in junctions:
        cv2.circle(final, j, 6, (239, 68, 68), -1)
        cv2.circle(final, j, 8, (239, 68, 68), 1)

    vis_steps["final"] = final

    return vis_steps


def visualize_graph_interactive(
    components: List[Any],
    nets: List[Any],
    width: int = 1000,
    height: int = 800,
) -> Dict[str, Any]:
    """Generate interactive graph data for D3/vis-network."""
    nodes = []
    edges = []

    for comp in components:
        if isinstance(comp, dict):
            comp_id = comp.get("id", "")
            designator = comp.get("designator", comp_id)
            bbox = comp.get("bbox", (0,0,0,0))
            comp_type = comp.get("type", "unknown")
            pins = comp.get("pins", [])
        else:
            comp_id = getattr(comp, "id", "")
            designator = getattr(comp, "designator", comp_id)
            bbox = getattr(comp, "bbox", (0,0,0,0))
            comp_type = getattr(comp, "type", "unknown")
            if hasattr(comp_type, "value"):
                comp_type = comp_type.value
            pins = getattr(comp, "pins", [])

        x = (bbox[0] + bbox[2]) // 2 if len(bbox) >= 4 else 0
        y = (bbox[1] + bbox[3]) // 2 if len(bbox) >= 4 else 0

        nodes.append({
            "id": comp_id,
            "label": designator,
            "type": "component",
            "component_type": comp_type,
            "x": x,
            "y": y,
            "bbox": bbox,
            "color": f"rgb{GOST_COLORS.get(comp_type, (128,128,128))}",
            "size": 20,
        })

        for pin in pins:
            if isinstance(pin, dict):
                pin_num = pin.get("pin_number", "")
                coord = pin.get("coord", (0,0))
            else:
                pin_num = getattr(pin, "pin_number", "")
                coord = getattr(pin, "coord", (0,0))

            nodes.append({
                "id": f"{comp_id}_pin_{pin_num}",
                "label": f"{designator}:{pin_num}",
                "type": "pin",
                "component_id": comp_id,
                "pin_number": pin_num,
                "x": coord[0],
                "y": coord[1],
                "color": "rgb(234, 179, 8)",
                "size": 8,
            })

            edges.append({
                "from": comp_id,
                "to": f"{comp_id}_pin_{pin_num}",
                "type": "has_pin",
                "color": "rgba(200,200,200,0.5)",
                "dashes": True,
            })

    for net in nets:
        if isinstance(net, dict):
            net_id = net.get("net_id", "")
            connections = net.get("connections", [])
            path_points = net.get("path_points", [])
        else:
            net_id = getattr(net, "net_id", "")
            connections = getattr(net, "connections", [])
            path_points = getattr(net, "path_points", [])

        pin_ids = []
        for conn in connections:
            if isinstance(conn, dict):
                comp_id = conn.get("component_id", "")
                pin = conn.get("pin", "")
            else:
                comp_id = getattr(conn, "component_id", "")
                pin = getattr(conn, "pin", "")
            pin_ids.append(f"{comp_id}_pin_{pin}")

        for i in range(len(pin_ids)):
            for j in range(i+1, len(pin_ids)):
                edges.append({
                    "from": pin_ids[i],
                    "to": pin_ids[j],
                    "type": "net",
                    "net_id": net_id,
                    "label": net_id,
                    "color": "rgb(239, 68, 68)",
                    "width": 2,
                    "path_points": path_points,
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "components": len([n for n in nodes if n["type"] == "component"]),
            "pins": len([n for n in nodes if n["type"] == "pin"]),
            "nets": len(nets),
            "connections": len([e for e in edges if e["type"] == "net"]),
        }
    }


def visualize_vlm_arbitration(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    roi_size: int = 256,
    rag_examples: Optional[List[Dict]] = None,
) -> Dict[str, np.ndarray]:
    """Visualize VLM arbitration."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    cx, cy = (x1+x2)//2, (y1+y2)//2

    expand = 1.5
    bw, bh = (x2-x1), (y2-y1)
    new_w, new_h = int(bw * expand), int(bh * expand)

    x1_exp = max(0, cx - new_w//2)
    y1_exp = max(0, cy - new_h//2)
    x2_exp = min(w, cx + new_w//2)
    y2_exp = min(h, cy + new_h//2)

    context = image[y1_exp:y2_exp, x1_exp:x2_exp]
    roi = image[max(0, cy-roi_size//2):min(h, cy+roi_size//2),
                max(0, cx-roi_size//2):min(w, cx+roi_size//2)]
    roi_resized = cv2.resize(roi, (roi_size, roi_size))

    context_vis = context.copy()
    local_x1, local_y1 = x1 - x1_exp, y1 - y1_exp
    local_x2, local_y2 = x2 - x1_exp, y2 - y1_exp
    cv2.rectangle(context_vis, (local_x1, local_y1), (local_x2, local_y2), (239, 68, 68), 2)

    ch_x, ch_y = context_vis.shape[1]//2, context_vis.shape[0]//2
    cv2.line(context_vis, (ch_x-20, ch_y), (ch_x+20, ch_y), (239, 68, 68), 1)
    cv2.line(context_vis, (ch_x, ch_y-20), (ch_x, ch_y+20), (239, 68, 68), 1)

    result = {
        "context": context_vis,
        "roi": roi_resized,
        "original_bbox": np.array([[x1, y1], [x2, y2]]),
    }

    if rag_examples:
        grid_size = 256
        num_examples = min(len(rag_examples), 4)
        grid_width = grid_size * (num_examples + 1)
        grid_height = grid_size
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        grid[0:grid_size, 0:grid_size] = roi_resized
        for i, ex in enumerate(rag_examples[:num_examples]):
            x_offset = (i+1) * grid_size
            ex_img = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 240
            cv2.putText(ex_img, ex.get("label", "example")[:15], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(ex_img, f"score: {ex.get('score', 0):.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
            grid[0:grid_size, x_offset:x_offset+grid_size] = ex_img
        result["rag_comparison"] = grid

    return result


def create_pipeline_report(
    image: np.ndarray,
    manifest: Any,
    stage_timings: Dict[str, float],
    output_path: Path,
) -> Path:
    """Create HTML report with all stages visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from avers.core.validators import SlicedDetector, SchematicOCR

    tiles_vis = visualize_sahi_tiles(image, tile_size=1024, overlap_ratio=0.2)

    detector = SlicedDetector(device="cpu")
    detections = detector.detect(image)
    det_vis = visualize_detections_enhanced(image, detections)

    ocr = SchematicOCR()
    texts = ocr.recognize(image)
    ocr_vis = visualize_ocr_enhanced(image, texts)

    comp_bboxes = []
    comps = manifest.components if hasattr(manifest, 'components') else []
    for c in comps:
        if hasattr(c, 'bbox'):
            comp_bboxes.append(c.bbox)
        elif isinstance(c, dict):
            comp_bboxes.append(c.get('bbox', (0,0,0,0)))

    text_bboxes = [t["bbox"] for t in texts]
    vec_steps = visualize_vectorization_steps(image, comp_bboxes + text_bboxes)

    graph_data = visualize_graph_interactive(
        manifest.components if hasattr(manifest, 'components') else [],
        manifest.nets if hasattr(manifest, 'nets') else []
    )

    report_dir = output_path.parent / f"{output_path.stem}_assets"
    report_dir.mkdir(exist_ok=True)

    def save_img(name, img):
        path = report_dir / f"{name}.jpg"
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
        cv2.imwrite(str(path), bgr)
        return path.name

    assets = {}
    assets["tiles"] = save_img("01_tiles", tiles_vis)
    assets["detections"] = save_img("02_detections", det_vis)
    assets["ocr"] = save_img("03_ocr", ocr_vis)
    for step_name, step_img in vec_steps.items():
        assets[f"vec_{step_name}"] = save_img(f"04_vec_{step_name}", step_img)

    # Precompute text for HTML to avoid backslash in f-string
    det_lines = []
    for d in detections[:10]:
        cat = d.get('category', 'unknown')
        bbox = d.get('bbox', (0,0,0,0))
        conf = d.get('confidence', 0)
        det_lines.append(f"  {cat} {bbox} conf={conf:.2f}")
    det_text = "\n".join(det_lines)

    ocr_lines = []
    for t in texts[:10]:
        txt = t.get('text', '')
        bbox = t.get('bbox', (0,0,0,0))
        conf = t.get('confidence', 0)
        ocr_lines.append(f"  '{txt}' {bbox} conf={conf:.2f}")
    ocr_text = "\n".join(ocr_lines)

    comp_lines = []
    for c in comps[:10]:
        if hasattr(c, 'id'):
            cid = c.id
            des = getattr(c, 'designator', cid)
            bbox = getattr(c, 'bbox', (0,0,0,0))
        else:
            cid = c.get('id', '')
            des = c.get('designator', cid)
            bbox = c.get('bbox', (0,0,0,0))
        comp_lines.append(f"  {cid}: {des} {bbox}")
    comp_text = "\n".join(comp_lines)

    issues = manifest.human_review_required if hasattr(manifest, 'human_review_required') else []
    issue_lines = []
    for iss in issues[:10]:
        if hasattr(iss, 'issue_type'):
            itype = iss.issue_type
            bbox = getattr(iss, 'bbox', (0,0,0,0))
            desc = getattr(iss, 'description', '')
        else:
            itype = iss.get('issue_type', '')
            bbox = iss.get('bbox', (0,0,0,0))
            desc = iss.get('description', '')
        issue_lines.append(f"  {itype}: {bbox} - {desc}")
    issue_text = "\n".join(issue_lines)

    source_file = getattr(getattr(manifest, 'schema_metadata', None), 'source_file', 'unknown')
    width = getattr(getattr(manifest, 'schema_metadata', None), 'width', image.shape[1])
    height = getattr(getattr(manifest, 'schema_metadata', None), 'height', image.shape[0])
    num_comps = len(comps)
    num_nets = len(manifest.nets if hasattr(manifest, 'nets') else [])
    num_issues = len(issues)

    # Timings HTML
    timings_html = ""
    for stage_name, t in stage_timings.items():
        timings_html += f"<div class=\"metric\"><b>{stage_name}</b><br>{t*1000:.1f}ms</div>"

    # Manifest dump
    try:
        manifest_str = str(manifest.model_dump() if hasattr(manifest, 'model_dump') else manifest)[:2000]
    except Exception:
        manifest_str = "manifest dump failed"

    html = f"""
<!DOCTYPE html>
<html lang=\"ru\">
<head>
<meta charset=\"UTF-8\">
<title>АВЕРС Отчет - {source_file}</title>
<style>
body {{ font-family: Inter, sans-serif; background: #0a0a0b; color: #e8e8ec; margin: 0; padding: 20px; }}
.header {{ background: #141416; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a30; }}
h1 {{ margin: 0; font-size: 24px; }}
h2 {{ color: #6c5cff; border-bottom: 1px solid #2a2a30; padding-bottom: 8px; }}
.stage {{ background: #141416; border: 1px solid #2a2a30; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
.stage-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
img {{ max-width: 100%; border-radius: 8px; border: 1px solid #2a2a30; }}
.metrics {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.metric {{ background: #1c1c1f; padding: 12px 16px; border-radius: 8px; border: 1px solid #2a2a30; }}
.metric b {{ color: #6c5cff; }}
pre {{ background: #0a0a0b; padding: 12px; border-radius: 8px; overflow: auto; font-size: 12px; }}
</style>
</head>
<body>
<div class=\"header\">
<h1>АВЕРС — Отчет обработки</h1>
<p>Файл: {source_file} | Размер: {width}x{height} | Компонентов: {num_comps} | Цепей: {num_nets} | Проблем: {num_issues}</p>
<div class=\"metrics\">
{timings_html}
</div>
</div>

<div class=\"stage\">
<h2>Стадия 1: SAHI Нарезка</h2>
<p>Проблема: ресайз 14000×3500 до 640×640 уничтожает тонкие линии. Решение: нарезка на тайлы 1024×1024 с overlap 0.2.</p>
<img src=\"{report_dir.name}/{assets['tiles']}\" alt=\"SAHI tiles\">
</div>

<div class=\"stage\">
<h2>Стадия 2: Детекция УГО (RT-DETR/YOLO)</h2>
<p>Детекция 10 классов ГОСТ: connector_body, pin, junction_dot, ground, shield, offpage, diode, relay, resistor, capacitor.</p>
<img src=\"{report_dir.name}/{assets['detections']}\" alt=\"Detections\">
<pre>Найдено: {len(detections)} объектов
{det_text}
</pre>
</div>

<div class=\"stage\">
<h2>Стадия 3: OCR текста (PaddleOCR)</h2>
<p>Детекция под 0°/90°/270°, regex валидация: Х1, Ш1, СНЦ144, пины, БПВЛ, +27В.</p>
<img src=\"{report_dir.name}/{assets['ocr']}\" alt=\"OCR\">
<pre>Распознано: {len(texts)} текстов
{ocr_text}
</pre>
</div>

<div class=\"stage\">
<h2>Стадия 4: Векторизация (OpenCV)</h2>
<p>1. Маска исключений → 2. Скелетизация Guo-Hall → 3. Hough + RDP → 4. Junction detection.</p>
<div class=\"stage-grid\">
<div><h3>Binary</h3><img src=\"{report_dir.name}/{assets.get('vec_binary', '')}\" alt=\"binary\"></div>
<div><h3>Masked</h3><img src=\"{report_dir.name}/{assets.get('vec_masked', '')}\" alt=\"masked\"></div>
<div><h3>Skeleton (1px)</h3><img src=\"{report_dir.name}/{assets.get('vec_skeleton', '')}\" alt=\"skeleton\"></div>
<div><h3>Segments</h3><img src=\"{report_dir.name}/{assets.get('vec_segments', '')}\" alt=\"segments\"></div>
<div><h3>Junctions</h3><img src=\"{report_dir.name}/{assets.get('vec_junctions', '')}\" alt=\"junctions\"></div>
<div><h3>Final overlay</h3><img src=\"{report_dir.name}/{assets.get('vec_final', '')}\" alt=\"final\"></div>
</div>
</div>

<div class=\"stage\">
<h2>Стадия 5: Графовый синтез (NetworkX)</h2>
<p>Snapping R=15px, k-d tree текст, схлопывание цепочек, T-узлы.</p>
<pre>Граф: {graph_data['stats']['components']} компонентов, {graph_data['stats']['pins']} пинов, {graph_data['stats']['nets']} цепей
Компоненты:
{comp_text}
</pre>
</div>

<div class=\"stage\">
<h2>Стадия 6: VLM-арбитраж (Qwen-VL + Vision RAG)</h2>
<p>ROI 256×256 при коллизии: confidence<0.65, висящие линии, спорный перекресток. RAG: CLIP 512d + FAISS Top-K.</p>
<pre>Проблем для проверки: {num_issues}
{issue_text}
</pre>
</div>

<div class=\"stage\">
<h2>Выходной Netlist JSON</h2>
<pre>{manifest_str}...</pre>
</div>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Pipeline report saved to {output_path}")
    return output_path


# =============================================================================
# Side-by-side diff and 3D visualization
# =============================================================================

def visualize_diff_original_vs_vectorized(
    original: np.ndarray,
    vectorized_overlay: Optional[np.ndarray] = None,
    exclusion_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    mode: str = "overlay",
) -> Dict[str, np.ndarray]:
    """Side-by-side diff оригинал vs векторизованный."""
    if len(original.shape) == 2:
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = original.copy()
        if original.shape[2] == 4:
            orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGRA2RGB)
        elif original.shape[2] == 3:
            # Assume BGR from cv2, convert to RGB
            # Check if looks like BGR (if first pixel white, both same)
            # For safety, keep as is if already RGB-like
            # We will treat as RGB for visualization
            pass

    if vectorized_overlay is None:
        from avers.core.validators import WireVectorizer
        vec = WireVectorizer()
        segments, junctions = vec.vectorize(original, exclusion_bboxes)

        vec_img = np.ones_like(orig_rgb) * 255
        for seg in segments:
            cv2.line(vec_img, seg.start, seg.end, (0, 0, 0), 2)
        for j in junctions:
            cv2.circle(vec_img, j, 5, (0, 0, 255), -1)

        overlay = orig_rgb.copy()
        for seg in segments:
            cv2.line(overlay, seg.start, seg.end, (255, 0, 0), 2)
        for j in junctions:
            cv2.circle(overlay, j, 6, (0, 0, 255), -1)
    else:
        vec_img = vectorized_overlay
        overlay = vectorized_overlay

    results = {}
    h, w = orig_rgb.shape[:2]

    if vec_img.shape[:2] != orig_rgb.shape[:2]:
        vec_img_resized = cv2.resize(vec_img, (w, h))
    else:
        vec_img_resized = vec_img

    # Side by side
    side_by_side = np.hstack([orig_rgb, vec_img_resized])
    cv2.line(side_by_side, (w, 0), (w, h), (108, 92, 255), 3)
    cv2.putText(side_by_side, "ORIGINAL (raster)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(side_by_side, "ORIGINAL (raster)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(side_by_side, "VECTORIZED", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(side_by_side, "VECTORIZED", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    results["side_by_side"] = side_by_side

    if len(orig_rgb.shape) == 3 and len(vec_img_resized.shape) == 3:
        overlay_blend = cv2.addWeighted(orig_rgb, 0.7, vec_img_resized, 0.3, 0)
        results["overlay"] = overlay_blend

        orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY) if len(orig_rgb.shape) == 3 else orig_rgb
        vec_gray = cv2.cvtColor(vec_img_resized, cv2.COLOR_RGB2GRAY) if len(vec_img_resized.shape) == 3 else vec_img_resized

        diff = cv2.absdiff(orig_gray, vec_gray)
        _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff_color = np.zeros((h, w, 3), dtype=np.uint8)
        diff_color[:, :, 1] = 255 - diff_thresh
        diff_color[:, :, 2] = diff_thresh
        diff_vis = cv2.addWeighted(orig_rgb, 0.3, diff_color, 0.7, 0)
        results["difference"] = diff_vis

        blend = orig_rgb.copy()
        blend_width = w // 4
        center = w // 2
        for x in range(w):
            if x < center - blend_width//2:
                blend[:, x] = orig_rgb[:, x]
            elif x > center + blend_width//2:
                blend[:, x] = vec_img_resized[:, x]
            else:
                alpha = (x - (center - blend_width//2)) / blend_width
                blend[:, x] = (orig_rgb[:, x] * (1-alpha) + vec_img_resized[:, x] * alpha).astype(np.uint8)

        cv2.line(blend, (center, 0), (center, h), (108, 92, 255), 2)
        cv2.circle(blend, (center, h//2), 12, (108, 92, 255), -1)
        cv2.putText(blend, "<", (center-6, h//2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        results["blend_slider"] = blend

        checker = orig_rgb.copy()
        checker_size = 64
        for y in range(0, h, checker_size):
            for x in range(0, w, checker_size):
                if (x//checker_size + y//checker_size) % 2 == 1:
                    x2 = min(x+checker_size, w)
                    y2 = min(y+checker_size, h)
                    checker[y:y2, x:x2] = vec_img_resized[y:y2, x:x2]
        results["checker"] = checker

    final_overlay = orig_rgb.copy()
    if exclusion_bboxes:
        for bbox in exclusion_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (108, 92, 255), 2)

    results["final_overlay"] = overlay if 'overlay' in locals() and isinstance(overlay, np.ndarray) and len(overlay.shape) == 3 else final_overlay

    return results


def visualize_graph_3d(
    components: List[Any],
    nets: List[Any],
    width: int = 1000,
    height: int = 800,
) -> Dict[str, Any]:
    """Generate 3D graph data for Three.js visualization."""
    import math
    import random

    nodes = []
    edges = []
    node_map = {}

    def get_comp_data(comp):
        if isinstance(comp, dict):
            return {
                "id": comp.get("id", ""),
                "designator": comp.get("designator", ""),
                "bbox": comp.get("bbox", (0,0,0,0)),
                "type": comp.get("type", "unknown"),
                "pins": comp.get("pins", []),
            }
        else:
            comp_type = getattr(comp, "type", "unknown")
            if hasattr(comp_type, "value"):
                comp_type = comp_type.value
            return {
                "id": getattr(comp, "id", ""),
                "designator": getattr(comp, "designator", ""),
                "bbox": getattr(comp, "bbox", (0,0,0,0)),
                "type": comp_type,
                "pins": getattr(comp, "pins", []),
            }

    for comp in components:
        data = get_comp_data(comp)
        comp_id = data["id"]
        bbox = data["bbox"]
        x = (bbox[0] + bbox[2]) // 2 if len(bbox) >= 4 else random.randint(0, width)
        y = (bbox[1] + bbox[3]) // 2 if len(bbox) >= 4 else random.randint(0, height)

        if data["type"] == "connector":
            z = -20 if x < width // 2 else 20
        elif data["type"] in ("resistor", "diode", "relay", "capacitor"):
            z = 50 + random.randint(-10, 10)
        elif data["type"] in ("ground", "shield"):
            z = -30
        else:
            z = 30

        color_rgb = GOST_COLORS.get(data["type"], (128, 128, 128))

        node = {
            "id": comp_id,
            "label": data["designator"],
            "type": "component",
            "component_type": data["type"],
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "bbox": bbox,
            "color": f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})",
            "color_hex": f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}",
            "size": 12 if data["type"] == "connector" else 8,
            "shape": "box" if data["type"] == "connector" else "sphere",
        }
        nodes.append(node)
        node_map[comp_id] = node

        for pin in data["pins"]:
            if isinstance(pin, dict):
                pin_num = pin.get("pin_number", "")
                coord = pin.get("coord", (0,0))
            else:
                pin_num = getattr(pin, "pin_number", "")
                coord = getattr(pin, "coord", (0,0))

            pin_id = f"{comp_id}_pin_{pin_num}"
            pin_node = {
                "id": pin_id,
                "label": f"{data['designator']}:{pin_num}",
                "type": "pin",
                "component_id": comp_id,
                "pin_number": pin_num,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(z + 5),
                "color": "rgb(234, 179, 8)",
                "color_hex": "#eab308",
                "size": 3,
                "shape": "sphere",
            }
            nodes.append(pin_node)
            node_map[pin_id] = pin_node

            edges.append({
                "id": f"edge_{comp_id}_{pin_id}",
                "from": comp_id,
                "to": pin_id,
                "type": "has_pin",
                "color": "rgba(200,200,200,0.5)",
                "width": 1,
                "dashes": True,
            })

    for net in nets:
        if isinstance(net, dict):
            net_id = net.get("net_id", "")
            connections = net.get("connections", [])
            path_points = net.get("path_points", [])
        else:
            net_id = getattr(net, "net_id", "")
            connections = getattr(net, "connections", [])
            path_points = getattr(net, "path_points", [])

        pin_ids = []
        for conn in connections:
            if isinstance(conn, dict):
                comp_id = conn.get("component_id", "")
                pin = conn.get("pin", "")
            else:
                comp_id = getattr(conn, "component_id", "")
                pin = getattr(conn, "pin", "")
            pin_id = f"{comp_id}_pin_{pin}"
            if pin_id in node_map:
                pin_ids.append(pin_id)

        if path_points and len(path_points) > 1:
            prev_node_id = None
            for i, (x, y) in enumerate(path_points):
                z = 10 + math.sin(i * 0.5) * 10 + random.uniform(-2, 2)

                wire_node_id = f"{net_id}_wire_{i}"
                wire_node = {
                    "id": wire_node_id,
                    "label": "",
                    "type": "wire",
                    "net_id": net_id,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "color": "rgb(100, 100, 100)",
                    "color_hex": "#646464",
                    "size": 1,
                    "shape": "sphere",
                    "opacity": 0.6,
                }
                nodes.append(wire_node)
                node_map[wire_node_id] = wire_node

                if prev_node_id:
                    edges.append({
                        "id": f"edge_{prev_node_id}_{wire_node_id}",
                        "from": prev_node_id,
                        "to": wire_node_id,
                        "type": "wire",
                        "net_id": net_id,
                        "color": "rgb(239, 68, 68)",
                        "width": 2,
                    })
                prev_node_id = wire_node_id

            if pin_ids and prev_node_id:
                if len(path_points) > 0:
                    first_wire = f"{net_id}_wire_0"
                    last_wire = f"{net_id}_wire_{len(path_points)-1}"
                    if first_wire in node_map and pin_ids:
                        edges.append({
                            "id": f"edge_{pin_ids[0]}_{first_wire}",
                            "from": pin_ids[0],
                            "to": first_wire,
                            "type": "net",
                            "net_id": net_id,
                            "color": "rgb(239, 68, 68)",
                            "width": 3,
                        })
                    if last_wire in node_map and len(pin_ids) > 1:
                        edges.append({
                            "id": f"edge_{last_wire}_{pin_ids[-1]}",
                            "from": last_wire,
                            "to": pin_ids[-1],
                            "type": "net",
                            "net_id": net_id,
                            "color": "rgb(239, 68, 68)",
                            "width": 3,
                        })
        else:
            for i in range(len(pin_ids)):
                for j in range(i+1, len(pin_ids)):
                    edges.append({
                        "id": f"edge_{pin_ids[i]}_{pin_ids[j]}_{net_id}",
                        "from": pin_ids[i],
                        "to": pin_ids[j],
                        "type": "net",
                        "net_id": net_id,
                        "label": net_id,
                        "color": "rgb(239, 68, 68)",
                        "width": 2,
                    })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "components": len([n for n in nodes if n["type"] == "component"]),
            "pins": len([n for n in nodes if n["type"] == "pin"]),
            "wires": len([n for n in nodes if n["type"] == "wire"]),
            "nets": len(nets),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
        "metadata": {
            "width": width,
            "height": height,
            "coordinate_system": "image_coords with Z layers",
            "z_layers": {
                "connectors": "Z=-20 to 20 (left/right)",
                "discrete": "Z=50 (resistors, diodes, etc)",
                "ground": "Z=-30",
                "wires": "Z=10 +/- sin wave for routing visualization",
            }
        }
    }


def create_diff_visualization_report(
    original: np.ndarray,
    manifest: Any,
    output_path: Path,
) -> Path:
    """Create HTML report for side-by-side diff visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comp_bboxes = []
    comps = manifest.components if hasattr(manifest, 'components') else []
    for c in comps:
        if hasattr(c, 'bbox'):
            comp_bboxes.append(c.bbox)
        elif isinstance(c, dict):
            comp_bboxes.append(c.get('bbox', (0,0,0,0)))

    diffs = visualize_diff_original_vs_vectorized(original, exclusion_bboxes=comp_bboxes)

    report_dir = output_path.parent / f"{output_path.stem}_diff_assets"
    report_dir.mkdir(exist_ok=True)

    def save_img(name, img):
        path = report_dir / f"{name}.jpg"
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 and img.shape[2] == 3 else img
        cv2.imwrite(str(path), bgr)
        return path.name

    assets = {}
    for mode, img in diffs.items():
        assets[mode] = save_img(f"diff_{mode}", img)

    html = f"""
<!DOCTYPE html>
<html lang=\"ru\">
<head>
<meta charset=\"UTF-8\">
<title>АВЕРС Diff Отчет - Side-by-side</title>
<style>
body {{ font-family: Inter, sans-serif; background: #0a0a0b; color: #e8e8ec; margin: 0; padding: 20px; }}
.header {{ background: #141416; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a30; }}
h1 {{ margin: 0; font-size: 24px; }}
h2 {{ color: #6c5cff; }}
.diff-container {{ background: #141416; border: 1px solid #2a2a30; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
img {{ max-width: 100%; border-radius: 8px; border: 1px solid #2a2a30; }}
</style>
</head>
<body>
<div class=\"header\">
<h1>АВЕРС — Side-by-side Diff Отчет</h1>
<p>Оригинал vs Векторизованный — наглядное сравнение</p>
</div>

<div class=\"diff-container\">
<h2>Side-by-side (Рядом)</h2>
<img src=\"{report_dir.name}/{assets.get('side_by_side', '')}\" alt=\"side by side\">
</div>

<div class=\"diff-container\">
<h2>Overlay (Наложение 70/30)</h2>
<img src=\"{report_dir.name}/{assets.get('overlay', '')}\" alt=\"overlay\">
</div>

<div class=\"diff-container\">
<h2>Difference (Разница)</h2>
<p>Зеленый — совпадает, красный — отличается</p>
<img src=\"{report_dir.name}/{assets.get('difference', '')}\" alt=\"difference\">
</div>

<div class=\"diff-container\">
<h2>Blend Slider (Слайдер)</h2>
<img src=\"{report_dir.name}/{assets.get('blend_slider', '')}\" alt=\"blend slider\">
</div>

<div class=\"diff-container\">
<h2>Checkerboard (Шахматка)</h2>
<img src=\"{report_dir.name}/{assets.get('checker', '')}\" alt=\"checker\">
</div>

<div class=\"diff-container\">
<h2>Final Overlay</h2>
<img src=\"{report_dir.name}/{assets.get('final_overlay', '')}\" alt=\"final overlay\">
</div>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Diff report saved to {output_path}")
    return output_path
