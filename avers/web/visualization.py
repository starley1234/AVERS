"""
Enhanced Visualization for AVERS - наглядная визуализация процесса.

Генерирует богатые визуализации для каждой стадии:
  - Stage 1: SAHI tiles grid with overlap highlighting
  - Stage 2: Detection with class colors, confidence, grouped components
  - Stage 3: OCR with text types, orientations, associations
  - Stage 4: Vectorization steps (binary -> skeleton -> segments -> junctions)
  - Stage 5: Interactive graph (nodes, edges, nets)
  - Stage 6: VLM arbitration with ROI crops and RAG examples

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
    "connector_body": (108, 92, 255),    # Purple
    "pin": (234, 179, 8),                # Yellow
    "junction_dot": (239, 68, 68),       # Red
    "ground": (59, 130, 246),            # Blue
    "shield": (139, 92, 246),            # Violet
    "offpage_connector": (236, 72, 153), # Pink
    "diode": (249, 115, 22),             # Orange
    "relay": (6, 182, 212),              # Cyan
    "resistor": (132, 204, 22),          # Lime
    "capacitor": (168, 85, 247),         # Purple light
    "wire": (100, 100, 100),             # Gray
    "text": (59, 130, 246),              # Blue
}


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex to BGR."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


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
    
    # Draw tiles with distinct colors
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 150, 50), (150, 50, 255),
    ]
    
    for i, tile in enumerate(tiles):
        color = colors[i % len(colors)]
        
        # Tile border
        cv2.rectangle(vis, (tile.x_min, tile.y_min), (tile.x_max, tile.y_max), color, 2)
        
        # Overlap regions (semi-transparent)
        if show_overlaps and (tile.overlap_left or tile.overlap_top):
            overlay = vis.copy()
            if tile.overlap_left > 0:
                cv2.rectangle(overlay, (tile.x_min, tile.y_min), 
                             (tile.x_min + tile.overlap_left, tile.y_max), (0, 255, 0), -1)
            if tile.overlap_top > 0:
                cv2.rectangle(overlay, (tile.x_min, tile.y_min),
                             (tile.x_max, tile.y_min + tile.overlap_top), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
        
        # Tile ID
        if show_ids:
            cv2.putText(vis, f"T{tile.tile_id}", (tile.x_min + 8, tile.y_min + 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis, f"{tile.width}x{tile.height}", (tile.x_min + 8, tile.y_min + 44),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    
    # Info overlay
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
    
    # Group by class for legend
    by_class = {}
    for det in detections:
        cls = det.get("category", "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("confidence", 1.0)
        cls = det.get("category", "unknown")
        
        # Color by class
        color_rgb = GOST_COLORS.get(cls, (128, 128, 128))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Confidence affects thickness/opacity
        thickness = 2 if conf > 0.7 else 1
        if conf < 0.5:
            color_bgr = tuple(int(c * 0.6) for c in color_bgr)  # Dim low conf
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, thickness)
        
        # Label background
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
        
        # Center dot
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx, cy), 2, color_bgr, -1)
    
    # Legend
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
    
    # Color by text type
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
        
        # Bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 1)
        
        # Text with background
        label = f"{text} ({text_type})" if show_types else text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Confidence indicator
        if conf < 0.65:
            cv2.circle(vis, (x2 - 5, y1 + 5), 4, (239, 68, 68), -1)  # Red dot for low conf
    
    return vis


def visualize_vectorization_steps(
    image: np.ndarray,
    exclusion_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Visualize all steps of vectorization - наглядно показывает трансформацию.
    
    Returns dict with:
      - binary: thresholded image
      - masked: with exclusions
      - skeleton: 1-pixel skeleton
      - segments: vectorized segments
      - junctions: junction points
      - final: overlay on original
    """
    from avers.core.validators import WireVectorizer
    
    vis_steps = {}
    
    # Original
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Step 1: Binary
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    vis_steps["binary"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    # Step 2: Masked
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
    
    # Step 3: Skeleton
    from skimage import morphology
    skeleton = morphology.skeletonize((thresh_for_skeleton > 0).astype(np.uint8))
    skeleton_vis = (skeleton * 255).astype(np.uint8)
    vis_steps["skeleton"] = cv2.cvtColor(skeleton_vis, cv2.COLOR_GRAY2RGB)
    
    # Step 4: Vectorize
    vectorizer = WireVectorizer()
    segments, junctions = vectorizer.vectorize(image, exclusion_bboxes)
    
    # Segments visualization
    seg_vis = np.ones_like(image_rgb) * 255
    for seg in segments:
        cv2.line(seg_vis, seg.start, seg.end, (100, 100, 100), 2)
    vis_steps["segments"] = seg_vis
    
    # Junctions
    junc_vis = seg_vis.copy()
    for j in junctions:
        cv2.circle(junc_vis, j, 5, (239, 68, 68), -1)
        # Degree indicator
        cv2.putText(junc_vis, f"{len(junctions)}", (j[0]+8, j[1]-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (239, 68, 68), 1)
    vis_steps["junctions"] = junc_vis
    
    # Final overlay
    final = image.copy()
    if len(final.shape) == 2:
        final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    
    for seg in segments:
        cv2.line(final, seg.start, seg.end, (100, 100, 100), 2)
    for j in junctions:
        cv2.circle(final, j, 6, (239, 68, 68), -1)
        cv2.circle(final, j, 8, (239, 68, 68, 0.3), 1)
    
    vis_steps["final"] = final
    
    return vis_steps


def visualize_graph_interactive(
    components: List[Any],
    nets: List[Any],
    width: int = 1000,
    height: int = 800,
) -> Dict[str, Any]:
    """
    Generate interactive graph data for D3/vis-network.
    
    Returns JSON with nodes and edges for frontend.
    """
    nodes = []
    edges = []
    
    # Component nodes
    for comp in components:
        # Handle both dict and object
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
        
        # Component node
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
        
        # Pin nodes
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
            
            # Edge component -> pin
            edges.append({
                "from": comp_id,
                "to": f"{comp_id}_pin_{pin_num}",
                "type": "has_pin",
                "color": "rgba(200,200,200,0.5)",
                "dashes": True,
            })
    
    # Net edges (wire connections)
    for net in nets:
        if isinstance(net, dict):
            net_id = net.get("net_id", "")
            connections = net.get("connections", [])
            path_points = net.get("path_points", [])
        else:
            net_id = getattr(net, "net_id", "")
            connections = getattr(net, "connections", [])
            path_points = getattr(net, "path_points", [])
        
        # Connect pins in same net
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
    """
    Visualize VLM arbitration - показывает ROI и RAG примеры.
    
    Returns dict with ROI crops and explanation.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    cx, cy = (x1+x2)//2, (y1+y2)//2
    
    # Expand for context
    expand = 1.5
    bw, bh = (x2-x1), (y2-y1)
    new_w, new_h = int(bw * expand), int(bh * expand)
    
    x1_exp = max(0, cx - new_w//2)
    y1_exp = max(0, cy - new_h//2)
    x2_exp = min(w, cx + new_w//2)
    y2_exp = min(h, cy + new_h//2)
    
    # Full context
    context = image[y1_exp:y2_exp, x1_exp:x2_exp]
    
    # ROI 256x256
    roi = image[max(0, cy-roi_size//2):min(h, cy+roi_size//2),
                max(0, cx-roi_size//2):min(w, cx+roi_size//2)]
    roi_resized = cv2.resize(roi, (roi_size, roi_size))
    
    # Highlight bbox in context
    context_vis = context.copy()
    # Convert bbox to context-local coords
    local_x1, local_y1 = x1 - x1_exp, y1 - y1_exp
    local_x2, local_y2 = x2 - x1_exp, y2 - y1_exp
    cv2.rectangle(context_vis, (local_x1, local_y1), (local_x2, local_y2), (239, 68, 68), 2)
    
    # Center crosshair
    ch_x, ch_y = context_vis.shape[1]//2, context_vis.shape[0]//2
    cv2.line(context_vis, (ch_x-20, ch_y), (ch_x+20, ch_y), (239, 68, 68), 1)
    cv2.line(context_vis, (ch_x, ch_y-20), (ch_x, ch_y+20), (239, 68, 68), 1)
    
    result = {
        "context": context_vis,
        "roi": roi_resized,
        "original_bbox": np.array([[x1, y1], [x2, y2]]),
    }
    
    # RAG examples visualization
    if rag_examples:
        # Create comparison grid
        grid_size = 256
        num_examples = min(len(rag_examples), 4)
        grid_width = grid_size * (num_examples + 1)  # +1 for query
        grid_height = grid_size
        
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Query ROI
        grid[0:grid_size, 0:grid_size] = roi_resized
        
        # Examples
        for i, ex in enumerate(rag_examples[:num_examples]):
            x_offset = (i+1) * grid_size
            # For demo, create dummy example visualization
            # In real, would load example images from RAG
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
    """
    Create HTML report with all stages visualization - наглядный отчет.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    from avers.core.validators import SlicedDetector, SchematicOCR, WireVectorizer
    
    # Stage 1: Tiles
    tiles_vis = visualize_sahi_tiles(image, tile_size=1024, overlap_ratio=0.2)
    
    # Stage 2: Detections
    detector = SlicedDetector(device="cpu")
    detections = detector.detect(image)
    det_vis = visualize_detections_enhanced(image, detections)
    
    # Stage 3: OCR
    ocr = SchematicOCR()
    texts = ocr.recognize(image)
    ocr_vis = visualize_ocr_enhanced(image, texts)
    
    # Stage 4: Vectorization steps
    comp_bboxes = [c.bbox if hasattr(c, 'bbox') else c.get('bbox') for c in (manifest.components if hasattr(manifest, 'components') else [])]
    text_bboxes = [t["bbox"] for t in texts]
    vec_steps = visualize_vectorization_steps(image, comp_bboxes + text_bboxes)
    
    # Stage 5: Graph
    graph_data = visualize_graph_interactive(
        manifest.components if hasattr(manifest, 'components') else [],
        manifest.nets if hasattr(manifest, 'nets') else []
    )
    
    # Save images
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
    
    # Create HTML
    html = f"""
<!DOCTYPE html>
<html lang=\"ru\">
<head>
<meta charset=\"UTF-8\">
<title>АВЕРС Отчет - {manifest.schema_metadata.source_file if hasattr(manifest, 'schema_metadata') else 'schema'}</title>
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
<p>Файл: {manifest.schema_metadata.source_file if hasattr(manifest, 'schema_metadata') else 'unknown'} | 
Размер: {manifest.schema_metadata.width if hasattr(manifest, 'schema_metadata') else image.shape[1]}×{manifest.schema_metadata.height if hasattr(manifest, 'schema_metadata') else image.shape[0]} |
Компонентов: {len(manifest.components) if hasattr(manifest, 'components') else 0} | 
Цепей: {len(manifest.nets) if hasattr(manifest, 'nets') else 0} |
Проблем: {len(manifest.human_review_required) if hasattr(manifest, 'human_review_required') else 0}</p>
<div class=\"metrics\">
"""
    
    for stage, t in stage_timings.items():
        html += f"<div class=\"metric\"><b>{stage}</b><br>{t*1000:.1f}ms</div>"
    
    html += f"""
</div>
</div>

<div class=\"stage\">
<h2>Стадия 1: SAHI Нарезка</h2>
<p>Проблема: ресайз 14000×3500 до 640×640 уничтожает тонкие линии. Решение: нарезка на тайлы 1024×1024 с overlap 0.2, параллельная обработка, NMS склейка.</p>
<img src=\"{report_dir.name}/{assets['tiles']}\" alt=\"SAHI tiles\">
</div>

<div class=\"stage\">
<h2>Стадия 2: Детекция УГО (RT-DETR/YOLO)</h2>
<p>Детекция 10 классов ГОСТ: connector_body, pin, junction_dot, ground, shield, offpage, diode, relay, resistor, capacitor.</p>
<img src=\"{report_dir.name}/{assets['detections']}\" alt=\"Detections\">
<pre>Найдено: {len(detections)} объектов
{chr(10).join([f\"  {d['category']} {d['bbox']} conf={d['confidence']:.2f}\" for d in detections[:10]])}
</pre>
</div>

<div class=\"stage\">
<h2>Стадия 3: OCR текста (PaddleOCR)</h2>
<p>Детекция под 0°/90°/270°, regex валидация: Х1, Ш1, СНЦ144, пины, БПВЛ, +27В.</p>
<img src=\"{report_dir.name}/{assets['ocr']}\" alt=\"OCR\">
<pre>Распознано: {len(texts)} текстов
{chr(10).join([f\"  '{t['text']}' {t['bbox']} conf={t['confidence']:.2f}\" for t in texts[:10]])}
</pre>
</div>

<div class=\"stage\">
<h2>Стадия 4: Векторизация (OpenCV)</h2>
<p>1. Маска исключений (вырезаем УГО и текст) → 2. Скелетизация Guo-Hall → 3. Hough + RDP → 4. Junction detection (degree>=3).</p>
<div class=\"stage-grid\">
<div><h3>Binary</h3><img src=\"{report_dir.name}/{assets.get('vec_binary', '')}\" alt=\"binary\"></div>
<div><h3>Masked (исключены УГО/текст)</h3><img src=\"{report_dir.name}/{assets.get('vec_masked', '')}\" alt=\"masked\"></div>
<div><h3>Skeleton (1px)</h3><img src=\"{report_dir.name}/{assets.get('vec_skeleton', '')}\" alt=\"skeleton\"></div>
<div><h3>Segments</h3><img src=\"{report_dir.name}/{assets.get('vec_segments', '')}\" alt=\"segments\"></div>
<div><h3>Junctions</h3><img src=\"{report_dir.name}/{assets.get('vec_junctions', '')}\" alt=\"junctions\"></div>
<div><h3>Final overlay</h3><img src=\"{report_dir.name}/{assets.get('vec_final', '')}\" alt=\"final\"></div>
</div>
</div>

<div class=\"stage\">
<h2>Стадия 5: Графовый синтез (NetworkX)</h2>
<p>Snapping проводов к пинам R=15px, k-d tree для текста, схлопывание цепочек, поиск T-узлов.</p>
<pre>Граф: {graph_data['stats']['components']} компонентов, {graph_data['stats']['pins']} пинов, {graph_data['stats']['nets']} цепей
Компоненты:
{chr(10).join([f\"  {c.id if hasattr(c, 'id') else c.get('id')}: {c.designator if hasattr(c, 'designator') else c.get('designator')} {c.bbox if hasattr(c, 'bbox') else c.get('bbox')}\" for c in (manifest.components if hasattr(manifest, 'components') else [])[:10]])}
</pre>
</div>

<div class=\"stage\">
<h2>Стадия 6: VLM-арбитраж (Qwen-VL + Vision RAG)</h2>
<p>Точечный кроп ROI 256×256 при коллизии: confidence<0.65, висящие линии, спорный перекресток. RAG: CLIP 512d + FAISS Top-K → few-shot prompt.</p>
<pre>Проблем для проверки: {len(manifest.human_review_required) if hasattr(manifest, 'human_review_required') else 0}
{chr(10).join([f\"  {iss.issue_type if hasattr(iss, 'issue_type') else iss.get('issue_type')}: {iss.bbox if hasattr(iss, 'bbox') else iss.get('bbox')} - {iss.description if hasattr(iss, 'description') else iss.get('description')}\" for iss in (manifest.human_review_required if hasattr(manifest, 'human_review_required') else [])[:10]])}
</pre>
</div>

<div class=\"stage\">
<h2>Выходной Netlist JSON</h2>
<pre>{str(manifest.model_dump() if hasattr(manifest, 'model_dump') else manifest)[:2000]}...</pre>
</div>

</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"Pipeline report saved to {output_path}")
    return output_path
