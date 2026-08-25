"""Stage 6: VLM Arbitration stub module."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

# VLM client (lazy-loaded)
_vlm_client = None


def load_vlm_client(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
):
    """Load VLM client for arbitration."""
    global _vlm_client
    
    if _vlm_client is not None:
        return _vlm_client
    
    # Placeholder - would load actual VLM
    # from transformers import AutoProcessor, AutoModelForVision2Seq
    # processor = AutoProcessor.from_pretrained(model_name)
    # model = AutoModelForVision2Seq.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map=device,
    # )
    
    _vlm_client = {"loaded": True, "model_name": model_name}
    return _vlm_client


def extract_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    roi_size: int = 256,
) -> np.ndarray:
    """
    Extract Region of Interest crop from image.

    Args:
        image: Full image
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        roi_size: Output size

    Returns:
        Cropped and resized ROI
    """
    import cv2
    
    x_min, y_min, x_max, y_max = bbox
    h, w = y_max - y_min, x_max - x_min
    
    # Crop
    crop = image[y_min:y_max, x_min:x_max]
    
    # Resize to standard size
    resized = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
    
    return resized


def query_vlm_for_connection(
    roi_image: np.ndarray,
    model_client: Optional[Dict] = None,
) -> Dict:
    """
    Query VLM to determine if junction point is a connection.

    Args:
        roi_image: 256x256 crop around junction
        model_client: VLM client instance

    Returns:
        {'connected': bool, 'confidence': float}
    """
    if model_client is None:
        model_client = load_vlm_client()
    
    # Placeholder response
    # Full implementation would:
    # 1. Prepare prompt with image
    # 2. Query VLM
    # 3. Parse JSON response
    
    return {
        "connected": False,
        "confidence": 0.5,
        "reasoning": "VLM not implemented",
    }


def resolve_crossing(
    image: np.ndarray,
    crossing_bbox: Tuple[int, int, int, int],
    roi_size: int = 256,
) -> Dict:
    """
    Determine if wire crossing is a connection or overlap.

    Args:
        image: Full image
        crossing_bbox: Bounding box around crossing
        roi_size: ROI size for VLM

    Returns:
        Resolution result
    """
    # Extract ROI
    roi = extract_roi(image, crossing_bbox, roi_size)
    
    # Query VLM
    result = query_vlm_for_connection(roi)
    
    return {
        "type": "crossing",
        "resolution": "connected" if result["connected"] else "overlap",
        "confidence": result["confidence"],
        "bbox": crossing_bbox,
    }


def resolve_low_confidence_detection(
    image: np.ndarray,
    detection_bbox: Tuple[int, int, int, int],
    hypotheses: List[str],
    roi_size: int = 256,
) -> Dict:
    """
    Use VLM to resolve ambiguous detection.

    Args:
        image: Full image
        detection_bbox: Bounding box of uncertain detection
        hypotheses: List of possible interpretations
        roi_size: ROI size

    Returns:
        Resolution with chosen hypothesis
    """
    # Extract ROI
    roi = extract_roi(image, detection_bbox, roi_size)
    
    # Placeholder - would provide hypotheses in prompt
    result = query_vlm_for_connection(roi)
    
    chosen = hypotheses[0] if hypotheses else "unknown"
    
    return {
        "type": "detection",
        "chosen": chosen,
        "confidence": result["confidence"],
        "bbox": detection_bbox,
    }


def resolve_text_ocr(
    image: np.ndarray,
    text_bbox: Tuple[int, int, int, int],
    ocr_hypotheses: List[Tuple[str, float]],
    roi_size: int = 256,
) -> Dict:
    """
    Use VLM to select correct OCR interpretation.

    Args:
        image: Full image
        text_bbox: Bounding box of text
        ocr_hypotheses: List of (text, confidence) tuples
        roi_size: ROI size

    Returns:
        Selected text with confidence
    """
    # Extract ROI with context
    x_min, y_min, x_max, y_max = text_bbox
    
    # Expand ROI for context
    expand = 50
    expanded_bbox = (
        max(0, x_min - expand),
        max(0, y_min - expand),
        min(image.shape[1], x_max + expand),
        min(image.shape[0], y_max + expand),
    )
    
    roi = extract_roi(image, expanded_bbox, roi_size)
    
    # Placeholder
    result = query_vlm_for_connection(roi)
    
    best_hypothesis = max(ocr_hypotheses, key=lambda h: h[1]) if ocr_hypotheses else ("", 0.0)
    
    return {
        "type": "text",
        "selected_text": best_hypothesis[0],
        "confidence": result["confidence"],
        "bbox": text_bbox,
    }


def arbitrate_issues(
    image: np.ndarray,
    issues: List[Dict],
    max_calls: int = 50,
) -> List[Dict]:
    """
    Process list of issues requiring VLM arbitration.

    Args:
        image: Full image
        issues: List of issue dicts
        max_calls: Maximum VLM calls (cost control)

    Returns:
        List of resolutions
    """
    client = load_vlm_client()
    resolutions = []
    calls_made = 0
    
    for issue in issues:
        if calls_made >= max_calls:
            resolutions.append({
                "issue": issue,
                "resolved": False,
                "reason": "max_calls_reached",
            })
            continue
        
        issue_type = issue.get("type")
        
        if issue_type == "crossing":
            resolution = resolve_crossing(
                image, issue["bbox"], issue.get("roi_size", 256)
            )
        elif issue_type == "detection":
            resolution = resolve_low_confidence_detection(
                image, issue["bbox"], issue.get("hypotheses", []),
            )
        elif issue_type == "text":
            resolution = resolve_text_ocr(
                image, issue["bbox"], issue.get("ocr_hypotheses", []),
            )
        else:
            resolution = {"type": issue_type, "resolved": False}
        
        resolutions.append(resolution)
        calls_made += 1
    
    return resolutions
