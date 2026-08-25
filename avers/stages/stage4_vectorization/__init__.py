"""Stage 4: Wire Vectorization stub module."""

from typing import List, Tuple, Optional
import numpy as np

# Import skeleton types for WireSegment
from avers.stages.stage5_graph_synthesis.graph_builder import WireSegment


def create_exclusion_mask(
    image_shape: Tuple[int, int],
    ugo_bboxes: List[Tuple[int, int, int, int]],
    text_bboxes: List[Tuple[int, int, int, int]],
    padding: int = 5,
) -> np.ndarray:
    """
    Create mask excluding УГО and text regions.

    Args:
        image_shape: (height, width)
        ugo_bboxes: List of УГО bounding boxes
        text_bboxes: List of text bounding boxes
        padding: Padding around excluded regions

    Returns:
        Binary mask (1 = include, 0 = exclude)
    """
    import cv2
    
    mask = np.ones(image_shape[:2], dtype=np.uint8)
    
    all_bboxes = ugo_bboxes + text_bboxes
    
    for bbox in all_bboxes:
        x_min, y_min, x_max, y_max = bbox
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image_shape[1], x_max + padding)
        y_max = min(image_shape[0], y_max + padding)
        
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 0, -1)
    
    return mask


def skeletonize(
    binary_image: np.ndarray,
    method: str = "guo_hall",
) -> np.ndarray:
    """
    Skeletonize binary image to 1-pixel wide lines.

    Args:
        binary_image: Binary image (0 = background, 255 = foreground)
        method: Thinning algorithm ('guo_hall' or 'zhang_suen')

    Returns:
        Skeleton image
    """
    from skimage import morphology
    
    # Ensure binary
    binary = (binary_image > 0).astype(np.uint8)
    
    if method == "guo_hall":
        skeleton = morphology.skeletonize_guo_hall(binary)
    else:
        skeleton = morphology.skeletonize_zhang_suen(binary)
    
    return (skeleton * 255).astype(np.uint8)


def extract_polylines(
    skeleton: np.ndarray,
    min_length: int = 10,
) -> List[List[Tuple[int, int]]]:
    """
    Extract polylines from skeleton.

    Args:
        skeleton: Skeletonized image
        min_length: Minimum polyline length

    Returns:
        List of polylines as lists of (x, y) coordinates
    """
    import cv2
    
    polylines = []
    
    # Find contours (for closed loops)
    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        if len(contour) >= min_length:
            polyline = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
            polylines.append(polyline)
    
    # Also extract lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=min_length,
        minLineLength=min_length,
        maxLineGap=5,
    )
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if this line overlaps existing polylines
            # (simplified - would need more sophisticated merging)
            polylines.append([(x1, y1), (x2, y2)])
    
    return polylines


def simplify_polyline(
    points: List[Tuple[int, int]],
    epsilon: float = 2.0,
) -> List[Tuple[int, int]]:
    """
    Simplify polyline using Ramer-Douglas-Peucker algorithm.

    Args:
        points: List of (x, y) coordinates
        epsilon: Simplification tolerance

    Returns:
        Simplified polyline
    """
    if len(points) < 3:
        return points
    
    # RDP algorithm
    def rdp_recursive(pts, epsilon):
        if len(pts) < 3:
            return pts
        
        # Find point with max distance from line
        x1, y1 = pts[0]
        x2, y2 = pts[-1]
        
        max_dist = 0
        max_idx = 0
        
        for i, (x, y) in enumerate(pts[1:-1], 1):
            dist = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / \
                   np.sqrt((y2 - y1)**2 + (x2 - x1)**2 + 1e-10)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > epsilon:
            left = rdp_recursive(pts[:max_idx + 1], epsilon)
            right = rdp_recursive(pts[max_idx:], epsilon)
            return left[:-1] + right
        else:
            return [pts[0], pts[-1]]
    
    return rdp_recursive(points, epsilon)


def detect_junction_points(
    skeleton: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Detect junction points in skeleton (T and X crossings).

    Args:
        skeleton: Skeletonized image

    Returns:
        List of junction point coordinates
    """
    import cv2
    
    junctions = []
    
    # Find connected components at each pixel
    # A junction has degree > 2 in the skeleton graph
    
    # Using corner detection as proxy (simplified)
    # Full implementation would trace the graph
    
    return junctions


def vectorize_wires(
    image: np.ndarray,
    ugo_bboxes: List[Tuple[int, int, int, int]] = None,
    text_bboxes: List[Tuple[int, int, int, int]] = None,
    rdp_epsilon: float = 2.0,
    min_wire_length: int = 10,
) -> List[WireSegment]:
    """
    Full wire vectorization pipeline.

    Args:
        image: Grayscale image
        ugo_bboxes: Bounding boxes of detected УГО
        text_bboxes: Bounding boxes of detected text
        rdp_epsilon: RDP simplification epsilon
        min_wire_length: Minimum wire segment length

    Returns:
        List of WireSegment objects
    """
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold to binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Create exclusion mask
    if ugo_bboxes or text_bboxes:
        mask = create_exclusion_mask(
            binary.shape,
            ugo_bboxes or [],
            text_bboxes or [],
        )
        binary = cv2.bitwise_and(binary, binary, mask=mask)
    
    # Skeletonize
    skel = skeletonize(binary, method="guo_hall")
    
    # Extract polylines
    polylines = extract_polylines(skel, min_length=min_wire_length)
    
    # Simplify and convert to WireSegments
    segments = []
    for i, polyline in enumerate(polylines):
        simplified = simplify_polyline(polyline, epsilon=rdp_epsilon)
        
        if len(simplified) >= 2:
            start = simplified[0]
            end = simplified[-1]
            
            segment = WireSegment(
                segment_id=i,
                start=start,
                end=end,
                points=simplified,
            )
            segments.append(segment)
    
    return segments
