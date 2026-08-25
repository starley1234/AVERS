"""
Stage 4: Wire Vectorization using OpenCV

Converts raster wire traces to vector polylines:
1. Create exclusion mask (remove УГО and text regions)
2. Skeletonize lines (Guo-Hall algorithm)
3. Extract polylines via Hough transform + contour tracing
4. Simplify using Ramer-Douglas-Peucker
5. Detect junction points
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from pathlib import Path
import numpy as np
import cv2

from avers.stages.stage5_graph_synthesis import WireSegment
from avers.core.logger import get_logger

logger = get_logger("avers.vectorization")


@dataclass
class VectorizationConfig:
    """Vectorization configuration."""
    # Line detection
    line_thickness_min: int = 2
    line_thickness_max: int = 20
    
    # Skeletonization
    skeletonize_method: str = "guo_hall"  # 'guo_hall' or 'zhang_suen'
    
    # Polyline extraction
    hough_threshold: int = 50
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 10
    
    # RDP simplification
    rdp_epsilon: float = 2.0
    
    # Junction detection
    junction_degree_threshold: int = 3
    
    # Exclusion mask padding
    exclusion_padding: int = 5
    
    # Output
    min_segment_length: int = 10


class WireVectorizer:
    """
    Vectorizes wire traces from raster schematic images.
    
    Pipeline:
    1. Preprocess: grayscale + threshold
    2. Mask exclusion zones (УГО, text)
    3. Skeletonize to 1-pixel lines
    4. Extract polylines (Hough + contour tracing)
    5. Simplify with RDP
    6. Detect junctions
    """
    
    def __init__(self, config: Optional[VectorizationConfig] = None):
        self.config = config or VectorizationConfig()
        self._skeleton = None
        self._junction_points: Set[Tuple[int, int]] = set()
    
    def vectorize(
        self,
        image: np.ndarray,
        exclusion_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[List[WireSegment], Set[Tuple[int, int]]]:
        """
        Vectorize wire traces from image.
        
        Args:
            image: Input image (H, W, 3) RGB or (H, W) grayscale
            exclusion_bboxes: Bounding boxes to exclude (УГО, text regions)
            
        Returns:
            Tuple of (wire_segments, junction_points)
        """
        import time
        start = time.time()
        
        logger.info(f"Starting vectorization of {image.shape[1]}x{image.shape[0]} image")
        
        # Step 1: Preprocess
        gray = self._preprocess(image)
        
        # Step 2: Create exclusion mask
        if exclusion_bboxes:
            mask = self._create_exclusion_mask(gray.shape, exclusion_bboxes)
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Step 3: Skeletonize
        skeleton = self._skeletonize(gray)
        self._skeleton = skeleton
        
        # Step 4: Extract polylines
        polylines = self._extract_polylines(skeleton)
        logger.debug(f"Extracted {len(polylines)} raw polylines")
        
        # Step 5: Simplify with RDP
        simplified = [self._rdp_simplify(p) for p in polylines]
        
        # Step 6: Convert to WireSegments
        segments = self._create_segments(simplified)
        
        # Step 7: Detect junctions
        junctions = self._detect_junctions(skeleton)
        self._junction_points = junctions
        
        elapsed = time.time() - start
        logger.info(f"Vectorization complete: {len(segments)} segments, "
                    f"{len(junctions)} junctions ({elapsed*1000:.1f}ms)")
        
        return segments, junctions
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and binarize."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive threshold for uneven illumination
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def _create_exclusion_mask(
        self,
        shape: Tuple[int, int],
        bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Create mask with excluded regions set to 0."""
        mask = np.ones(shape[:2], dtype=np.uint8) * 255
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            padding = self.config.exclusion_padding
            
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(shape[1], x_max + padding)
            y_max = min(shape[0], y_max + padding)
            
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 0, -1)
        
        return mask
    
    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Skeletonize binary image to 1-pixel wide lines.
        
        Args:
            binary: Binary image (0 or 255)
            
        Returns:
            Skeleton image (0 or 255)
        """
        from skimage import morphology
        
        # Ensure binary
        img = (binary > 0).astype(np.uint8)
        
        # Use scikit-image skeletonize (Zhang-Suen by default)
        skeleton = morphology.skeletonize(img)
        
        return (skeleton * 255).astype(np.uint8)
    
    def _extract_polylines(self, skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Extract polylines from skeleton.
        
        Combines:
        - Probabilistic Hough Transform for straight lines
        - Contour tracing for curves
        """
        polylines = []
        
        # Method 1: Hough Transform for straight lines
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )
        
        if lines is not None:
            for line in lines:
                # Handle different OpenCV versions
                if len(line.shape) == 3:
                    x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
                else:
                    x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                polylines.append([(x1, y1), (x2, y2)])
        
        # Method 2: Contour tracing for complex shapes
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Approximate to reduce points
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            if len(approx) >= 2:
                polyline = [(int(p[0][0]), int(p[0][1])) for p in approx]
                
                # Calculate length
                length = sum(
                    np.sqrt((polyline[i][0] - polyline[i+1][0])**2 + 
                           (polyline[i][1] - polyline[i+1][1])**2)
                    for i in range(len(polyline) - 1)
                )
                
                if length >= self.config.min_segment_length:
                    polylines.append(polyline)
        
        return polylines
    
    def _rdp_simplify(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Ramer-Douglas-Peucker algorithm for polyline simplification.
        
        Args:
            points: List of (x, y) coordinates
            
        Returns:
            Simplified polyline
        """
        if len(points) < 3:
            return points
        
        epsilon = self.config.rdp_epsilon
        
        def perpendicular_distance(point: Tuple[int, int], 
                                   line_start: Tuple[int, int], 
                                   line_end: Tuple[int, int]) -> float:
            """Calculate perpendicular distance from point to line."""
            x, y = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                return np.sqrt((x - x1)**2 + (y - y1)**2)
            
            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx**2 + dy**2)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            
            return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
        
        def rdp_recursive(pts: List[Tuple[int, int]], eps: float) -> List[Tuple[int, int]]:
            if len(pts) < 3:
                return pts
            
            # Find point with max distance
            start = pts[0]
            end = pts[-1]
            
            max_dist = 0
            max_idx = 0
            
            for i in range(1, len(pts) - 1):
                d = perpendicular_distance(pts[i], start, end)
                if d > max_dist:
                    max_dist = d
                    max_idx = i
            
            if max_dist > eps:
                left = rdp_recursive(pts[:max_idx + 1], eps)
                right = rdp_recursive(pts[max_idx:], eps)
                return left[:-1] + right
            else:
                return [start, end]
        
        return rdp_recursive(points, epsilon)
    
    def _create_segments(
        self,
        polylines: List[List[Tuple[int, int]]],
    ) -> List[WireSegment]:
        """Convert polylines to WireSegment objects."""
        segments = []
        segment_id = 0
        
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            
            # Calculate total length
            length = sum(
                np.sqrt((polyline[i][0] - polyline[i+1][0])**2 +
                       (polyline[i][1] - polyline[i+1][1])**2)
                for i in range(len(polyline) - 1)
            )
            
            if length < self.config.min_segment_length:
                continue
            
            # Create segment for each segment of polyline
            for i in range(len(polyline) - 1):
                start = polyline[i]
                end = polyline[i + 1]
                
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                
                segment = WireSegment(
                    segment_id=segment_id,
                    start=start,
                    end=end,
                    points=[start, end],
                    confidence=1.0,
                )
                
                segments.append(segment)
                segment_id += 1
        
        return segments
    
    def _detect_junctions(self, skeleton: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Detect junction points in skeleton.
        
        A junction is a pixel where 3+ wire segments meet.
        
        Returns:
            Set of (x, y) junction coordinates
        """
        junctions = set()
        
        # Look at each white pixel
        height, width = skeleton.shape
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] == 0:
                    continue
                
                # Count neighbors
                neighbors = [
                    skeleton[y-1, x],   # top
                    skeleton[y+1, x],   # bottom
                    skeleton[y, x-1],   # left
                    skeleton[y, x+1],   # right
                    skeleton[y-1, x-1], # top-left
                    skeleton[y-1, x+1], # top-right
                    skeleton[y+1, x-1], # bottom-left
                    skeleton[y+1, x+1], # bottom-right
                ]
                
                degree = sum(1 for n in neighbors if n > 0)
                
                # Junction if degree >= 3 (T, Y, or X junction)
                if degree >= 3:
                    junctions.add((x, y))
        
        return junctions
    
    def connect_segments(
        self,
        segments: List[WireSegment],
        tolerance: float = 5.0,
    ) -> List[WireSegment]:
        """
        Connect wire segments at their endpoints.
        
        Merges segments that share endpoints or are very close.
        """
        if len(segments) < 2:
            return segments
        
        # Build endpoint lookup
        endpoints: Dict[Tuple[int, int], List[int]] = {}
        
        for seg in segments:
            for endpoint in [seg.start, seg.end]:
                # Round to grid for snapping
                grid_point = (round(endpoint[0] / tolerance) * int(tolerance),
                             round(endpoint[1] / tolerance) * int(tolerance))
                
                if grid_point not in endpoints:
                    endpoints[grid_point] = []
                endpoints[grid_point].append(seg.segment_id)
        
        # Find connected groups
        connected = {}
        for point, seg_ids in endpoints.items():
            if len(seg_ids) > 1:
                for seg_id in seg_ids:
                    connected[seg_id] = True
        
        return segments  # Return as-is for now (full merging done in GraphBuilder)


def vectorize_wires(
    image: np.ndarray,
    ugo_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    text_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    config: Optional[VectorizationConfig] = None,
) -> Tuple[List[WireSegment], Set[Tuple[int, int]]]:
    """
    Full wire vectorization pipeline.
    
    Args:
        image: Input image
        ugo_bboxes: Bounding boxes of detected УГО
        text_bboxes: Bounding boxes of detected text
        config: Vectorization configuration
        
    Returns:
        Tuple of (wire_segments, junction_points)
    """
    config = config or VectorizationConfig()
    
    vectorizer = WireVectorizer(config)
    
    # Combine exclusion bboxes
    all_bboxes = []
    if ugo_bboxes:
        all_bboxes.extend(ugo_bboxes)
    if text_bboxes:
        all_bboxes.extend(text_bboxes)
    
    return vectorizer.vectorize(image, all_bboxes if all_bboxes else None)


def visualize_vectorization(
    image: np.ndarray,
    segments: List[WireSegment],
    junctions: Set[Tuple[int, int]],
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Create visualization of vectorized wires.
    
    Args:
        image: Original image
        segments: Vectorized wire segments
        junctions: Detected junction points
        output_path: Optional path to save visualization
        
    Returns:
        Visualization image
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        vis = image.copy()
    else:
        vis = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    
    # Draw wire segments
    for seg in segments:
        color = (100, 100, 100) if seg.confidence > 0.5 else (200, 100, 100)
        cv2.line(vis, seg.start, seg.end, color, 2)
    
    # Draw junctions
    for junc in junctions:
        cv2.circle(vis, junc, 5, (0, 0, 255), -1)
    
    if output_path:
        cv2.imwrite(str(output_path), vis)
    
    return vis
