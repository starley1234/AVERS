"""
Stage 1: SAHI-based Image Slicing Engine

Implements sliding window tiling with overlap for large schematic images.
Projects detections back to global coordinate space with NMS deduplication.

Problem addressed:
- Direct resize of 14000x3500px to 640x640px destroys thin lines and small symbols
- Solution: Slice into overlapping 1024x1024 tiles, process in parallel,
  project coordinates back to global space
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
from pathlib import Path
import numpy as np
import cv2


@dataclass
class Tile:
    """Represents a single image tile with its metadata."""
    tile_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    image: np.ndarray
    overlap_left: int = 0
    overlap_top: int = 0
    overlap_right: int = 0
    overlap_bottom: int = 0

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def area(self) -> int:
        """Get tile area in pixels."""
        return self.width * self.height

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within tile bounds."""
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def to_global_coords(self, local_x: int, local_y: int) -> Tuple[int, int]:
        """Convert local tile coordinates to global image coordinates."""
        return (self.x_min + local_x, self.y_min + local_y)

    def to_local_coords(self, global_x: int, global_y: int) -> Tuple[int, int]:
        """Convert global coordinates to local tile coordinates."""
        return (global_x - self.x_min, global_y - self.y_min)

    def crop_to_roi(self, image: np.ndarray) -> np.ndarray:
        """Crop image to this tile's region of interest."""
        return image[self.y_min:self.y_max, self.x_min:self.x_max]


@dataclass
class DetectionBox:
    """
    Detected object box with coordinate conversion support.
    Used for projecting tile-level detections back to global coordinates.
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = ""
    tile_id: Optional[int] = None

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point."""
        return (
            (self.x_min + self.x_max) // 2,
            (self.y_min + self.y_max) // 2,
        )

    @property
    def area(self) -> float:
        """Get box area."""
        return float((self.x_max - self.x_min) * (self.y_max - self.y_min))

    def to_global(self, tile: Tile) -> "DetectionBox":
        """Project detection from tile coordinates to global coordinates."""
        return DetectionBox(
            x_min=tile.x_min + self.x_min,
            y_min=tile.y_min + self.y_min,
            x_max=tile.x_min + self.x_max,
            y_max=tile.y_min + self.y_max,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            tile_id=tile.tile_id,
        )

    def to_local(self, tile: Tile) -> Tuple[int, int, int, int]:
        """Convert global coordinates to tile-local coordinates."""
        return (
            self.x_min - tile.x_min,
            self.y_min - tile.y_min,
            self.x_max - tile.x_min,
            self.y_max - tile.y_min,
        )

    def iou(self, other: "DetectionBox") -> float:
        """Calculate Intersection over Union with another box."""
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


class SlicingEngine:
    """
    SAHI-style sliding window slicer for large schematic images.

    Features:
    - Configurable tile size and overlap
    - Automatic stride calculation
    - Edge handling with padding
    - Iterator-based tile generation for memory efficiency
    - NMS deduplication for projected detections
    """

    def __init__(
        self,
        tile_size: int = 1024,
        overlap_ratio: float = 0.2,
        target_stride: Optional[int] = None,
        min_tile_area: int = 100,
    ):
        """
        Initialize the slicing engine.

        Args:
            tile_size: Size of each tile in pixels (square)
            overlap_ratio: Overlap ratio between adjacent tiles (0.0-0.5)
            target_stride: Override automatic stride calculation
            min_tile_area: Minimum tile area to process
        """
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.target_stride = target_stride
        self.min_tile_area = min_tile_area

        # Calculate stride from tile_size and overlap
        self.stride = target_stride or int(tile_size * (1 - overlap_ratio))

    def generate_tiles(
        self,
        image: np.ndarray,
        start_id: int = 0,
    ) -> Iterator[Tile]:
        """
        Generate overlapping tiles from an image.

        Args:
            image: Input image (H, W, C) or (H, W)
            start_id: Starting tile ID for unique identification

        Yields:
            Tile objects with image data and metadata
        """
        height, width = image.shape[:2]
        tile_id = start_id

        # Sliding window with stride
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # Calculate tile bounds with padding at edges
                x_max = min(x + self.tile_size, width)
                y_max = min(y + self.tile_size, height)
                x_min = x_max - self.tile_size if x_max == width else x
                y_min = y_max - self.tile_size if y_max == height else y

                # Calculate overlaps (for reference, not actual padding)
                overlap_left = x - x_min if x > 0 else 0
                overlap_top = y - y_min if y > 0 else 0
                overlap_right = (x + self.tile_size) - x_max if x_max < width else 0
                overlap_bottom = (y + self.tile_size) - y_max if y_max < height else 0

                # Extract tile
                tile_image = image[y_min:y_max, x_min:x_max]

                # Skip small/empty tiles
                if tile_image.size < self.min_tile_area:
                    continue

                tile = Tile(
                    tile_id=tile_id,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    image=tile_image,
                    overlap_left=overlap_left,
                    overlap_top=overlap_top,
                    overlap_right=overlap_right,
                    overlap_bottom=overlap_bottom,
                )

                yield tile
                tile_id += 1

    def project_detections_to_global(
        self,
        tile_detections: List[DetectionBox],
        tiles: List[Tile],
    ) -> List[DetectionBox]:
        """
        Project tile-level detections to global coordinate space.

        Args:
            tile_detections: Detections with tile_id attribute
            tiles: List of tiles (for coordinate lookup)

        Returns:
            Detections in global coordinates
        """
        # Create tile lookup
        tile_map = {t.tile_id: t for t in tiles}

        global_detections = []
        for det in tile_detections:
            if det.tile_id in tile_map:
                tile = tile_map[det.tile_id]
                global_det = det.to_global(tile)
                global_detections.append(global_det)

        return global_detections

    @staticmethod
    def nms(
        detections: List[DetectionBox],
        iou_threshold: float = 0.45,
    ) -> List[DetectionBox]:
        """
        Non-Maximum Suppression for deduplicating overlapping detections.

        Args:
            detections: List of detections in same coordinate space
            iou_threshold: IOU threshold for suppression

        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []

        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while sorted_dets:
            current = sorted_dets.pop(0)
            keep.append(current)

            # Remove overlapping detections
            sorted_dets = [
                det for det in sorted_dets
                if current.iou(det) < iou_threshold
            ]

        return keep

    def visualize_tiles(
        self,
        image: np.ndarray,
        tiles: List[Tile],
        output_path: Optional[Path] = None,
        show_overlaps: bool = True,
    ) -> np.ndarray:
        """
        Visualize tile grid on image for debugging.

        Args:
            image: Original image
            tiles: List of tiles to visualize
            output_path: Optional path to save visualization
            show_overlaps: Color overlaps differently

        Returns:
            Visualization image
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Color palette for tiles
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for i, tile in enumerate(tiles):
            color = colors[i % len(colors)]

            # Draw rectangle
            cv2.rectangle(vis, (tile.x_min, tile.y_min), (tile.x_max, tile.y_max), color, 2)

            # Draw tile ID
            cv2.putText(
                vis,
                f"T{tile.tile_id}",
                (tile.x_min + 5, tile.y_min + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            # Highlight overlap regions
            if show_overlaps and (tile.overlap_left or tile.overlap_top):
                overlay = vis.copy()
                # Left overlap
                if tile.overlap_left > 0:
                    cv2.rectangle(
                        overlay,
                        (tile.x_min, tile.y_min),
                        (tile.x_min + tile.overlap_left, tile.y_max),
                        (0, 255, 0),
                        -1,
                    )
                # Top overlap
                if tile.overlap_top > 0:
                    cv2.rectangle(
                        overlay,
                        (tile.x_min, tile.y_min),
                        (tile.x_max, tile.y_min + tile.overlap_top),
                        (0, 255, 0),
                        -1,
                    )
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        if output_path:
            cv2.imwrite(str(output_path), vis)

        return vis


def load_image(path: str | Path) -> np.ndarray:
    """
    Load image from file with automatic format handling.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array (H, W, C) or (H, W)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Try different loading methods based on extension
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        # TIF can be multi-page
        import PIL.Image
        img = PIL.Image.open(path)
        if hasattr(img, "n_frames") and img.n_frames > 1:
            # Use first frame for now
            img.seek(0)
        image = np.array(img.convert("RGB"))
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
