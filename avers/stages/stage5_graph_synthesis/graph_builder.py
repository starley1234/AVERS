"""
Stage 5: Graph-based Syntax Synthesis (Assembly Engine)

Builds a topological graph from detected components and vectorized wires.
Implements:
- Wire snapping to component pins
- Text-to-component association via k-d tree
- Net inference from wire connectivity
- Collinear segment merging
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Iterator
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import KDTree


@dataclass
class WireSegment:
    """
    Vectorized wire segment from Stage 4.
    Represents a line or polyline in the schematic.
    """
    segment_id: int
    start: Tuple[int, int]
    end: Tuple[int, int]
    points: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 1.0
    is_horizontal: bool = False
    is_vertical: bool = False
    length: float = 0.0

    def __post_init__(self):
        """Calculate properties after initialization."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.length = np.sqrt(dx**2 + dy**2)
        self.is_horizontal = abs(dx) > abs(dy) and abs(dy) < 5
        self.is_vertical = abs(dy) > abs(dx) and abs(dx) < 5

    @property
    def midpoint(self) -> Tuple[int, int]:
        """Get midpoint of segment."""
        return (
            (self.start[0] + self.end[0]) // 2,
            (self.start[1] + self.end[1]) // 2,
        )

    def distance_to_point(self, point: Tuple[int, int]) -> float:
        """Calculate perpendicular distance from point to segment."""
        px, py = point
        x1, y1 = self.start
        x2, y2 = self.end

        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx**2 + dy**2

        if length_sq == 0:
            # Segment is a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # Project point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def connects_to(self, other: "WireSegment", tolerance: float = 5.0) -> bool:
        """Check if this segment connects to another."""
        dist_start = self.distance_to_point(other.start)
        dist_end = self.distance_to_point(other.end)

        return dist_start < tolerance or dist_end < tolerance


@dataclass
class PinReference:
    """Reference to a component pin."""
    component_id: str
    pin_number: str
    coord: Tuple[int, int]
    snapped_segment_id: Optional[int] = None


class GraphBuilder:
    """
    Builds a NetworkX graph from schematic components and wire segments.

    Graph structure:
    - Nodes: Wire junctions, component pins, wire endpoints
    - Edges: Wire connections between nodes

    Features:
    - Wire-to-pin snapping
    - T-junction detection
    - Wire chain collapsing
    - Net extraction
    """

    def __init__(
        self,
        snap_enabled: bool = True,
        snap_radius: int = 15,
        text_association_radius: int = 50,
        merge_collinear: bool = True,
        merge_tolerance: float = 5.0,
    ):
        """
        Initialize graph builder.

        Args:
            snap_enabled: Enable wire snapping to pins
            snap_radius: Snap radius in pixels
            text_association_radius: Max distance for text association
            merge_collinear: Merge collinear wire segments
            merge_tolerance: Tolerance for collinear merge
        """
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.text_association_radius = text_association_radius
        self.merge_collinear = merge_collinear
        self.merge_tolerance = merge_tolerance

        # Graph data structures
        self.graph = nx.MultiGraph()
        self.wire_segments: Dict[int, WireSegment] = {}
        self.pins: List[PinReference] = []
        self.component_pins_kdtree: Optional[KDTree] = None
        self.next_segment_id = 0

    def add_wire_segment(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        points: Optional[List[Tuple[int, int]]] = None,
        confidence: float = 1.0,
    ) -> int:
        """
        Add a wire segment to the graph builder.

        Args:
            start: Start coordinates (x, y)
            end: End coordinates (x, y)
            points: Optional intermediate points
            confidence: Segment confidence

        Returns:
            Segment ID
        """
        if points is None:
            points = [start, end]

        segment = WireSegment(
            segment_id=self.next_segment_id,
            start=start,
            end=end,
            points=points,
            confidence=confidence,
        )

        self.wire_segments[self.next_segment_id] = segment
        self.next_segment_id += 1

        return segment.segment_id

    def add_component_pins(self, pins: List[PinReference]) -> None:
        """
        Add component pins for snapping.

        Args:
            pins: List of pin references
        """
        self.pins = pins

        # Build k-d tree for fast spatial lookup
        if pins:
            coords = np.array([p.coord for p in pins])
            self.component_pins_kdtree = KDTree(coords)

    def snap_wire_to_pins(self) -> Dict[int, Optional[PinReference]]:
        """
        Snap wire segment endpoints to nearby component pins.

        Returns:
            Mapping of segment_id to snapped pin (or None)
        """
        if not self.snap_enabled or not self.pins or self.component_pins_kdtree is None:
            return {sid: None for sid in self.wire_segments}

        snapping = {}

        for seg_id, segment in self.wire_segments.items():
            snapped = None

            # Check both endpoints
            for endpoint in [segment.start, segment.end]:
                # Query nearby pins
                dist, idx = self.component_pins_kdtree.query(
                    endpoint,
                    k=1,
                    distance_upper_bound=self.snap_radius,
                )

                if dist <= self.snap_radius:
                    snapped = self.pins[idx]
                    # Update wire endpoint to exact pin position
                    if endpoint == segment.start:
                        segment.start = snapped.coord
                    else:
                        segment.end = snapped.coord
                    break

            snapping[seg_id] = snapped

        return snapping

    def merge_collinear_segments(self) -> int:
        """
        Merge collinear adjacent wire segments.

        Returns:
            Number of segments merged
        """
        if not self.merge_collinear or len(self.wire_segments) < 2:
            return 0

        merged = 0
        segments_to_remove = set()

        segment_ids = list(self.wire_segments.keys())

        for i, seg_id in enumerate(segment_ids):
            if seg_id in segments_to_remove:
                continue

            seg1 = self.wire_segments[seg_id]

            for other_id in segment_ids[i + 1:]:
                if other_id in segments_to_remove:
                    continue

                seg2 = self.wire_segments[other_id]

                # Check if collinear and adjacent
                if self._are_collinear_and_connected(seg1, seg2):
                    # Merge: create new segment spanning both
                    new_start = min(seg1.start, seg2.start, key=lambda p: (p[0], p[1]))
                    new_end = max(seg1.end, seg2.end, key=lambda p: (p[0], p[1]))

                    merged_seg = WireSegment(
                        segment_id=self.next_segment_id,
                        start=new_start,
                        end=new_end,
                        points=[new_start, new_end],
                        confidence=min(seg1.confidence, seg2.confidence),
                    )

                    self.wire_segments[self.next_segment_id] = merged_seg
                    self.next_segment_id += 1

                    segments_to_remove.add(seg_id)
                    segments_to_remove.add(other_id)
                    merged += 1

        # Remove merged segments
        for seg_id in segments_to_remove:
            del self.wire_segments[seg_id]

        return merged

    def _are_collinear_and_connected(
        self,
        seg1: WireSegment,
        seg2: WireSegment,
        angle_tolerance: float = 5.0,
    ) -> bool:
        """Check if two segments are collinear and share an endpoint."""
        # Check if they share an endpoint
        if seg1.end == seg2.start:
            shared = seg1.end
        elif seg1.start == seg2.end:
            shared = seg1.start
        else:
            # Check if endpoints are close
            dist = np.sqrt(
                (seg1.end[0] - seg2.start[0])**2 +
                (seg1.end[1] - seg2.start[1])**2
            )
            if dist > self.merge_tolerance:
                return False
            shared = seg1.end

        # Check collinearity (both horizontal or both vertical)
        if not (seg1.is_horizontal and seg2.is_horizontal) and \
           not (seg1.is_vertical and seg2.is_vertical):
            return False

        return True

    def build_graph(self) -> nx.MultiGraph:
        """
        Build NetworkX graph from wire segments and component pins.

        Returns:
            NetworkX MultiGraph representing the schematic
        """
        self.graph = nx.MultiGraph()

        # Add wire segments as edges
        for seg_id, segment in self.wire_segments.items():
            start_node = f"wire_{segment.start[0]}_{segment.start[1]}"
            end_node = f"wire_{segment.end[0]}_{segment.end[1]}"

            self.graph.add_edge(
                start_node,
                end_node,
                key=seg_id,
                segment_id=seg_id,
                weight=segment.length,
            )

        # Add component pins as nodes
        for pin in self.pins:
            node_id = f"pin_{pin.component_id}_{pin.pin_number}"
            self.graph.add_node(
                node_id,
                type="pin",
                component_id=pin.component_id,
                pin_number=pin.pin_number,
                coord=pin.coord,
            )

        # Add junctions (T and X crossings)
        self._detect_junctions()

        return self.graph

    def _detect_junctions(self) -> List[Tuple[int, int, str]]:
        """
        Detect T-junctions and X-crossings in the wire network.

        Returns:
            List of (x, y, junction_type) tuples
        """
        junctions = []

        # Build coordinate to node mapping
        coord_to_nodes: Dict[Tuple[int, int], List[str]] = {}

        for node in self.graph.nodes():
            if node.startswith("wire_"):
                _, x, y = node.rsplit("_", 2)
                coord = (int(x), int(y))
                if coord not in coord_to_nodes:
                    coord_to_nodes[coord] = []
                coord_to_nodes[coord].append(node)

        # Detect nodes with degree > 2 (junctions)
        for coord, nodes in coord_to_nodes.items():
            if len(nodes) > 2:
                # T or X junction
                junction_type = "T" if len(nodes) == 3 else "X"
                junctions.append((*coord, junction_type))

                # Update node attributes
                for node in nodes:
                    if self.graph.has_node(node):
                        self.graph.nodes[node]["is_junction"] = True
                        self.graph.nodes[node]["junction_type"] = junction_type

        return junctions

    def extract_nets(
        self,
        components: List[Dict],
    ) -> List[Dict]:
        """
        Extract electrical nets from the graph.

        Args:
            components: List of component dictionaries with pins

        Returns:
            List of net dictionaries
        """
        nets = []
        net_id_counter = 0

        # Find connected components in the graph
        for component in nx.connected_components(self.graph):
            subgraph = self.graph.subgraph(component)

            # Get pin connections
            connections = []
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                if node_data.get("type") == "pin":
                    connections.append({
                        "component_id": node_data["component_id"],
                        "pin": node_data["pin_number"],
                    })

            # Extract wire path
            path_points = []
            for u, v, data in subgraph.edges(data=True):
                if u.startswith("wire_"):
                    _, x1, y1 = u.rsplit("_", 2)
                    path_points.append((int(x1), int(y1)))
                if v.startswith("wire_"):
                    _, x2, y2 = v.rsplit("_", 2)
                    path_points.append((int(x2), int(y2)))

            # Deduplicate path points
            if path_points:
                path_points = self._deduplicate_path(path_points)

            # Calculate confidence
            confidence = 1.0
            if len(connections) < 2:
                confidence *= 0.5  # Incomplete net

            net = {
                "net_id": f"NET_{net_id_counter:03d}",
                "connections": connections,
                "path_points": path_points,
                "confidence": confidence,
            }

            nets.append(net)
            net_id_counter += 1

        return nets

    def _deduplicate_path(
        self,
        points: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Remove duplicate adjacent points from path."""
        if not points:
            return []

        deduped = [points[0]]
        for point in points[1:]:
            if point != deduped[-1]:
                deduped.append(point)

        return deduped

    def associate_text_labels(
        self,
        text_labels: List[Dict],
    ) -> Dict[str, str]:
        """
        Associate text labels with nearby components.

        Args:
            text_labels: List of {'text': str, 'coord': (x, y)} dicts

        Returns:
            Mapping of component_id to associated labels
        """
        associations = {}

        if not self.wire_segments:
            return associations

        # Build k-d tree of segment midpoints
        segment_midpoints = np.array([
            seg.midpoint for seg in self.wire_segments.values()
        ])
        segment_ids = list(self.wire_segments.keys())

        if not segment_midpoints.size:
            return associations

        seg_tree = KDTree(segment_midpoints)

        for label in text_labels:
            text = label.get("text", "")
            coord = label.get("coord")

            if not coord or not text:
                continue

            # Find nearest wire segment
            dist, idx = seg_tree.query(coord, k=1)

            if dist <= self.text_association_radius:
                seg_id = segment_ids[idx]
                associations[f"segment_{seg_id}"] = text

        return associations

    def visualize(self, output_path: Optional[Path] = None) -> np.ndarray:
        """
        Create visualization of the graph.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Visualization as numpy array
        """
        # Create blank canvas
        if not self.wire_segments and not self.pins:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Get bounds
        all_coords = []
        for seg in self.wire_segments.values():
            all_coords.extend([seg.start, seg.end])
        for pin in self.pins:
            all_coords.append(pin.coord)

        if not all_coords:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        min_x = min(c[0] for c in all_coords)
        max_x = max(c[0] for c in all_coords)
        min_y = min(c[1] for c in all_coords)
        max_y = max(c[1] for c in all_coords)

        # Add padding
        padding = 50
        width = max_x - min_x + padding * 2
        height = max_y - min_y + padding * 2

        # Create visualization
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw wire segments
        for seg_id, segment in self.wire_segments.items():
            x1 = segment.start[0] - min_x + padding
            y1 = segment.start[1] - min_y + padding
            x2 = segment.end[0] - min_x + padding
            y2 = segment.end[1] - min_y + padding

            color = (100, 100, 100) if segment.confidence > 0.5 else (200, 100, 100)
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)

        # Draw pins
        for pin in self.pins:
            x = pin.coord[0] - min_x + padding
            y = pin.coord[1] - min_y + padding
            cv2.circle(vis, (x, y), 5, (0, 200, 0), -1)

        # Draw junctions
        for coord, nodes in self._group_nodes_by_coord().items():
            if len(nodes) > 2:
                x = coord[0] - min_x + padding
                y = coord[1] - min_y + padding
                cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)

        if output_path:
            import cv2
            cv2.imwrite(str(output_path), vis)

        return vis

    def _group_nodes_by_coord(self) -> Dict[Tuple[int, int], List[str]]:
        """Group wire nodes by their coordinates."""
        coord_to_nodes: Dict[Tuple[int, int], List[str]] = {}

        for node in self.graph.nodes():
            if node.startswith("wire_"):
                _, x, y = node.rsplit("_", 2)
                coord = (int(x), int(y))
                if coord not in coord_to_nodes:
                    coord_to_nodes[coord] = []
                coord_to_nodes[coord].append(node)

        return coord_to_nodes


# Import cv2 for visualization
import cv2
