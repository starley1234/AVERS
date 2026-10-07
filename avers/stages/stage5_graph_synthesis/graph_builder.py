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
from math import hypot
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import cv2
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
        junction_tolerance: float = 2.0,
        gap_image: Optional[np.ndarray] = None,
        max_supported_gap: int = 60,
    ):
        """
        Initialize graph builder.

        Args:
            snap_enabled: Enable wire snapping to pins
            snap_radius: Snap radius in pixels
            text_association_radius: Max distance for text association
            merge_collinear: Merge collinear wire segments
            merge_tolerance: Tolerance for collinear merge
            junction_tolerance: Maximum endpoint/segment gap to join (pixels)
            gap_image: Source image for verifying longer collinear wire gaps
            max_supported_gap: Largest gap joined only with continuous ink evidence
        """
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.text_association_radius = text_association_radius
        self.merge_collinear = merge_collinear
        self.merge_tolerance = merge_tolerance
        self.junction_tolerance = junction_tolerance
        self.gap_image = gap_image
        self.max_supported_gap = max_supported_gap

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

            # Both ends may terminate at different pins. Keep the historical
            # return value (the first match), but attach both in build_graph.
            for end_name in ("start", "end"):
                endpoint = getattr(segment, end_name)
                dist, idx = self.component_pins_kdtree.query(
                    endpoint, k=1, distance_upper_bound=self.snap_radius,
                )
                if dist <= self.snap_radius:
                    pin = self.pins[idx]
                    if snapped is None:
                        snapped = pin
                    setattr(segment, end_name, pin.coord)
            segment.__post_init__()
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
        self._pin_coords = {tuple(pin.coord) for pin in self.pins}
        radius = min(self.merge_tolerance, self.junction_tolerance)
        # Restart after each merge so a segment is never consumed twice.
        # Candidate pairs come from a k-d tree over endpoints (only segments
        # with endpoints closer than ``radius`` can merge), which keeps this
        # near-linear on noisy scans with thousands of Hough segments. The
        # first pair in (first_id, second_id) order is chosen, exactly as the
        # previous exhaustive O(n^2) scan did.
        while True:
            segment_ids = sorted(self.wire_segments)
            owners = []
            coords = []
            for sid in segment_ids:
                seg = self.wire_segments[sid]
                for point in (seg.start, seg.end):
                    owners.append(sid)
                    coords.append(point)
            candidates = set()
            if len(coords) >= 2:
                tree = KDTree(np.asarray(coords, dtype=float))
                for i, j in tree.query_pairs(radius + 1e-9):
                    a, b = owners[i], owners[j]
                    if a != b:
                        candidates.add((min(a, b), max(a, b)))
            pair = next(
                ((first, second) for first, second in sorted(candidates)
                 if self._are_collinear_and_connected(
                     self.wire_segments[first], self.wire_segments[second],
                 )),
                None,
            )
            if pair is None:
                break
            seg1, seg2 = (self.wire_segments.pop(sid) for sid in pair)
            endpoints = (seg1.start, seg1.end, seg2.start, seg2.end)
            new_start, new_end = min(endpoints), max(endpoints)
            self.wire_segments[self.next_segment_id] = WireSegment(
                segment_id=self.next_segment_id,
                start=new_start,
                end=new_end,
                points=[new_start, new_end],
                confidence=min(seg1.confidence, seg2.confidence),
            )
            self.next_segment_id += 1
            merged += 1
        return merged

    def _are_collinear_and_connected(
        self,
        seg1: WireSegment,
        seg2: WireSegment,
        angle_tolerance: float = 5.0,
    ) -> bool:
        """Merge only straight, aligned wires sharing an unpinned endpoint."""
        if (len(seg1.points) > 2 or len(seg2.points) > 2):
            return False
        horizontal = (seg1.start[1] == seg1.end[1] ==
                      seg2.start[1] == seg2.end[1])
        vertical = (seg1.start[0] == seg1.end[0] ==
                    seg2.start[0] == seg2.end[0])
        if not (horizontal or vertical):
            return False
        pin_coords = getattr(self, "_pin_coords", None)
        if pin_coords is None:
            pin_coords = {tuple(pin.coord) for pin in self.pins}
        for a in (seg1.start, seg1.end):
            for b in (seg2.start, seg2.end):
                if hypot(a[0] - b[0], a[1] - b[1]) <= min(
                    self.merge_tolerance, self.junction_tolerance,
                ) and tuple(a) not in pin_coords and tuple(b) not in pin_coords:
                    return True
        return False

    @staticmethod
    def _wire_node(coord: Tuple[int, int]) -> str:
        return f"wire_{coord[0]}_{coord[1]}"

    @staticmethod
    def _project(point, start, end):
        """Closest point on a finite leg, with its normalized position."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if not length_sq:
            return 0.0, start, hypot(point[0] - start[0], point[1] - start[1])
        t = max(0.0, min(1.0, ((point[0] - start[0]) * dx +
                                 (point[1] - start[1]) * dy) / length_sq))
        x, y = start[0] + t * dx, start[1] + t * dy
        coord = (round(x), round(y))
        return t, coord, hypot(point[0] - x, point[1] - y)

    def _add_wire_edge(self, start, end, segment_id=None):
        if start == end:
            return
        for coord in (start, end):
            self.graph.add_node(self._wire_node(coord), type="wire", coord=coord)
        self.graph.add_edge(
            self._wire_node(start), self._wire_node(end),
            kind="wire", segment_id=segment_id,
            points=[start, end], weight=hypot(end[0] - start[0], end[1] - start[1]),
        )

    def _gap_has_ink(self, start: Tuple[int, int], end: Tuple[int, int], gray: np.ndarray) -> bool:
        """Require an almost continuous thin stroke before bridging a Hough gap."""
        length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        if length < 3 or length > self.max_supported_gap:
            return False
        height, width = gray.shape[:2]
        samples = 0
        covered = 0
        for t in np.linspace(0, 1, length + 1)[1:-1]:
            x = round(start[0] + t * (end[0] - start[0]))
            y = round(start[1] + t * (end[1] - start[1]))
            if not (0 <= x < width and 0 <= y < height):
                return False
            samples += 1
            covered += bool(np.any(gray[max(0, y - 1):min(height, y + 2),
                                        max(0, x - 1):min(width, x + 2)] < 160))
        return samples > 0 and covered / samples >= 0.9

    def _connect_supported_gaps(self) -> None:
        """Join collinear wire ends only when the source image shows ink between them."""
        if self.gap_image is None or self.max_supported_gap < 3:
            return
        gray = (cv2.cvtColor(self.gap_image, cv2.COLOR_RGB2GRAY)
                if self.gap_image.ndim == 3 else self.gap_image)
        leaves = [node for node, data in self.graph.nodes(data=True)
                  if data.get("type") == "wire" and self.graph.degree(node) == 1]
        if len(leaves) < 2:
            return
        coords = np.array([self.graph.nodes[node]["coord"] for node in leaves])
        tree = KDTree(coords)
        candidates = sorted(tree.query_pairs(self.max_supported_gap),
                            key=lambda pair: (hypot(*(coords[pair[0]] - coords[pair[1]])), pair))
        connected = nx.utils.UnionFind(self.graph.nodes)
        for a, b in self.graph.edges():
            connected.union(a, b)
        for i, j in candidates:
            a, b = leaves[i], leaves[j]
            if connected[a] == connected[b]:
                continue
            start = tuple(map(int, coords[i]))
            end = tuple(map(int, coords[j]))
            dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
            horizontal = dy <= self.junction_tolerance and dx >= 3
            vertical = dx <= self.junction_tolerance and dy >= 3
            if not (horizontal or vertical):
                continue
            if (horizontal and start[0] > end[0]) or (vertical and start[1] > end[1]):
                a, b = b, a
                start, end = end, start
            # Both ends must point toward the gap, not be parallel tails.
            aligned = True
            for node, is_first in ((a, True), (b, False)):
                neighbor = next(iter(self.graph.neighbors(node)))
                x, y = self.graph.nodes[node]["coord"]
                nx_, ny_ = self.graph.nodes[neighbor]["coord"]
                if horizontal and (abs(y - ny_) > 2 or
                                   (nx_ >= x if is_first else nx_ <= x)):
                    aligned = False
                if vertical and (abs(x - nx_) > 2 or
                                 (ny_ >= y if is_first else ny_ <= y)):
                    aligned = False
            if aligned and self._gap_has_ink(start, end, gray):
                self._add_wire_edge(start, end)
                connected.union(a, b)

    def build_graph(self) -> nx.MultiGraph:
        """Build a conservative topology: only endpoints can establish joins.

        A T endpoint splits the leg it touches; a crossing between two leg
        interiors is never joined without explicit junction evidence.
        """
        self.graph = nx.MultiGraph()
        segments = sorted(self.wire_segments.items())
        endpoints = sorted({point for _, seg in segments for point in (seg.start, seg.end)})
        # Canonicalize close endpoints, without modifying the input segments.
        # Same result as "first earlier representative within tolerance", but
        # looked up in a uniform grid instead of scanning every representative.
        tol = self.junction_tolerance
        cell = max(1.0, float(tol))
        canonical = {}
        rep_grid: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]] = {}
        rep_count = 0
        for point in endpoints:
            gx, gy = int(point[0] // cell), int(point[1] // cell)
            best = None
            for cx in (gx - 1, gx, gx + 1):
                for cy in (gy - 1, gy, gy + 1):
                    for order, p in rep_grid.get((cx, cy), ()):
                        if (hypot(point[0] - p[0], point[1] - p[1]) <= tol
                                and (best is None or order < best[0])):
                            best = (order, p)
            if best is None:
                match = point
                rep_grid.setdefault((gx, gy), []).append((rep_count, point))
                rep_count += 1
            else:
                match = best[1]
            canonical[point] = match

        legs = {}
        for seg_id, seg in segments:
            # Stage 4 supplies full polylines. The stored endpoints may have
            # changed since then when snap_wire_to_pins() was called.
            points = [canonical[seg.start], *seg.points[1:-1], canonical[seg.end]]
            for index, (a, b) in enumerate(zip(points, points[1:])):
                if a != b:
                    legs[(seg_id, index)] = (a, b)

        splits = {key: [(0.0, a), (1.0, b)] for key, (a, b) in legs.items()}
        joins = set()
        # Spatial grid over leg bounding boxes (expanded by the tolerance), so
        # each endpoint is projected only onto nearby legs.
        leg_cell = 32
        leg_grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for key, (a, b) in legs.items():
            x0 = int((min(a[0], b[0]) - tol) // leg_cell)
            x1 = int((max(a[0], b[0]) + tol) // leg_cell)
            y0 = int((min(a[1], b[1]) - tol) // leg_cell)
            y1 = int((max(a[1], b[1]) + tol) // leg_cell)
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    leg_grid.setdefault((gx, gy), []).append(key)
        for seg_id, seg in segments:
            for raw in (seg.start, seg.end):
                point = canonical[raw]
                nearby = sorted(set(leg_grid.get(
                    (int(point[0] // leg_cell), int(point[1] // leg_cell)), ())))
                for other_id, index in nearby:
                    a, b = legs[(other_id, index)]
                    if other_id == seg_id:
                        continue
                    t, target, distance = self._project(point, a, b)
                    if distance > self.junction_tolerance:
                        continue
                    # A rounded projection at a leg end must use its existing node.
                    if t == 0.0:
                        target = a
                    elif t == 1.0:
                        target = b
                    splits[(other_id, index)].append((t, target))
                    if point != target:
                        joins.add(tuple(sorted((point, target))))

        for (seg_id, index), candidates in sorted(splits.items()):
            ordered = sorted(candidates)
            for (_, a), (_, b) in zip(ordered, ordered[1:]):
                self._add_wire_edge(a, b, segment_id=seg_id)
        for a, b in sorted(joins):
            # The short snapped bridge is genuine geometry, not an arbitrary
            # link between unrelated edges of a net.
            self._add_wire_edge(a, b)

        self._connect_supported_gaps()

        for pin in self.pins:
            node_id = f"pin_{pin.component_id}_{pin.pin_number}"
            self.graph.add_node(
                node_id, type="pin", component_id=pin.component_id,
                pin_number=pin.pin_number, coord=pin.coord,
            )
            # Pins attach only to actual (possibly pre-snapped) wire endpoints.
            if pin.coord in canonical:
                wire_node = self._wire_node(canonical[pin.coord])
                if self.graph.has_node(wire_node):
                    self.graph.add_edge(node_id, wire_node, kind="pin")

        self._detect_junctions()
        return self.graph

    def _detect_junctions(self) -> List[Tuple[int, int, str]]:
        """Label connected wire nodes with three or more incident wire legs."""
        junctions = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "wire":
                continue
            degree = sum(attributes.get("kind") == "wire"
                         for parallel_edges in self.graph[node].values()
                         for attributes in parallel_edges.values())
            if degree >= 3:
                kind = "T" if degree == 3 else "X"
                data["is_junction"] = True
                data["junction_type"] = kind
                junctions.append((*data["coord"], kind))
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

        # Stable ordering independent of NetworkX's edge traversal order.
        for component in sorted(nx.connected_components(self.graph), key=lambda c: min(c)):
            subgraph = self.graph.subgraph(component)
            connections = sorted(
                ({"component_id": data["component_id"], "pin": data["pin_number"]}
                 for _, data in subgraph.nodes(data=True) if data.get("type") == "pin"),
                key=lambda c: (c["component_id"], c["pin"]),
            )

            wire_graph = nx.Graph()
            wire_segments = []
            for u, v, data in subgraph.edges(data=True):
                if data.get("kind") != "wire":
                    continue
                a, b = data["points"]
                wire_segments.append([min(a, b), max(a, b)])
                wire_graph.add_edge(u, v)
            wire_segments.sort()

            # A single path remains backwards compatible. A branch/cycle
            # cannot be flattened into one polyline without inventing strokes.
            leaves = [node for node, degree in wire_graph.degree() if degree == 1]
            path_points = []
            if len(leaves) == 2 and all(degree <= 2 for _, degree in wire_graph.degree()):
                start, end = sorted(leaves, key=lambda node: self.graph.nodes[node]["coord"])
                path = nx.shortest_path(wire_graph, start, end)
                path_points = [self.graph.nodes[node]["coord"] for node in path]

            confidence = 1.0 if len(connections) >= 2 else 0.5
            nets.append({
                "net_id": f"NET_{net_id_counter:03d}",
                "connections": connections,
                "path_points": path_points,
                "wire_segments": wire_segments,
                "confidence": confidence,
            })
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

        # Draw only joined junctions (not interior/interior crossings).
        for _, data in self.graph.nodes(data=True):
            if data.get("is_junction"):
                x, y = data["coord"]
                cv2.circle(vis, (x - min_x + padding, y - min_y + padding),
                           8, (0, 0, 255), -1)

        if output_path:
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
