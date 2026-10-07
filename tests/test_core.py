"""Tests for AVERS core modules."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from avers.core.types import (
    Component,
    Pin,
    Net,
    WireConnection,
    SchemaMetadata,
    AVERSManifest,
    HumanReviewIssue,
    ComponentType,
    BoundingBox,
)
from avers.config import AVERSConfig, SlicingConfig, GraphSynthesisConfig
from avers.stages.stage1_slicing import SlicingEngine, Tile, DetectionBox
from avers.stages.stage5_graph_synthesis import GraphBuilder, WireSegment, PinReference


class TestTypes:
    """Test Pydantic data types."""

    def test_component_creation(self):
        """Test Component model creation."""
        component = Component(
            id="comp_001",
            designator="X1",
            type=ComponentType.CONNECTOR,
            part_number="СНЦ144-6/10РО11",
            bbox=(1240, 500, 1480, 890),
            pins=[
                Pin(pin_number="1", coord=(1480, 520)),
                Pin(pin_number="2", coord=(1480, 560)),
            ],
        )

        assert component.id == "comp_001"
        assert component.designator == "X1"
        assert component.type == ComponentType.CONNECTOR
        assert len(component.pins) == 2

    def test_bbox_properties(self):
        """Test BoundingBox properties."""
        bbox = BoundingBox(x_min=100, y_min=50, x_max=200, y_max=150)

        assert bbox.center == (150, 100)
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.as_tuple == (100, 50, 200, 150)

    def test_net_creation(self):
        """Test Net model creation."""
        net = Net(
            net_id="NET_001",
            net_name="+27V",
            wire_type="БПВЛ-0.35",
            connections=[
                WireConnection(component_id="comp_001", pin="1"),
                WireConnection(component_id="comp_002", pin="3"),
            ],
            path_points=[(100, 100), (200, 100), (200, 200)],
        )

        assert net.net_id == "NET_001"
        assert len(net.connections) == 2
        assert net.confidence == 1.0

    def test_manifest_json_export(self):
        """Test manifest JSON serialization."""
        manifest = AVERSManifest(
            schema_metadata=SchemaMetadata(
                source_file="test.tif",
                resolution_dpi=300,
                width=1024,
                height=768,
            ),
            components=[],
            nets=[],
        )

        json_str = manifest.to_json()
        assert "test.tif" in json_str
        assert '"width":1024' in json_str or '"width": 1024' in json_str

    def test_manifest_exports_processing_warnings(self, tmp_path):
        """A partial result must remain marked as partial after export."""
        manifest = AVERSManifest(
            schema_metadata=SchemaMetadata(
                source_file="scan.png", resolution_dpi=300, width=100, height=100,
            ),
            processing_warnings=["Детекция УГО недоступна"],
        )
        json_path = tmp_path / "partial.json"
        xml_path = tmp_path / "partial.xml"
        manifest.save(json_path, format="json")
        manifest.save(xml_path, format="xml")

        assert "Детекция УГО недоступна" in json_path.read_text(encoding="utf-8")
        assert "Детекция УГО недоступна" in xml_path.read_text(encoding="utf-8")

    def test_manifest_save_json(self):
        """Test manifest save to JSON file."""
        manifest = AVERSManifest(
            schema_metadata=SchemaMetadata(
                source_file="test.tif",
                resolution_dpi=300,
                width=1024,
                height=768,
            ),
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manifest.save(path, format="json")
            assert path.exists()
            content = path.read_text()
            assert "test.tif" in content
        finally:
            path.unlink(missing_ok=True)


class TestSlicingEngine:
    """Test Stage 1: SAHI-based image slicing."""

    def test_tile_generation(self):
        """Test tile generation from image."""
        engine = SlicingEngine(tile_size=256, overlap_ratio=0.2)

        # Create test image
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        tiles = list(engine.generate_tiles(image))
        assert len(tiles) > 0
        assert all(isinstance(t, Tile) for t in tiles)

    def test_tile_properties(self):
        """Test Tile properties."""
        engine = SlicingEngine(tile_size=256, overlap_ratio=0.2)
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        tiles = list(engine.generate_tiles(image))
        tile = tiles[0]

        assert tile.tile_id == 0
        assert tile.width <= 256
        assert tile.height <= 256
        assert tile.x_min < tile.x_max
        assert tile.y_min < tile.y_max

    def test_nms_deduplication(self):
        """Test Non-Maximum Suppression."""
        detections = [
            DetectionBox(x_min=10, y_min=10, x_max=50, y_max=50, confidence=0.9, class_id=0),
            DetectionBox(x_min=15, y_min=15, x_max=55, y_max=55, confidence=0.8, class_id=0),
            DetectionBox(x_min=100, y_min=100, x_max=150, y_max=150, confidence=0.85, class_id=0),
        ]

        filtered = SlicingEngine.nms(detections, iou_threshold=0.5)
        assert len(filtered) == 2  # Overlapping ones merged

    def test_detection_coordinate_conversion(self):
        """Test detection coordinate projection."""
        tile = Tile(
            tile_id=0,
            x_min=100,
            y_min=100,
            x_max=356,
            y_max=356,
            width=256,
            height=256,
            image=np.zeros((256, 256, 3), dtype=np.uint8),
        )

        det = DetectionBox(
            x_min=10,
            y_min=10,
            x_max=50,
            y_max=50,
            confidence=0.9,
            tile_id=0,
        )

        global_det = det.to_global(tile)
        assert global_det.x_min == 110
        assert global_det.y_min == 110


class TestGraphBuilder:
    """Test Stage 5: Graph synthesis."""

    def test_add_wire_segment(self):
        """Test adding wire segments."""
        builder = GraphBuilder()

        seg_id = builder.add_wire_segment(
            start=(100, 100),
            end=(200, 200),
            confidence=0.95,
        )

        assert seg_id == 0
        assert len(builder.wire_segments) == 1

    def test_wire_segment_properties(self):
        """Test WireSegment properties."""
        segment = WireSegment(
            segment_id=0,
            start=(100, 100),
            end=(100, 200),
        )

        assert segment.is_vertical
        assert not segment.is_horizontal
        assert segment.length == 100

    def test_pin_snap(self):
        """Test wire snapping to pins."""
        builder = GraphBuilder(snap_enabled=True, snap_radius=15)

        # Add wire
        builder.add_wire_segment(start=(100, 100), end=(200, 100))

        # Add pin nearby
        builder.add_component_pins([
            PinReference(
                component_id="comp_001",
                pin_number="1",
                coord=(202, 100),  # Close to wire end
            )
        ])

        # Snap
        snapping = builder.snap_wire_to_pins()
        assert snapping[0] is not None

    def test_visualize_returns_image(self):
        """Regression test: GraphBuilder.visualize() used to crash with
        UnboundLocalError on `cv2` (a stray `import cv2` was placed inside an
        `if output_path:` branch later in the same method, which made Python
        treat `cv2` as function-local for the *whole* method - including the
        cv2.line()/cv2.circle() calls earlier in the function body that ran
        before that branch). Fixed by importing cv2 at module level."""
        builder = GraphBuilder()
        builder.add_wire_segment(start=(0, 0), end=(100, 100))
        builder.add_component_pins([
            PinReference(component_id="X1", pin_number="1", coord=(50, 50)),
        ])

        vis = builder.visualize()

        assert isinstance(vis, np.ndarray)
        assert vis.ndim == 3 and vis.shape[2] == 3
        # Should have actually drawn something (not a blank white/black canvas).
        assert not np.all(vis == vis[0, 0])

    def test_visualize_empty_graph_returns_blank_canvas(self):
        builder = GraphBuilder()
        vis = builder.visualize()
        assert vis.shape == (100, 100, 3)

    def test_build_graph(self):
        """Test NetworkX graph building."""
        builder = GraphBuilder()

        # Add segments forming a path
        builder.add_wire_segment(start=(0, 0), end=(100, 0))
        builder.add_wire_segment(start=(100, 0), end=(200, 0))

        graph = builder.build_graph()

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_merge_collinear(self):
        """Test collinear segment merging."""
        builder = GraphBuilder(merge_collinear=True, merge_tolerance=10)

        # Add collinear segments
        builder.add_wire_segment(start=(0, 0), end=(100, 0))
        builder.add_wire_segment(start=(100, 0), end=(200, 0))

        initial_count = len(builder.wire_segments)
        merged = builder.merge_collinear_segments()

        assert merged > 0
        assert len(builder.wire_segments) < initial_count

    def test_extract_nets(self):
        """Test net extraction."""
        builder = GraphBuilder()

        # Add component pins
        builder.add_component_pins([
            PinReference(component_id="X1", pin_number="1", coord=(0, 0)),
            PinReference(component_id="X2", pin_number="3", coord=(200, 0)),
        ])

        # Add connecting wire
        builder.add_wire_segment(start=(0, 0), end=(200, 0))

        builder.build_graph()
        nets = builder.extract_nets([])

        assert len(nets) > 0


class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = AVERSConfig()

        assert config.slicing.tile_size == 1024
        assert config.slicing.overlap_ratio == 0.2
        assert config.graph_synthesis.snap_enabled
        assert config.web.port == 8030

    def test_web_port_matches_cli_and_yaml(self):
        """All default Web UI launch paths use the same port."""
        from avers.main import create_main_parser

        assert create_main_parser().parse_args(["web"]).port == 8030
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        assert AVERSConfig.from_yaml(config_path).web.port == 8030

    def test_config_override(self):
        """Test configuration overrides."""
        config = AVERSConfig(
            slicing=SlicingConfig(tile_size=512, overlap_ratio=0.3),
            graph_synthesis=GraphSynthesisConfig(snap_radius=20),
        )

        assert config.slicing.tile_size == 512
        assert config.slicing.overlap_ratio == 0.3
        assert config.graph_synthesis.snap_radius == 20

    def test_config_yaml_roundtrip(self):
        """Test YAML serialization."""
        config = AVERSConfig(
            slicing=SlicingConfig(tile_size=512),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        try:
            config.to_yaml(path)
            loaded = AVERSConfig.from_yaml(path)

            assert loaded.slicing.tile_size == 512
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
