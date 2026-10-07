"""Conservative graph joins and drawable geometry for schematic nets."""

import networkx as nx
import numpy as np
import cv2

from avers.core.types import Component, ComponentType, Net, Pin
from avers.core.validators import ProductionPipeline
from avers.stages.stage5_graph_synthesis import GraphBuilder, PinReference, WireSegment


def test_t_endpoint_splits_horizontal_and_keeps_branches_separate():
    builder = GraphBuilder()
    builder.add_wire_segment((0, 10), (100, 10))
    builder.add_wire_segment((50, 50), (50, 10))
    graph = builder.build_graph()

    assert nx.has_path(graph, "wire_0_10", "wire_50_50")
    assert graph.nodes["wire_50_10"]["junction_type"] == "T"
    net, = builder.extract_nets([])
    assert net["path_points"] == []  # No flattened path across two branches.
    assert net["wire_segments"] == [
        [(0, 10), (50, 10)],
        [(50, 10), (50, 50)],
        [(50, 10), (100, 10)],
    ]
    # Building twice must not silently add extra edges or mutate the sources.
    assert builder.build_graph().number_of_edges() == 3


def test_near_t_endpoint_connects_with_short_explicit_bridge():
    builder = GraphBuilder()
    builder.add_wire_segment((0, 10), (100, 10))
    builder.add_wire_segment((50, 40), (50, 11))
    builder.build_graph()

    net, = builder.extract_nets([])
    assert [(50, 10), (50, 11)] in net["wire_segments"]
    assert len(net["wire_segments"]) == 4
    assert net["path_points"] == []


def test_x_crossing_without_endpoint_remains_two_nets():
    builder = GraphBuilder()
    builder.add_wire_segment((0, 10), (100, 10))
    builder.add_wire_segment((50, 0), (50, 20))
    graph = builder.build_graph()

    assert not nx.has_path(graph, "wire_0_10", "wire_50_0")
    nets = builder.extract_nets([])
    assert len(nets) == 2
    assert {tuple(net["path_points"]) for net in nets} == {
        ((0, 10), (100, 10)), ((50, 0), (50, 20)),
    }
    assert all(len(net["wire_segments"]) == 1 for net in nets)


def test_close_endpoints_join_but_separate_wires_do_not():
    builder = GraphBuilder(merge_collinear=True)
    builder.add_wire_segment((0, 0), (40, 0))
    builder.add_wire_segment((42, 0), (80, 0))
    builder.add_wire_segment((90, 0), (120, 0))
    builder.build_graph()
    nets = builder.extract_nets([])
    assert len(nets) == 2
    assert any(net["path_points"] == [(0, 0), (40, 0), (80, 0)] for net in nets)
    assert any(net["path_points"] == [(90, 0), (120, 0)] for net in nets)


def test_merge_does_not_swallow_a_pin_at_a_shared_endpoint():
    builder = GraphBuilder()
    builder.add_wire_segment((0, 0), (50, 0))
    builder.add_wire_segment((50, 0), (100, 0))
    builder.add_component_pins([PinReference("X", "1", (50, 0))])
    assert builder.merge_collinear_segments() == 0
    builder.build_graph()
    net, = builder.extract_nets([])
    assert net["connections"] == [{"component_id": "X", "pin": "1"}]
    assert net["path_points"] == [(0, 0), (50, 0), (100, 0)]


def test_small_tolerance_does_not_join_parallel_nearby_wires():
    builder = GraphBuilder(merge_collinear=True)
    builder.add_wire_segment((0, 0), (40, 0))
    builder.add_wire_segment((40, 4), (100, 4))
    assert builder.merge_collinear_segments() == 0
    builder.build_graph()
    assert len(builder.extract_nets([])) == 2


def test_pixel_supported_gap_reconnects_a_continuous_wire():
    image = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 20), (180, 20), (0, 0, 0), 2)
    builder = GraphBuilder(gap_image=image)
    builder.add_wire_segment((10, 20), (80, 20))
    builder.add_wire_segment((110, 20), (180, 20))

    builder.build_graph()
    nets = builder.extract_nets([])
    assert len(nets) == 1
    assert [(80, 20), (110, 20)] in nets[0]["wire_segments"]


def test_supported_gap_can_be_disabled():
    image = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 20), (180, 20), (0, 0, 0), 2)
    builder = GraphBuilder(gap_image=image, max_supported_gap=0)
    builder.add_wire_segment((10, 20), (80, 20))
    builder.add_wire_segment((110, 20), (180, 20))

    builder.build_graph()
    assert len(builder.extract_nets([])) == 2


def test_pixel_evidence_does_not_join_an_undotted_x_crossing():
    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 40), (130, 40), (0, 0, 0), 2)
    cv2.line(image, (70, 10), (70, 70), (0, 0, 0), 2)
    builder = GraphBuilder(gap_image=image)
    builder.add_wire_segment((10, 40), (130, 40))
    builder.add_wire_segment((70, 10), (70, 70))

    builder.build_graph()
    assert len(builder.extract_nets([])) == 2


def test_pixel_gap_rejects_missing_ink_between_two_wires():
    image = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 20), (80, 20), (0, 0, 0), 2)
    cv2.line(image, (110, 20), (180, 20), (0, 0, 0), 2)
    builder = GraphBuilder(gap_image=image)
    builder.add_wire_segment((10, 20), (80, 20))
    builder.add_wire_segment((110, 20), (180, 20))

    builder.build_graph()
    assert len(builder.extract_nets([])) == 2


def test_pin_snap_at_both_ends_attaches_to_same_wire_net():
    builder = GraphBuilder(snap_radius=5)
    builder.add_wire_segment((1, 0), (99, 0))
    builder.add_component_pins([
        PinReference("X1", "1", (0, 0)),
        PinReference("X2", "2", (100, 0)),
    ])
    builder.snap_wire_to_pins()
    graph = builder.build_graph()

    assert nx.has_path(graph, "pin_X1_1", "pin_X2_2")
    net, = builder.extract_nets([])
    assert net["connections"] == [
        {"component_id": "X1", "pin": "1"},
        {"component_id": "X2", "pin": "2"},
    ]
    assert net["path_points"] == [(0, 0), (100, 0)]
    assert net["wire_segments"] == [[(0, 0), (100, 0)]]


def test_branched_net_never_invents_diagonal_between_unrelated_edges():
    builder = GraphBuilder()
    builder.add_wire_segment((0, 0), (100, 0))
    builder.add_wire_segment((50, 0), (50, 40))
    builder.build_graph()
    net, = builder.extract_nets([])

    assert net["path_points"] == []
    assert all(a[0] == b[0] or a[1] == b[1] for a, b in net["wire_segments"])
    assert len(net["wire_segments"]) == 3


def test_pipeline_passes_drawable_segments_into_manifest():
    segments = [
        WireSegment(segment_id=0, start=(0, 0), end=(100, 0)),
        WireSegment(segment_id=1, start=(50, 0), end=(50, 40)),
    ]
    components = [Component(
        id="X1", designator="X1", type=ComponentType.CONNECTOR,
        bbox=(0, 0, 2, 2), pins=[Pin(pin_number="1", coord=(0, 0))],
    )]
    net, = ProductionPipeline()._run_graph_synthesis(segments, set(), components)
    assert isinstance(net, Net)
    assert net.connections[0].component_id == "X1"
    assert net.path_points == []
    assert len(net.wire_segments) == 3
    assert len(net.model_dump()["wire_segments"]) == 3
