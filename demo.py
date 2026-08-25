#!/usr/bin/env python3
"""
AVERS Full Demo - Complete demonstration of all 6 stages.

This script demonstrates the full AVERS pipeline capabilities.
"""

import numpy as np
import sys
import cv2

from avers.core.types import (
    Component,
    Pin,
    Net,
    WireConnection,
    SchemaMetadata,
    AVERSManifest,
    ComponentType,
)
from avers.stages.stage1_slicing import SlicingEngine
from avers.stages.stage2_detection import YOLODetector, DetectionConfig
from avers.stages.stage3_ocr import OCRConfig, classify_text, normalize_designation
from avers.stages.stage4_vectorization import vectorize_wires, VectorizationConfig, visualize_vectorization
from avers.stages.stage5_graph_synthesis import GraphBuilder, PinReference
from avers.stages.stage6_vlm_arbitrator import VLMWrapper, VLMConfig, ArbitrationEngine


def create_test_schematic(width: int = 2000, height: int = 1000) -> np.ndarray:
    """Create a synthetic schematic image for testing."""
    # White background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw connector X1 (left side)
    cv2.rectangle(image, (100, 200), (180, 400), (100, 100, 100), 2)
    cv2.putText(image, "X1", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Pins
    cv2.circle(image, (180, 250), 5, (0, 0, 0), -1)
    cv2.circle(image, (180, 300), 5, (0, 0, 0), -1)
    cv2.circle(image, (180, 350), 5, (0, 0, 0), -1)
    
    # Draw connector X2 (right side)
    cv2.rectangle(image, (900, 150), (980, 350), (100, 100, 100), 2)
    cv2.putText(image, "X2", (910, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(image, (900, 200), 5, (0, 0, 0), -1)
    cv2.circle(image, (900, 250), 5, (0, 0, 0), -1)
    
    # Draw wires
    # Wire 1: X1-1 to X2-1 (horizontal then vertical)
    cv2.line(image, (180, 250), (500, 250), (0, 0, 0), 3)
    cv2.line(image, (500, 250), (500, 450), (0, 0, 0), 3)
    cv2.line(image, (500, 450), (900, 200), (0, 0, 0), 3)
    
    # Wire 2: X1-2 to X2-2
    cv2.line(image, (180, 300), (400, 300), (0, 0, 0), 3)
    cv2.line(image, (400, 300), (400, 500), (0, 0, 0), 3)
    cv2.line(image, (400, 500), (900, 250), (0, 0, 0), 3)
    
    # Wire 3: X1-3 to ground
    cv2.line(image, (180, 350), (600, 350), (0, 0, 0), 3)
    cv2.line(image, (600, 350), (600, 500), (0, 0, 0), 3)
    cv2.line(image, (600, 500), (650, 550), (0, 0, 0), 3)
    cv2.line(image, (630, 550), (670, 550), (0, 0, 0), 3)
    cv2.line(image, (640, 550), (640, 580), (0, 0, 0), 3)
    cv2.line(image, (660, 550), (660, 580), (0, 0, 0), 3)
    
    # Junction dot at wire intersection
    cv2.circle(image, (400, 300), 5, (0, 0, 0), -1)
    
    # Text labels
    cv2.putText(image, "+27В", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image, "БПВЛ-0.35", (450, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
    cv2.putText(image, "1", (190, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "2", (190, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "3", (190, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add some noise (scanning artifacts)
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image


def demo_stage1_slicing(image: np.ndarray):
    """Demonstrate Stage 1: SAHI-based Image Slicing."""
    print("\n" + "=" * 60)
    print("Stage 1: SAHI-based Image Slicing")
    print("=" * 60)
    
    engine = SlicingEngine(tile_size=512, overlap_ratio=0.25)
    tiles = list(engine.generate_tiles(image))
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]} px")
    print(f"Tile size: 512x512 with 25% overlap")
    print(f"Tiles generated: {len(tiles)}")
    
    # Show first few tiles
    for tile in tiles[:6]:
        print(f"  Tile {tile.tile_id}: ({tile.x_min},{tile.y_min})-({tile.x_max},{tile.y_max})")


def demo_stage2_detection(image: np.ndarray):
    """Demonstrate Stage 2: УГО Detection."""
    print("\n" + "=" * 60)
    print("Stage 2: УГО Detection")
    print("=" * 60)
    
    config = DetectionConfig(device="cpu")
    detector = YOLODetector(config)
    detector.load()
    
    # Detect in full image (simulating SAHI would be parallel)
    detections = detector.predict(image)
    
    print(f"Detected {len(detections)} objects")
    
    # Classify by type
    type_counts = {}
    for det in detections:
        name = det.class_name or f"class_{det.class_id}"
        type_counts[name] = type_counts.get(name, 0) + 1
    
    for name, count in type_counts.items():
        print(f"  {name}: {count}")


def demo_stage3_ocr(image: np.ndarray):
    """Demonstrate Stage 3: OCR."""
    print("\n" + "=" * 60)
    print("Stage 3: OCR Text Recognition")
    print("=" * 60)
    
    config = OCRConfig(lang="ru", text_confidence_threshold=0.6)
    
    # Test text classification
    test_texts = ["X1", "2", "+27В", "БПВЛ-0.35", "VD1", "GND", "invalid"]
    
    print("Text classification test:")
    for text in test_texts:
        text_type = classify_text(text)
        normalized = normalize_designation(text)
        print(f"  '{text}' -> type: {text_type}, normalized: '{normalized}'")


def demo_stage4_vectorization(image: np.ndarray):
    """Demonstrate Stage 4: Wire Vectorization."""
    print("\n" + "=" * 60)
    print("Stage 4: Wire Vectorization")
    print("=" * 60)
    
    config = VectorizationConfig(rdp_epsilon=2.0)
    segments, junctions = vectorize_wires(image, config=config)
    
    print(f"Wire segments: {len(segments)}")
    print(f"Junction points: {len(junctions)}")
    
    # Show segment stats
    if segments:
        lengths = [s.length for s in segments]
        print(f"  Length range: {min(lengths):.1f} - {max(lengths):.1f} px")
        print(f"  Average length: {np.mean(lengths):.1f} px")
    
    # Show junction locations
    if junctions:
        print(f"  Junction locations: {list(junctions)[:5]}...")


def demo_stage5_graph_synthesis():
    """Demonstrate Stage 5: Graph Synthesis."""
    print("\n" + "=" * 60)
    print("Stage 5: Graph-based Syntax Synthesis")
    print("=" * 60)
    
    builder = GraphBuilder(snap_radius=15, merge_collinear=True)
    
    # Simulate extracted wires
    print("Adding wire segments...")
    builder.add_wire_segment((180, 250), (500, 250))
    builder.add_wire_segment((500, 250), (500, 450))
    builder.add_wire_segment((500, 450), (900, 200))
    
    builder.add_wire_segment((180, 300), (400, 300))
    builder.add_wire_segment((400, 300), (400, 500))
    builder.add_wire_segment((400, 500), (900, 250))
    
    # Add component pins
    print("Adding component pins...")
    pins = [
        PinReference("X1", "1", (180, 250)),
        PinReference("X1", "2", (180, 300)),
        PinReference("X2", "1", (900, 200)),
        PinReference("X2", "2", (900, 250)),
    ]
    builder.add_component_pins(pins)
    
    # Snap wires to pins
    print("Snapping wires to pins...")
    snapping = builder.snap_wire_to_pins()
    snapped = sum(1 for s in snapping.values() if s)
    print(f"  Snapped {snapped} wire endpoints")
    
    # Build graph
    print("Building graph...")
    graph = builder.build_graph()
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Extract nets
    print("Extracting nets...")
    nets = builder.extract_nets([])
    print(f"  Found {len(nets)} electrical nets")
    
    for net in nets:
        conn_count = len(net.get("connections", []))
        path_len = len(net.get("path_points", []))
        print(f"    {net['net_id']}: {conn_count} connections, {path_len} path points")


def demo_stage6_vlm():
    """Demonstrate Stage 6: VLM Arbitration."""
    print("\n" + "=" * 60)
    print("Stage 6: VLM Arbitration")
    print("=" * 60)
    
    config = VLMConfig()
    vlm = VLMWrapper(config)
    vlm.load()
    
    # Create test ROI
    test_roi = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Add a circle (junction dot)
    cv2.circle(test_roi, (128, 128), 15, (0, 0, 0), -1)
    
    print("Testing junction detection...")
    response = vlm.query(test_roi, "Is there a connection dot at the center?")
    print(f"  Response: {response}")


def demo_output_manifest():
    """Demonstrate output format."""
    print("\n" + "=" * 60)
    print("Output Format: AVERSManifest")
    print("=" * 60)
    
    manifest = AVERSManifest(
        schema_metadata=SchemaMetadata(
            source_file="demo_board.tif",
            resolution_dpi=300,
            width=2000,
            height=1000,
        ),
        components=[
            Component(
                id="comp_001",
                designator="X1",
                type=ComponentType.CONNECTOR,
                part_number="СНЦ144-6/10РО11",
                bbox=(100, 200, 180, 400),
                pins=[
                    Pin(pin_number="1", coord=(180, 250)),
                    Pin(pin_number="2", coord=(180, 300)),
                    Pin(pin_number="3", coord=(180, 350)),
                ],
            ),
            Component(
                id="comp_002",
                designator="X2",
                type=ComponentType.CONNECTOR,
                bbox=(900, 150, 980, 350),
                pins=[
                    Pin(pin_number="1", coord=(900, 200)),
                    Pin(pin_number="2", coord=(900, 250)),
                ],
            ),
        ],
        nets=[
            Net(
                net_id="NET_001",
                net_name="+27V",
                wire_type="БПВЛ-0.35",
                wire_color="К",
                connections=[
                    WireConnection(component_id="comp_001", pin="1"),
                    WireConnection(component_id="comp_002", pin="1"),
                ],
                path_points=[(180, 250), (500, 250), (500, 450), (900, 200)],
                confidence=0.95,
            ),
        ],
    )
    
    print(f"\nManifest:")
    print(f"  Source: {manifest.schema_metadata.source_file}")
    print(f"  Components: {len(manifest.components)}")
    print(f"  Nets: {len(manifest.nets)}")
    
    print(f"\nJSON output:")
    json_str = manifest.to_json()
    print("  " + json_str[:400].replace("\n", "\n  ") + "...")


def main():
    """Run full demo."""
    print("\n" + "=" * 60)
    print("AVERS Full Demo - All 6 Stages")
    print("Automated Vectorization and Recognition of Schematics")
    print("=" * 60)
    
    # Create test schematic
    print("\nCreating test schematic...")
    image = create_test_schematic()
    print(f"Created {image.shape[1]}x{image.shape[0]} test image")
    
    # Run all stage demos
    try:
        demo_stage1_slicing(image)
    except Exception as e:
        print(f"  Stage 1 demo error: {e}")
    
    try:
        demo_stage2_detection(image)
    except Exception as e:
        print(f"  Stage 2 demo error: {e}")
    
    try:
        demo_stage3_ocr(image)
    except Exception as e:
        print(f"  Stage 3 demo error: {e}")
    
    try:
        demo_stage4_vectorization(image)
    except Exception as e:
        print(f"  Stage 4 demo error: {e}")
    
    try:
        demo_stage5_graph_synthesis()
    except Exception as e:
        print(f"  Stage 5 demo error: {e}")
    
    try:
        demo_stage6_vlm()
    except Exception as e:
        print(f"  Stage 6 demo error: {e}")
    
    demo_output_manifest()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nRun full pipeline:")
    print("  python -m avers.main test_schematic.tif --output result.json")
    print("\nRun tests:")
    print("  python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
