#!/usr/bin/env python3
"""
AVERS Demo - Quick demonstration of core functionality.

This script demonstrates the key features of the AVERS pipeline
without requiring actual ML models or large test images.
"""

import numpy as np
import sys

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
from avers.stages.stage5_graph_synthesis import GraphBuilder, PinReference
from avers.config import AVERSConfig


def create_sample_image(width: int = 2000, height: int = 1000) -> np.ndarray:
    """Create a synthetic test image."""
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw some simulated wire traces
    import cv2

    # Horizontal wire
    cv2.line(image, (200, 200), (800, 200), (0, 0, 0), 3)

    # Vertical wire
    cv2.line(image, (500, 150), (500, 400), (0, 0, 0), 3)

    # Another horizontal wire
    cv2.line(image, (400, 400), (900, 400), (0, 0, 0), 3)

    # Simulate connector at (200, 200)
    cv2.rectangle(image, (180, 180), (220, 220), (128, 128, 128), 2)
    cv2.putText(image, "X1", (175, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Simulate connector at (900, 400)
    cv2.rectangle(image, (880, 380), (920, 420), (128, 128, 128), 2)
    cv2.putText(image, "X2", (875, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Simulate junction dot
    cv2.circle(image, (500, 200), 5, (0, 0, 0), -1)

    return image


def demo_slicing():
    """Demonstrate Stage 1: Image slicing."""
    print("\n" + "=" * 60)
    print("Stage 1: SAHI-based Image Slicing")
    print("=" * 60)

    image = create_sample_image()
    print(f"Created synthetic image: {image.shape[1]}x{image.shape[0]} px")

    engine = SlicingEngine(tile_size=512, overlap_ratio=0.25)
    tiles = list(engine.generate_tiles(image))

    print(f"\nGenerated {len(tiles)} tiles:")
    for tile in tiles[:5]:  # Show first 5
        print(f"  Tile {tile.tile_id}: ({tile.x_min},{tile.y_min}) - "
              f"({tile.x_max},{tile.y_max}), size {tile.width}x{tile.height}")
    if len(tiles) > 5:
        print(f"  ... and {len(tiles) - 5} more tiles")


def demo_graph_synthesis():
    """Demonstrate Stage 5: Graph synthesis."""
    print("\n" + "=" * 60)
    print("Stage 5: Graph-based Syntax Synthesis")
    print("=" * 60)

    builder = GraphBuilder(snap_enabled=True, snap_radius=20)

    # Add simulated wire segments
    print("\nAdding wire segments:")
    builder.add_wire_segment(start=(200, 200), end=(500, 200), confidence=0.95)
    print("  Wire 1: (200,200) → (500,200)")

    builder.add_wire_segment(start=(500, 200), end=(500, 400), confidence=0.95)
    print("  Wire 2: (500,200) → (500,400) [T-junction]")

    builder.add_wire_segment(start=(500, 400), end=(900, 400), confidence=0.95)
    print("  Wire 3: (500,400) → (900,400)")

    # Add component pins
    print("\nAdding component pins:")
    pins = [
        PinReference(component_id="comp_X1", pin_number="1", coord=(200, 200)),
        PinReference(component_id="comp_X2", pin_number="3", coord=(900, 400)),
    ]
    builder.add_component_pins(pins)
    print(f"  X1 Pin 1 at (200, 200)")
    print(f"  X2 Pin 3 at (900, 400)")

    # Snap wires to pins
    print("\nSnapping wires to pins...")
    snapping = builder.snap_wire_to_pins()
    snapped_count = sum(1 for s in snapping.values() if s)
    print(f"  Snapped {snapped_count} wire endpoints to pins")

    # Build graph
    print("\nBuilding NetworkX graph...")
    graph = builder.build_graph()
    print(f"  Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Detect junctions
    junctions = builder._group_nodes_by_coord()
    junction_count = sum(1 for nodes in junctions.values() if len(nodes) > 2)
    print(f"  Found {junction_count} junction point(s)")

    # Extract nets
    print("\nExtracting electrical nets...")
    nets = builder.extract_nets([])
    print(f"  Found {len(nets)} net(s)")
    for net in nets:
        print(f"    {net['net_id']}: {len(net['connections'])} connections, "
              f"confidence={net['confidence']:.2f}")


def demo_output_format():
    """Demonstrate output format."""
    print("\n" + "=" * 60)
    print("Output Format: AVERSManifest")
    print("=" * 60)

    # Create sample manifest
    manifest = AVERSManifest(
        schema_metadata=SchemaMetadata(
            source_file="sample_board.tif",
            resolution_dpi=300,
            width=14000,
            height=3500,
        ),
        components=[
            Component(
                id="comp_001",
                designator="X1",
                type=ComponentType.CONNECTOR,
                part_number="СНЦ144-6/10РО11",
                bbox=(1240, 500, 1480, 890),
                pins=[
                    Pin(pin_number="1", coord=(1480, 520)),
                    Pin(pin_number="2", coord=(1480, 560)),
                    Pin(pin_number="3", coord=(1480, 600)),
                ],
            ),
            Component(
                id="comp_002",
                designator="X2",
                type=ComponentType.CONNECTOR,
                part_number="2РМ14Б4Ш1В1",
                bbox=(8500, 510, 8720, 880),
                pins=[
                    Pin(pin_number="1", coord=(8500, 520)),
                ],
            ),
        ],
        nets=[
            Net(
                net_id="NET_PWR_27V",
                wire_type="БПВЛ-0.35",
                wire_color="К",
                connections=[
                    WireConnection(component_id="comp_001", pin="1"),
                    WireConnection(component_id="comp_002", pin="1"),
                ],
                path_points=[(1480, 520), (3200, 520), (3200, 850), (8500, 520)],
                confidence=0.98,
            ),
        ],
    )

    print("\nManifest structure:")
    print(f"  Source: {manifest.schema_metadata.source_file}")
    print(f"  Dimensions: {manifest.schema_metadata.width}x{manifest.schema_metadata.height}")
    print(f"  Components: {len(manifest.components)}")
    print(f"  Nets: {len(manifest.nets)}")

    print("\nJSON output:")
    json_str = manifest.to_json()
    # Pretty print first 500 chars
    print("  " + json_str[:500].replace("\n", "\n  ") + "...")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AVERS Demo - Automated Vectorization and Recognition of Schematics")
    print("=" * 60)

    try:
        import cv2
    except ImportError:
        print("Note: cv2 not available, using fallback for image generation")
        print("Full demo requires: pip install opencv-python-headless")
        # Continue with basic demos

    try:
        demo_slicing()
    except Exception as e:
        print(f"  Skipping slicing demo: {e}")

    demo_graph_synthesis()
    demo_output_format()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Add real schematic images to test")
    print("  2. Configure ML models for stages 2, 3, 6")
    print("  3. Run: python -m avers.main <image.tif> --output result.json")
    print()


if __name__ == "__main__":
    main()
