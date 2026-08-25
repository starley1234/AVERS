#!/usr/bin/env python3
"""
AVERS Production Pipeline Demo

Demonstrates the production-ready pipeline with:
- SAHI integration (if available)
- Error handling
- Validation
- Stage timings
"""

import numpy as np
import cv2

from avers.pipeline import ProductionPipeline, load_and_process
from avers.config import AVERSConfig


def create_test_schematic(width=2000, height=1000):
    """Create a test schematic image."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw connector X1
    cv2.rectangle(image, (100, 200), (180, 400), (100, 100, 100), 2)
    cv2.putText(image, "X1", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(image, (180, 250), 5, (0, 0, 0), -1)
    cv2.circle(image, (180, 300), 5, (0, 0, 0), -1)
    cv2.circle(image, (180, 350), 5, (0, 0, 0), -1)
    
    # Draw connector X2
    cv2.rectangle(image, (900, 150), (980, 350), (100, 100, 100), 2)
    cv2.putText(image, "X2", (910, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.circle(image, (900, 200), 5, (0, 0, 0), -1)
    cv2.circle(image, (900, 250), 5, (0, 0, 0), -1)
    
    # Draw wires
    cv2.line(image, (180, 250), (500, 250), (0, 0, 0), 3)
    cv2.line(image, (500, 250), (500, 450), (0, 0, 0), 3)
    cv2.line(image, (500, 450), (900, 200), (0, 0, 0), 3)
    
    cv2.line(image, (180, 300), (400, 300), (0, 0, 0), 3)
    cv2.line(image, (400, 300), (400, 500), (0, 0, 0), 3)
    cv2.line(image, (400, 500), (900, 250), (0, 0, 0), 3)
    
    # Junction dot
    cv2.circle(image, (400, 300), 5, (0, 0, 0), -1)
    
    # Text labels
    cv2.putText(image, "+27В", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image, "1", (190, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "2", (190, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add noise
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image


def main():
    print("=" * 60)
    print("AVERS Production Pipeline Demo")
    print("=" * 60)
    
    # Create test image
    print("\nCreating test schematic...")
    image = create_test_schematic()
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Configure pipeline
    config = AVERSConfig()
    config.vlm_arbitrator.enabled = False  # Disable VLM for demo
    
    # Run production pipeline
    print("\nRunning production pipeline...")
    pipeline = ProductionPipeline(config)
    
    result = pipeline.run(image, "demo_board.tif", dpi=300)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print(f"\nSuccess: {result.success}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print(f"\nStage Timings:")
    for stage, time_sec in result.stage_timings.items():
        print(f"  {stage}: {time_sec*1000:.1f}ms")
    
    print(f"\nManifest:")
    print(f"  Source: {result.manifest.schema_metadata.source_file}")
    print(f"  DPI: {result.manifest.schema_metadata.resolution_dpi}")
    print(f"  Size: {result.manifest.schema_metadata.width}x{result.manifest.schema_metadata.height}")
    print(f"  Components: {len(result.manifest.components)}")
    print(f"  Nets: {len(result.manifest.nets)}")
    print(f"  Issues: {len(result.manifest.human_review_required)}")
    
    print(f"\nTotal processing time: {result.manifest.schema_metadata.processing_time_seconds:.2f}s")
    
    # Print components
    if result.manifest.components:
        print(f"\nDetected Components:")
        for comp in result.manifest.components:
            pins_str = f"({len(comp.pins)} pins)" if comp.pins else ""
            print(f"  {comp.designator}: {comp.type.value} {pins_str}")
    
    # Print nets
    if result.manifest.nets:
        print(f"\nExtracted Nets (first 5):")
        for net in result.manifest.nets[:5]:
            print(f"  {net.net_id}: {len(net.connections)} connections")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
