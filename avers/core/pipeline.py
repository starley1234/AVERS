"""
AVERS Main Pipeline Orchestrator

Coordinates all stages of the schematic processing pipeline:
1. Image preprocessing and SAHI slicing
2. УГО detection (RT-DETR/YOLO)
3. OCR text recognition
4. Wire vectorization (OpenCV)
5. Graph synthesis (NetworkX)
6. VLM arbitration for conflicts
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np

from avers.core.logger import get_logger, setup_logger
from avers.core.types import (
    AVERSManifest,
    SchemaMetadata,
    Component,
    Net,
    WireConnection,
    HumanReviewIssue,
    IssueType,
)
from avers.config import AVERSConfig, DEFAULT_CONFIG
from avers.stages.stage1_slicing import SlicingEngine, Tile, DetectionBox, load_image
from avers.stages.stage5_graph_synthesis import GraphBuilder, WireSegment, PinReference


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage_name: str
    success: bool
    data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0


class aversPipeline:
    """
    Main pipeline for automated schematic vectorization.

    Orchestrates the multi-stage processing pipeline with:
    - Progress tracking
    - Error handling and recovery
    - Intermediate result caching
    - Parallel processing support (via sahi)
    """

    def __init__(
        self,
        config: Optional[AVERSConfig] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize AVERS pipeline.

        Args:
            config: Pipeline configuration (uses default if not provided)
            log_level: Logging level
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = setup_logger("avers", level=log_level)

        # Stage engines (lazy-loaded)
        self._slicing_engine: Optional[SlicingEngine] = None
        self._graph_builder: Optional[GraphBuilder] = None

        # Processing state
        self.current_image: Optional[np.ndarray] = None
        self.current_tiles: List[Tile] = []
        self.stage_results: Dict[str, StageResult] = {}

        # Processed data
        self.detected_components: List[Component] = []
        self.detected_nets: List[Net] = []
        self.human_review_issues: List[HumanReviewIssue] = []

    # ==================== Stage 1: Slicing ====================

    @property
    def slicing_engine(self) -> SlicingEngine:
        """Lazy-load slicing engine."""
        if self._slicing_engine is None:
            self._slicing_engine = SlicingEngine(
                tile_size=self.config.slicing.tile_size,
                overlap_ratio=self.config.slicing.overlap_ratio,
                target_stride=self.config.slicing.target_stride,
                min_tile_area=self.config.slicing.min_tile_area,
            )
        return self._slicing_engine

    def stage1_slice_image(
        self,
        image: np.ndarray,
        visualize: bool = False,
        output_path: Optional[Path] = None,
    ) -> StageResult:
        """
        Stage 1: Slice large image into overlapping tiles.

        Args:
            image: Input image
            visualize: Create tile visualization
            output_path: Path to save visualization

        Returns:
            StageResult with tiles
        """
        start_time = time.time()
        self.logger.info("Stage 1: Image slicing (SAHI)")

        try:
            # Generate tiles
            tiles = list(self.slicing_engine.generate_tiles(image))
            self.current_tiles = tiles

            self.logger.info(f"Generated {len(tiles)} tiles from image")

            # Optional visualization
            if visualize:
                self.slicing_engine.visualize_tiles(
                    image, tiles, output_path
                )
                self.logger.debug(f"Tile visualization saved to {output_path}")

            processing_time = time.time() - start_time

            return StageResult(
                stage_name="stage1_slicing",
                success=True,
                data={"tiles": tiles, "tile_count": len(tiles)},
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"Stage 1 failed: {e}")
            return StageResult(
                stage_name="stage1_slicing",
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    # ==================== Stage 2: Detection ====================

    def stage2_detect_ugo(
        self,
        image: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
    ) -> StageResult:
        """
        Stage 2: Detect УГО ( условные графические обозначения).

        Args:
            image: Full image (if not using tiles)
            model_path: Path to detection model weights

        Returns:
            StageResult with detected components
        """
        start_time = time.time()
        self.logger.info("Stage 2: УГО Detection (RT-DETR/YOLO)")

        try:
            # Placeholder for actual detection implementation
            # Would integrate with:
            # - sahi for tiled inference
            # - ultralytics for YOLO/RT-DETR
            # - Project coordinates back to global space

            # For now, return empty results with warning
            self.detected_components = []

            self.logger.warning("Stage 2: Detection model not yet loaded")

            return StageResult(
                stage_name="stage2_detection",
                success=True,
                data={"components": [], "detection_count": 0},
                warnings=["Detection model not implemented yet"],
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Stage 2 failed: {e}")
            return StageResult(
                stage_name="stage2_detection",
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    # ==================== Stage 3: OCR ====================

    def stage3_ocr_text(
        self,
        image: Optional[np.ndarray] = None,
    ) -> StageResult:
        """
        Stage 3: OCR text recognition and association.

        Args:
            image: Input image

        Returns:
            StageResult with recognized text
        """
        start_time = time.time()
        self.logger.info("Stage 3: OCR Text Recognition (PaddleOCR)")

        try:
            # Placeholder for OCR implementation
            # Would use:
            # - PaddleOCR with DBNet + CRNN
            # - Text direction classification (0°, 90°, 270°)
            # - Regex validation for ГОСТ designations

            text_labels = []

            self.logger.warning("Stage 3: OCR model not yet loaded")

            return StageResult(
                stage_name="stage3_ocr",
                success=True,
                data={"text_labels": text_labels, "text_count": 0},
                warnings=["OCR model not implemented yet"],
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Stage 3 failed: {e}")
            return StageResult(
                stage_name="stage3_ocr",
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    # ==================== Stage 4: Vectorization ====================

    def stage4_vectorize_wires(
        self,
        image: Optional[np.ndarray] = None,
    ) -> StageResult:
        """
        Stage 4: Vectorize wire traces using OpenCV.

        Args:
            image: Input image with wires

        Returns:
            StageResult with wire segments
        """
        start_time = time.time()
        self.logger.info("Stage 4: Wire Vectorization (OpenCV)")

        try:
            # Placeholder for vectorization
            # Would:
            # 1. Create exclusion mask from УГО bboxes
            # 2. Skeletonize lines (Guo-Hall or Zhang-Suen)
            # 3. Extract polylines
            # 4. RDP simplification

            wire_segments = []

            self.logger.warning("Stage 4: Vectorization not implemented yet")

            return StageResult(
                stage_name="stage4_vectorization",
                success=True,
                data={"wire_segments": wire_segments, "segment_count": 0},
                warnings=["Vectorization not implemented yet"],
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Stage 4 failed: {e}")
            return StageResult(
                stage_name="stage4_vectorization",
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    # ==================== Stage 5: Graph Synthesis ====================

    @property
    def graph_builder(self) -> GraphBuilder:
        """Lazy-load graph builder."""
        if self._graph_builder is None:
            self._graph_builder = GraphBuilder(
                snap_enabled=self.config.graph_synthesis.snap_enabled,
                snap_radius=self.config.graph_synthesis.snap_radius,
                text_association_radius=self.config.graph_synthesis.text_association_radius,
                merge_collinear=self.config.graph_synthesis.merge_collinear_segments,
                merge_tolerance=self.config.graph_synthesis.merge_tolerance,
            )
        return self._graph_builder

    def stage5_build_graph(
        self,
        wire_segments: Optional[List[WireSegment]] = None,
        components: Optional[List[Component]] = None,
        text_labels: Optional[List[Dict]] = None,
    ) -> StageResult:
        """
        Stage 5: Build topological graph from components and wires.

        Args:
            wire_segments: Vectorized wire segments
            components: Detected components
            text_labels: OCR text labels

        Returns:
            StageResult with graph and extracted nets
        """
        start_time = time.time()
        self.logger.info("Stage 5: Graph Synthesis (NetworkX)")

        try:
            # Add wire segments to graph builder
            if wire_segments:
                for seg in wire_segments:
                    self.graph_builder.add_wire_segment(
                        start=seg.start,
                        end=seg.end,
                        points=seg.points,
                        confidence=seg.confidence,
                    )

            # Add component pins
            if components:
                pins = []
                for comp in components:
                    for pin in comp.pins:
                        pins.append(PinReference(
                            component_id=comp.id,
                            pin_number=pin.pin_number,
                            coord=tuple(pin.coord),
                        ))

                self.graph_builder.add_component_pins(pins)

            # Snap wires to pins
            snapping = self.graph_builder.snap_wire_to_pins()
            snapped_count = sum(1 for p in snapping.values() if p is not None)
            self.logger.debug(f"Snapped {snapped_count} wires to pins")

            # Merge collinear segments
            merged = self.graph_builder.merge_collinear_segments()
            if merged > 0:
                self.logger.debug(f"Merged {merged} collinear segments")

            # Build graph
            graph = self.graph_builder.build_graph()
            self.logger.debug(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Extract nets
            nets_data = self.graph_builder.extract_nets(
                [c.model_dump() for c in components] if components else []
            )

            # Convert to Net objects
            nets = []
            for net_data in nets_data:
                net = Net(
                    net_id=net_data["net_id"],
                    connections=[
                        WireConnection(**conn)
                        for conn in net_data["connections"]
                    ],
                    path_points=net_data["path_points"],
                    confidence=net_data["confidence"],
                )
                nets.append(net)

            self.detected_nets = nets

            # Associate text labels
            text_associations = {}
            if text_labels:
                text_associations = self.graph_builder.associate_text_labels(text_labels)

            processing_time = time.time() - start_time

            return StageResult(
                stage_name="stage5_graph_synthesis",
                success=True,
                data={
                    "graph": graph,
                    "nets": nets,
                    "text_associations": text_associations,
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "net_count": len(nets),
                },
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"Stage 5 failed: {e}")
            return StageResult(
                stage_name="stage5_graph_synthesis",
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    # ==================== Stage 6: VLM Arbitration ====================

    def stage6_vlm_arbitrate(
        self,
        issues: Optional[List[HumanReviewIssue]] = None,
    ) -> StageResult:
        """
        Stage 6: Resolve ambiguities using Vision-Language Model.

        Args:
            issues: List of issues to resolve

        Returns:
            StageResult with resolved issues
        """
        start_time = time.time()
        self.logger.info("Stage 6: VLM Arbitration")

        if not self.config.vlm_arbitrator.enabled:
            self.logger.info("VLM arbitration disabled in config")
            return StageResult(
                stage_name="stage6_vlm_arbitrator",
                success=True,
                data={"resolved_issues": [], "vlm_calls": 0},
                warnings=["VLM arbitration disabled"],
                processing_time=time.time() - start_time,
            )

        try:
            # Placeholder for VLM integration
            # Would:
            # 1. Extract ROI crops (256x256)
            # 2. Query VLM with structured prompt
            # 3. Parse JSON response
            # 4. Update confidence scores

            self.logger.warning("Stage 6: VLM model not yet integrated")

            return StageResult(
                stage_name="stage6_vlm_arbitrator",
                success=True,
                data={"resolved_issues": [], "vlm_calls": 0},
                warnings=["VLM model not implemented yet"],
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Stage 6 failed: {e}")
            return StageResult(
                stage_name="stage6_vlm_arbitrator",
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    # ==================== Full Pipeline ====================

    def process(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "json",
        skip_stages: Optional[List[str]] = None,
    ) -> AVERSManifest:
        """
        Run full processing pipeline on a schematic image.

        Args:
            image_path: Path to input image (TIF, PNG, PDF)
            output_path: Path for output file (optional)
            output_format: Output format ('json' or 'xml')
            skip_stages: List of stage names to skip

        Returns:
            AVERSManifest with all extracted data
        """
        start_time = time.time()
        image_path = Path(image_path)

        self.logger.info(f"=" * 60)
        self.logger.info(f"AVERS Pipeline Starting")
        self.logger.info(f"Input: {image_path}")
        self.logger.info(f"=" * 60)

        # Load image
        self.logger.info("Loading image...")
        image = load_image(image_path)
        height, width = image.shape[:2]
        self.logger.info(f"Image loaded: {width}x{height} px")

        # Get image metadata
        import PIL.Image
        img = PIL.Image.open(image_path)
        dpi = img.info.get("dpi", (300, 300))
        if isinstance(dpi, tuple):
            dpi = dpi[0]

        # Initialize manifest
        manifest = AVERSManifest(
            schema_metadata=SchemaMetadata(
                source_file=image_path.name,
                resolution_dpi=int(dpi),
                width=width,
                height=height,
                format=image_path.suffix[1:].upper(),
            )
        )

        skip_stages = skip_stages or []
        self.stage_results = {}

        # Stage 1: Slicing
        if "stage1" not in skip_stages:
            result = self.stage1_slice_image(image)
            self.stage_results["stage1"] = result

        # Stage 2: Detection (placeholder)
        if "stage2" not in skip_stages:
            result = self.stage2_detect_ugo(image)
            self.stage_results["stage2"] = result
            manifest.components = self.detected_components

        # Stage 3: OCR (placeholder)
        if "stage3" not in skip_stages:
            result = self.stage3_ocr_text(image)
            self.stage_results["stage3"] = result
            text_labels = result.data.get("text_labels", []) if result.success else []

        # Stage 4: Vectorization (placeholder)
        if "stage4" not in skip_stages:
            result = self.stage4_vectorize_wires(image)
            self.stage_results["stage4"] = result
            wire_segments = result.data.get("wire_segments", []) if result.success else []

        # Stage 5: Graph synthesis
        if "stage5" not in skip_stages:
            result = self.stage5_build_graph(
                wire_segments=wire_segments if "stage4" in skip_stages else None,
                components=manifest.components,
                text_labels=text_labels if "stage3" in skip_stages else None,
            )
            self.stage_results["stage5"] = result
            manifest.nets = self.detected_nets

        # Stage 6: VLM arbitration
        if "stage6" not in skip_stages:
            result = self.stage6_vlm_arbitrate(manifest.human_review_required)
            self.stage_results["stage6"] = result

        # Update processing time
        manifest.schema_metadata.processing_time_seconds = time.time() - start_time

        # Save output
        if output_path:
            self.logger.info(f"Saving output to {output_path}")
            manifest.save(output_path, format=output_format)

        # Summary
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete")
        self.logger.info(f"Components found: {len(manifest.components)}")
        self.logger.info(f"Nets extracted: {len(manifest.nets)}")
        self.logger.info(f"Issues for review: {len(manifest.human_review_required)}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info("=" * 60)

        return manifest

    def process_tiles_parallel(
        self,
        image: np.ndarray,
        process_func,  # Function to apply to each tile
        num_workers: int = 4,
    ) -> List[Any]:
        """
        Process tiles in parallel.

        Args:
            image: Full image
            process_func: Function that takes a Tile and returns results
            num_workers: Number of parallel workers

        Returns:
            List of results from each tile
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tiles = list(self.slicing_engine.generate_tiles(image))
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_func, tile): tile
                for tile in tiles
            }

            for future in as_completed(futures):
                tile = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Tile {tile.tile_id} failed: {e}")
                    results.append(None)

        return results


# Convenience function
def process_schematic(
    image_path: str | Path,
    output_path: Optional[str | Path] = None,
    config: Optional[AVERSConfig] = None,
) -> AVERSManifest:
    """
    Process a schematic image with default settings.

    Args:
        image_path: Path to input image
        output_path: Path for output file
        config: Optional configuration

    Returns:
        AVERSManifest with results
    """
    pipeline = aversPipeline(config=config)
    return pipeline.process(image_path, output_path)
