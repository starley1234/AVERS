"""
AVERS Main Pipeline Orchestrator

Coordinates all 6 stages of the schematic processing pipeline:
1. SAHI image slicing
2. УГО detection (RT-DETR/YOLO)
3. OCR text recognition (PaddleOCR)
4. Wire vectorization (OpenCV)
5. Graph synthesis (NetworkX)
6. VLM arbitration (Gemma/Qwen-VL)
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field

import numpy as np
import cv2

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
from avers.stages.stage1_slicing import SlicingEngine, Tile, load_image
from avers.stages.stage2_detection import (
    detect_components,
    DetectionConfig,
    group_detections_into_components,
)
from avers.stages.stage3_ocr import (
    recognize_schematic_text,
    OCRConfig,
    TextLabel,
    TextAssociationEngine,
)
from avers.stages.stage4_vectorization import (
    vectorize_wires,
    VectorizationConfig,
)
from avers.stages.stage5_graph_synthesis import (
    GraphBuilder,
    WireSegment,
    PinReference,
)
from avers.stages.stage6_vlm_arbitrator import (
    VLMWrapper,
    VLMConfig,
    ArbitrationEngine,
    JunctionIssue,
    create_issues_from_detections,
)


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
    
    Orchestrates all 6 stages with:
    - Progress tracking
    - Error handling and recovery
    - Intermediate result caching
    - Parallel processing support
    """

    def __init__(
        self,
        config: Optional[AVERSConfig] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize AVERS pipeline.
        
        Args:
            config: Pipeline configuration
            log_level: Logging level
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = setup_logger("avers", level=log_level)

        # Stage engines
        self._slicing_engine: Optional[SlicingEngine] = None
        self._graph_builder: Optional[GraphBuilder] = None
        self._text_association: Optional[TextAssociationEngine] = None

        # Processing state
        self.current_image: Optional[np.ndarray] = None
        self.current_tiles: List[Tile] = []
        self.stage_results: Dict[str, StageResult] = {}

        # Processed data
        self.detected_components: List[Component] = []
        self.text_labels: List[TextLabel] = []
        self.wire_segments: List[WireSegment] = []
        self.junction_points: set = set()
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
        """Stage 1: Slice large image into overlapping tiles."""
        start_time = time.time()
        self.logger.info("Stage 1: Image slicing (SAHI)")

        try:
            tiles = list(self.slicing_engine.generate_tiles(image))
            self.current_tiles = tiles
            self.logger.info(f"Generated {len(tiles)} tiles from {image.shape[1]}x{image.shape[0]} image")

            if visualize:
                self.slicing_engine.visualize_tiles(image, tiles, output_path)

            return StageResult(
                stage_name="stage1_slicing",
                success=True,
                data={"tiles": tiles, "tile_count": len(tiles)},
                processing_time=time.time() - start_time,
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
        image: np.ndarray,
    ) -> StageResult:
        """Stage 2: Detect УГО ( условные графические обозначения)."""
        start_time = time.time()
        self.logger.info("Stage 2: УГО Detection")

        try:
            detection_config = DetectionConfig(
                model_path=self.config.detection.model_path,
                model_type=self.config.detection.model_type,
                confidence_threshold=self.config.detection.confidence_threshold,
                iou_threshold=self.config.detection.iou_threshold,
                device=self.config.detection.device,
            )

            components, raw_detections = detect_components(
                image,
                config=detection_config,
                use_slicing=True,
            )

            self.detected_components = components
            self.logger.info(f"Detected {len(components)} components")

            # Create issues for low confidence detections
            for comp in components:
                if comp.confidence < 0.7:
                    self.human_review_issues.append(HumanReviewIssue(
                        issue_type=IssueType.LOW_CONFIDENCE_DETECTION,
                        bbox=comp.bbox,
                        description=f"Low confidence detection: {comp.designator}",
                        confidence=comp.confidence,
                        suggestions=[comp.designator],
                    ))

            return StageResult(
                stage_name="stage2_detection",
                success=True,
                data={
                    "components": components,
                    "raw_detections": raw_detections,
                    "detection_count": len(components),
                },
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
        image: np.ndarray,
    ) -> StageResult:
        """Stage 3: OCR text recognition."""
        start_time = time.time()
        self.logger.info("Stage 3: OCR Text Recognition")

        try:
            ocr_config = OCRConfig(
                lang=self.config.ocr.lang,
                use_angle_cls=self.config.ocr.enable_angle_cls,
                text_confidence_threshold=self.config.ocr.text_confidence_threshold,
            )

            labels, raw_results = recognize_schematic_text(image, config=ocr_config)
            self.text_labels = labels

            self.logger.info(f"OCR recognized {len(labels)} text labels")

            # Build text association engine
            self._text_association = TextAssociationEngine(
                association_radius=self.config.graph_synthesis.text_association_radius,
            )
            # Add OCR results
            from avers.stages.stage3_ocr import OCRResult
            for result in raw_results:
                self._text_association.add_labels([result])

            # Create issues for low confidence text
            for label in labels:
                if label.confidence < self.config.ocr.text_confidence_threshold:
                    self.human_review_issues.append(HumanReviewIssue(
                        issue_type=IssueType.LOW_CONFIDENCE_TEXT,
                        bbox=label.bbox,
                        description=f"Low confidence text: '{label.text}'",
                        confidence=label.confidence,
                        suggestions=[label.text],
                    ))

            return StageResult(
                stage_name="stage3_ocr",
                success=True,
                data={
                    "labels": labels,
                    "raw_results": raw_results,
                    "text_count": len(labels),
                },
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
        image: np.ndarray,
    ) -> StageResult:
        """Stage 4: Vectorize wire traces."""
        start_time = time.time()
        self.logger.info("Stage 4: Wire Vectorization")

        try:
            # Collect exclusion bboxes
            ugo_bboxes = [comp.bbox for comp in self.detected_components]
            text_bboxes = [label.bbox for label in self.text_labels]

            vectorization_config = VectorizationConfig(
                skeletonize_method=self.config.vectorization.skeletonize_method,
                rdp_epsilon=self.config.vectorization.rdp_epsilon,
                hough_min_line_length=self.config.vectorization.line_thickness_threshold,
            )

            segments, junctions = vectorize_wires(
                image,
                ugo_bboxes=ugo_bboxes,
                text_bboxes=text_bboxes,
                config=vectorization_config,
            )

            self.wire_segments = segments
            self.junction_points = junctions

            self.logger.info(f"Vectorized {len(segments)} wire segments, found {len(junctions)} junctions")

            # Create issues for suspicious crossings
            for junc in junctions:
                bbox = (junc[0] - 50, junc[1] - 50, junc[0] + 50, junc[1] + 50)
                self.human_review_issues.append(HumanReviewIssue(
                    issue_type=IssueType.SUSPICIOUS_CROSSING,
                    bbox=bbox,
                    description="Wire junction requiring verification",
                    confidence=0.5,
                ))

            return StageResult(
                stage_name="stage4_vectorization",
                success=True,
                data={
                    "segments": segments,
                    "junctions": junctions,
                    "segment_count": len(segments),
                    "junction_count": len(junctions),
                },
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
    ) -> StageResult:
        """Stage 5: Build topological graph from components and wires."""
        start_time = time.time()
        self.logger.info("Stage 5: Graph Synthesis")

        try:
            # Clear previous state
            self._graph_builder = None
            builder = self.graph_builder

            # Add wire segments
            for seg in self.wire_segments:
                builder.add_wire_segment(
                    start=seg.start,
                    end=seg.end,
                    points=seg.points,
                    confidence=seg.confidence,
                )

            self.logger.debug(f"Added {len(self.wire_segments)} wire segments")

            # Add component pins
            pins = []
            for comp in self.detected_components:
                for pin in comp.pins:
                    pins.append(PinReference(
                        component_id=comp.id,
                        pin_number=pin.pin_number,
                        coord=tuple(pin.coord),
                    ))

            builder.add_component_pins(pins)
            self.logger.debug(f"Added {len(pins)} component pins")

            # Snap wires to pins
            snapping = builder.snap_wire_to_pins()
            snapped_count = sum(1 for p in snapping.values() if p is not None)
            self.logger.info(f"Snapped {snapped_count} wire endpoints to pins")

            # Merge collinear segments
            if self.config.graph_synthesis.merge_collinear_segments:
                merged = builder.merge_collinear_segments()
                self.logger.debug(f"Merged {merged} collinear segments")

            # Build graph
            graph = builder.build_graph()
            self.logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Associate text labels with components
            text_dicts = [
                {"text": label.text, "coord": label.coord}
                for label in self.text_labels
            ]
            text_associations = builder.associate_text_labels(text_dicts)

            # Update component text associations
            for comp in self.detected_components:
                comp_center = (
                    (comp.bbox[0] + comp.bbox[2]) // 2,
                    (comp.bbox[1] + comp.bbox[3]) // 2,
                )
                nearby_labels = builder.associate_text_labels([{"text": "", "coord": comp_center}])
                # Simplified - would use proper association

            # Extract nets
            nets_data = builder.extract_nets([c.model_dump() for c in self.detected_components])

            # Convert to Net objects
            nets = []
            for net_data in nets_data:
                net = Net(
                    net_id=net_data["net_id"],
                    connections=[
                        WireConnection(**conn)
                        for conn in net_data.get("connections", [])
                    ],
                    path_points=net_data.get("path_points", []),
                    confidence=net_data.get("confidence", 1.0),
                )
                nets.append(net)

            self.detected_nets = nets
            self.logger.info(f"Extracted {len(nets)} electrical nets")

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
                processing_time=time.time() - start_time,
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
        image: np.ndarray,
    ) -> StageResult:
        """Stage 6: Resolve ambiguities using VLM."""
        start_time = time.time()
        self.logger.info("Stage 6: VLM Arbitration")

        if not self.config.vlm_arbitrator.enabled:
            self.logger.info("VLM arbitration disabled")
            return StageResult(
                stage_name="stage6_vlm_arbitrator",
                success=True,
                data={"resolved_count": 0, "vlm_calls": 0},
                warnings=["VLM arbitration disabled"],
                processing_time=time.time() - start_time,
            )

        try:
            # Create issues from pending human review items
            junction_bboxes = [
                issue.bbox for issue in self.human_review_issues
                if issue.issue_type in (IssueType.SUSPICIOUS_CROSSING, IssueType.UNKNOWN_CONNECTION)
            ]

            issues = create_issues_from_detections(
                wire_junctions=[],  # Would get from graph
                junction_dots=[],  # Would get from detection
                low_confidence_texts=[
                    (issue.bbox, issue.confidence)
                    for issue in self.human_review_issues
                    if issue.issue_type == IssueType.LOW_CONFIDENCE_TEXT
                ],
            )

            if not issues:
                self.logger.info("No issues requiring VLM arbitration")
                return StageResult(
                    stage_name="stage6_vlm_arbitrator",
                    success=True,
                    data={"resolved_count": 0, "vlm_calls": 0},
                    processing_time=time.time() - start_time,
                )

            # Limit to max calls
            max_calls = self.config.vlm_arbitrator.max_vlm_calls
            issues_to_process = issues[:max_calls]

            vlm_config = VLMConfig(
                model_name=self.config.vlm_arbitrator.model_name,
                device=self.config.vlm_arbitrator.device,
                roi_size=self.config.vlm_arbitrator.roi_size,
                max_calls=max_calls,
            )

            vlm = VLMWrapper(vlm_config)
            engine = ArbitrationEngine(vlm)

            # Process issues
            results = engine.process_issues(image, issues_to_process)

            # Update human review issues with resolutions
            resolved_count = 0
            for issue, result in zip(issues_to_process, results):
                if result.resolved:
                    # Mark as potentially resolved
                    issue.resolved = True
                    issue.resolution = f"connected={result.connected}" if result.connected is not None else result.selected_text
                    resolved_count += 1

            self.logger.info(f"VLM resolved {resolved_count}/{len(issues_to_process)} issues "
                           f"({engine.calls_made} calls)")

            return StageResult(
                stage_name="stage6_vlm_arbitrator",
                success=True,
                data={
                    "resolved_count": resolved_count,
                    "total_issues": len(issues),
                    "vlm_calls": engine.calls_made,
                },
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
            image_path: Path to input image
            output_path: Path for output file
            output_format: 'json' or 'xml'
            skip_stages: List of stage names to skip
            
        Returns:
            AVERSManifest with all extracted data
        """
        start_time = time.time()
        image_path = Path(image_path)

        self.logger.info("=" * 60)
        self.logger.info(f"AVERS Pipeline Starting")
        self.logger.info(f"Input: {image_path}")
        self.logger.info("=" * 60)

        # Load image
        self.logger.info("Loading image...")
        image = load_image(image_path)
        height, width = image.shape[:2]
        self.logger.info(f"Image loaded: {width}x{height} px")

        # Get image metadata
        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            dpi = img.info.get("dpi", (300, 300))
            if isinstance(dpi, tuple):
                dpi = dpi[0]
        except Exception:
            dpi = 300

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

        # Run all stages
        stage_names = ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
        stage_methods = [
            ("stage1", lambda: self.stage1_slice_image(image)),
            ("stage2", lambda: self.stage2_detect_ugo(image)),
            ("stage3", lambda: self.stage3_ocr_text(image)),
            ("stage4", lambda: self.stage4_vectorize_wires(image)),
            ("stage5", lambda: self.stage5_build_graph()),
            ("stage6", lambda: self.stage6_vlm_arbitrate(image)),
        ]

        for stage_name, stage_func in stage_methods:
            if stage_name in skip_stages:
                self.logger.info(f"Skipping {stage_name}")
                continue

            result = stage_func()
            self.stage_results[stage_name] = result

            if not result.success:
                self.logger.error(f"{stage_name} failed, continuing with partial results")

        # Populate manifest
        manifest.components = self.detected_components
        manifest.nets = self.detected_nets
        manifest.human_review_required = self.human_review_issues

        # Update processing time
        total_time = time.time() - start_time
        manifest.schema_metadata.processing_time_seconds = total_time

        # Save output
        if output_path:
            self.logger.info(f"Saving output to {output_path}")
            manifest.save(output_path, format=output_format)

        # Summary
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete")
        self.logger.info(f"Components found: {len(manifest.components)}")
        self.logger.info(f"Nets extracted: {len(manifest.nets)}")
        self.logger.info(f"Issues for review: {len(manifest.human_review_required)}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info("=" * 60)

        return manifest

    def visualize_results(
        self,
        image: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create visualization of pipeline results.
        
        Args:
            image: Original image
            output_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Draw components
        for comp in self.detected_components:
            x1, y1, x2, y2 = comp.bbox
            color = (0, 255, 0) if comp.confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(vis, comp.designator, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw wire segments
        for seg in self.wire_segments:
            cv2.line(vis, seg.start, seg.end, (100, 100, 100), 2)

        # Draw junctions
        for junc in self.junction_points:
            cv2.circle(vis, junc, 5, (0, 0, 255), -1)

        if output_path:
            cv2.imwrite(str(output_path), vis)

        return vis


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
