"""Configuration management for AVERS."""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml


class SlicingConfig(BaseModel):
    """Stage 1: SAHI-based image slicing configuration."""
    tile_size: int = Field(default=1024, description="Tile size in pixels")
    overlap_ratio: float = Field(default=0.2, ge=0.0, le=0.5, description="Overlap ratio between tiles")
    target_stride: Optional[int] = Field(default=None, description="Override stride calculation")
    min_tile_area: int = Field(default=100, description="Minimum tile area to process")


class DetectionConfig(BaseModel):
    """Stage 2: Object detection (RT-DETR/YOLO) configuration."""
    model_path: Optional[str] = Field(default=None, description="Path to model weights")
    model_type: str = Field(default="yolo", description="Model type: 'yolo' or 'rtdetr'")
    model_name: str = Field(default="yolov11x", description="Model variant")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="NMS IOU threshold")
    device: str = Field(default="cuda", description="Device: 'cuda', 'cpu', 'mps'")
    batch_size: int = Field(default=4, description="Batch size for inference")
    img_size: int = Field(default=640, description="Model input size")


class OCRConfig(BaseModel):
    """Stage 3: OCR configuration (PaddleOCR)."""
    enabled: bool = Field(default=True)
    lang: str = Field(default="ru", description="Language: 'ru', 'en', 'ch'")
    text_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    enable_angle_cls: bool = Field(default=True, description="Enable text direction classification")
    allowed_rotations: List[int] = Field(
        default=[0, 90, 180, 270],
        description="Allowed text rotations"
    )
    # Regex patterns for validation
    connector_pattern: str = Field(
        default=r"^[ХX]\d+$",
        description="Regex for connector designators"
    )
    pin_pattern: str = Field(
        default=r"^\d+$",
        description="Regex for pin numbers"
    )
    wire_pattern: str = Field(
        default=r"^БПВЛ(-\d+(\.\d+)?)?$",
        description="Regex for wire type labels"
    )


class VectorizationConfig(BaseModel):
    """Stage 4: Wire vectorization configuration."""
    enabled: bool = Field(default=True)
    line_thickness_threshold: int = Field(default=3, description="Min line thickness in pixels")
    skeletonize_method: str = Field(
        default="guo_hall",
        description="Thinning algorithm: 'guo_hall' or 'zhang_suen'"
    )
    rdp_epsilon: float = Field(default=2.0, description="Ramer-Douglas-Peucker epsilon")
    junction_degree_threshold: int = Field(
        default=3,
        description="Node degree to consider as junction"
    )
    snap_radius: int = Field(
        default=15,
        ge=0,
        description="Radius for snapping wires to pins"
    )


class GraphSynthesisConfig(BaseModel):
    """Stage 5: Graph synthesis configuration."""
    enabled: bool = Field(default=True)
    snap_enabled: bool = Field(default=True, description="Enable wire-to-pin snapping")
    snap_radius: int = Field(default=15, ge=0, description="Snap radius in pixels")
    text_association_radius: int = Field(
        default=50,
        description="Max distance for text-to-component association"
    )
    merge_collinear_segments: bool = Field(
        default=True,
        description="Merge collinear wire segments"
    )
    merge_tolerance: float = Field(
        default=5.0,
        description="Tolerance for collinear merge (pixels)"
    )


class VLMArbitratorConfig(BaseModel):
    """Stage 6: VLM-based arbitration configuration."""
    enabled: bool = Field(default=True)
    model_name: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="VLM model name from HuggingFace"
    )
    device: str = Field(default="cuda", description="Device: 'cuda' or 'cpu'")
    roi_size: int = Field(default=256, description="ROI crop size in pixels")
    confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Min confidence to trigger VLM query"
    )
    max_vlm_calls: int = Field(
        default=50,
        description="Maximum VLM calls per image (cost control)"
    )


class OutputConfig(BaseModel):
    """Output format configuration."""
    default_format: str = Field(default="json", description="Default output format")
    json_indent: int = Field(default=2, description="JSON indentation")
    include_intermediate_stages: bool = Field(
        default=False,
        description="Include intermediate stage results"
    )
    output_dir: Optional[str] = Field(default=None, description="Default output directory")


class AVERSConfig(BaseModel):
    """Complete AVERS configuration."""
    project_name: str = Field(default="AVERS")
    version: str = Field(default="0.1.0")

    # Stage configurations
    slicing: SlicingConfig = Field(default_factory=SlicingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    vectorization: VectorizationConfig = Field(default_factory=VectorizationConfig)
    graph_synthesis: GraphSynthesisConfig = Field(default_factory=GraphSynthesisConfig)
    vlm_arbitrator: VLMArbitratorConfig = Field(default_factory=VLMArbitratorConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AVERSConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)


def get_default_config() -> AVERSConfig:
    """Get default AVERS configuration."""
    return AVERSConfig()


# Default configuration instance
DEFAULT_CONFIG = get_default_config()
