"""Type definitions for AVERS data structures (JSON Schema as Pydantic models)."""

from pathlib import Path
from typing import List, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ComponentType(str, Enum):
    """Types of schematic components."""
    CONNECTOR = "connector"
    DIODE = "diode"
    RESISTOR = "resistor"
    RELAY = "relay"
    GROUND = "ground"
    SHIELD = "shield"
    OFFPAGE_CONNECTOR = "offpage_connector"
    JUNCTION_DOT = "junction_dot"
    UNKNOWN = "unknown"


class WireColor(str, Enum):
    """Wire color codes (ГОСТ 2.304)."""
    K = "К"  # Красный
    B = "Б"  # Белый
    G = "З"  # Зелёный
    J = "Ж"  # Жёлтый
    S = "С"  # Серый
    P = "П"  # Пурпурный
    H = "Г"  # Голубой
    O = "О"  # Оранжевый
    CH = "Ч" # Чёрный
    KGN = "КГ"  # Камуфляж зелёный
    NONE = ""


class IssueType(str, Enum):
    """Types of issues requiring human review."""
    LOW_CONFIDENCE_TEXT = "low_confidence_text"
    LOW_CONFIDENCE_DETECTION = "low_confidence_detection"
    SUSPICIOUS_CROSSING = "suspicious_crossing"
    UNKNOWN_CONNECTION = "unknown_connection"
    SUSPECTED_BREAK = "suspected_break"


class BoundingBox(BaseModel):
    """Bounding box in image coordinates."""
    x_min: int = Field(ge=0, description="Left X coordinate")
    y_min: int = Field(ge=0, description="Top Y coordinate")
    x_max: int = Field(ge=0, description="Right X coordinate")
    y_max: int = Field(ge=0, description="Bottom Y coordinate")

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of the bounding box."""
        return (
            (self.x_min + self.x_max) // 2,
            (self.y_min + self.y_max) // 2,
        )

    @property
    def width(self) -> int:
        """Get width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Get height of the bounding box."""
        return self.y_max - self.y_min

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Get as (x_min, y_min, x_max, y_max) tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Get as (x_min, y_min, width, height) tuple for detection models."""
        return (self.x_min, self.y_min, self.width, self.height)


class Pin(BaseModel):
    """Pin/contact of a component."""
    pin_number: str = Field(description="Pin identifier (e.g., '1', '2', 'A')")
    coord: Tuple[int, int] = Field(description="Pin coordinate (x, y)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")

    @field_validator("coord")
    @classmethod
    def validate_coord(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v[0] < 0 or v[1] < 0:
            raise ValueError(f"Coordinates must be non-negative, got {v}")
        return v


class Component(BaseModel):
    """
    Schematic component (УГО - Условное Графическое Обозначение).
    """
    id: str = Field(description="Unique component identifier")
    designator: str = Field(description="Reference designator (e.g., 'X1', 'VD1', 'R2')")
    type: ComponentType = Field(description="Component type")
    part_number: Optional[str] = Field(default=None, description="Part number (e.g., 'СНЦ144-6/10РО11')")
    bbox: Tuple[int, int, int, int] = Field(
        description="Bounding box (x_min, y_min, x_max, y_max)"
    )
    pins: List[Pin] = Field(default_factory=list, description="List of component pins")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    text_associations: dict[str, str] = Field(
        default_factory=dict,
        description="Text labels associated with this component"
    )

    @property
    def bbox_model(self) -> BoundingBox:
        """Get bounding box as a model."""
        return BoundingBox(
            x_min=self.bbox[0],
            y_min=self.bbox[1],
            x_max=self.bbox[2],
            y_max=self.bbox[3],
        )


class WireConnection(BaseModel):
    """Connection point in a net."""
    component_id: str = Field(description="Component ID")
    pin: str = Field(description="Pin identifier")


class Net(BaseModel):
    """
    Electrical net (цепь/провод) in the schematic.
    """
    net_id: str = Field(description="Unique net identifier")
    net_name: Optional[str] = Field(default=None, description="Net name (e.g., '+27V', 'GND')")
    wire_type: Optional[str] = Field(default=None, description="Wire type (e.g., 'БПВЛ-0.35')")
    wire_color: Optional[str] = Field(default=None, description="Wire color code")
    connections: List[WireConnection] = Field(
        default_factory=list,
        description="List of component connections"
    )
    path_points: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Wire path as list of (x, y) points"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Net extraction confidence")
    intermediate_components: List[str] = Field(
        default_factory=list,
        description="Component IDs that this wire passes through"
    )

    def add_connection(self, component_id: str, pin: str) -> None:
        """Add a connection point to this net."""
        self.connections.append(WireConnection(component_id=component_id, pin=pin))


class SchemaMetadata(BaseModel):
    """Metadata about the source schematic."""
    source_file: str = Field(description="Original file name")
    resolution_dpi: int = Field(ge=0, description="Image resolution in DPI")
    width: int = Field(ge=0, description="Image width in pixels")
    height: int = Field(ge=0, description="Image height in pixels")
    format: Optional[str] = Field(default=None, description="File format (TIF, PNG, PDF)")
    processing_time_seconds: Optional[float] = Field(default=None, description="Total processing time")


class Dimensions(BaseModel):
    """Image dimensions."""
    width: int
    height: int


class HumanReviewIssue(BaseModel):
    """Issue requiring human review."""
    issue_type: IssueType = Field(description="Type of issue")
    bbox: Tuple[int, int, int, int] = Field(
        description="Region of interest (x_min, y_min, x_max, y_max)"
    )
    description: str = Field(description="Human-readable description of the issue")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence")
    suggestions: List[str] = Field(
        default_factory=list,
        description="Possible interpretations"
    )
    resolved: bool = Field(default=False, description="Whether the issue has been resolved")
    resolution: Optional[str] = Field(default=None, description="Resolution chosen by human")


class RecognizedText(BaseModel):
    """Распознанный OCR фрагмент текста с координатами на листе."""
    bbox: List[int] = Field(description="[x1, y1, x2, y2] в пикселях исходного изображения")
    text: str = Field(description="Распознанный текст")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AVERSManifest(BaseModel):
    """
    Complete output manifest from AVERS pipeline.
    Corresponds to the JSON Schema specified in the technical requirements.
    """
    schema_metadata: SchemaMetadata = Field(description="Source schematic metadata")
    texts: List[RecognizedText] = Field(
        default_factory=list,
        description="Распознанные OCR-тексты (для UI и экспорта)"
    )
    components: List[Component] = Field(
        default_factory=list,
        description="Detected components (УГО)"
    )
    nets: List[Net] = Field(
        default_factory=list,
        description="Extracted electrical nets"
    )
    human_review_required: List[HumanReviewIssue] = Field(
        default_factory=list,
        description="Issues requiring human verification"
    )

    model_config = {"use_enum_values": True}

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(**kwargs)

    def to_dict(self, **kwargs) -> dict:
        """Serialize to dictionary."""
        return self.model_dump(**kwargs)

    def save(self, path: str, format: str = "json") -> None:
        """
        Save manifest to file.

        Args:
            path: Output file path
            format: Output format ('json' or 'xml')
        """
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            path.write_text(self.to_json(indent=2), encoding="utf-8")
        elif format == "xml":
            self._to_xml(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _to_xml(self, path: Path) -> None:
        """Export to XML format for САПР compatibility."""
        from lxml import etree

        root = etree.Element("avers_manifest")

        # Metadata
        meta = etree.SubElement(root, "schema_metadata")
        for key, value in self.schema_metadata.model_dump().items():
            etree.SubElement(meta, key).text = str(value)

        # Components
        components_elem = etree.SubElement(root, "components")
        for comp in self.components:
            comp_elem = etree.SubElement(components_elem, "component")
            for key, value in comp.model_dump().items():
                if key == "pins":
                    for pin in value:
                        pin_elem = etree.SubElement(comp_elem, "pin")
                        for pin_key, pin_val in pin.items():
                            etree.SubElement(pin_elem, pin_key).text = str(pin_val)
                elif key == "bbox":
                    etree.SubElement(comp_elem, key).text = str(list(value))
                elif key == "text_associations":
                    for label_key, label_val in value.items():
                        etree.SubElement(comp_elem, f"label_{label_key}").text = str(label_val)
                else:
                    etree.SubElement(comp_elem, key).text = str(value)

        # Nets
        nets_elem = etree.SubElement(root, "nets")
        for net in self.nets:
            net_elem = etree.SubElement(nets_elem, "net")
            for key, value in net.model_dump().items():
                if key in ("connections", "intermediate_components"):
                    for item in value:
                        etree.SubElement(net_elem, key[:-1]).text = str(item)
                elif key == "path_points":
                    etree.SubElement(net_elem, key).text = str(value)
                else:
                    etree.SubElement(net_elem, key).text = str(value)

        # Human review issues
        issues_elem = etree.SubElement(root, "human_review_required")
        for issue in self.human_review_required:
            issue_elem = etree.SubElement(issues_elem, "issue")
            for key, value in issue.model_dump().items():
                etree.SubElement(issue_elem, key).text = str(value)

        tree = etree.ElementTree(root)
        tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)


# Detection class names (ГОСТ УГО)
DETECTION_CLASSES = {
    0: "connector_body",
    1: "pin",
    2: "junction_dot",
    3: "ground",
    4: "shield",
    5: "offpage_connector",
    6: "diode",
    7: "relay",
    8: "resistor",
}

CLASS_TO_COMPONENT_TYPE = {
    "connector_body": ComponentType.CONNECTOR,
    "pin": ComponentType.UNKNOWN,  # Pins are nested in components
    "junction_dot": ComponentType.JUNCTION_DOT,
    "ground": ComponentType.GROUND,
    "shield": ComponentType.SHIELD,
    "offpage_connector": ComponentType.OFFPAGE_CONNECTOR,
    "diode": ComponentType.DIODE,
    "relay": ComponentType.RELAY,
    "resistor": ComponentType.RESISTOR,
}
