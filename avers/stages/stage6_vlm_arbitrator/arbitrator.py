"""
Stage 6: VLM-based Arbitration for Uncertain Cases

Resolves ambiguous schematic elements using Vision-Language Models:
- Wire crossings: connection vs overlap
- Low confidence detections
- OCR disambiguation
- Missing junction dots

Only called for ROI patches (256x256) where confidence is low.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import cv2

from avers.core.logger import get_logger

logger = get_logger("avers.vlm")

# Prompts for different arbitration tasks
SYSTEM_PROMPT = """You are a CAD schematic QA inspector. Analyze the provided image of an electrical schematic junction and determine if there is an electrical connection.

Key rules:
- A junction dot (filled circle) at wire crossings indicates electrical connection
- Wires passing through each other without a dot are NOT connected
- Look carefully at the center of any wire intersection
- A small gap between wires means NO connection
- Merged/stacked wires with consistent line mean IS connected"""

JUNCTION_PROMPT = """Look at this wire junction. Is there an electrical connection dot at the center?
- Look for a filled circle at the intersection
- A visible dot means: connected=true
- No dot visible: connected=false

Respond in JSON format: {"connected": true/false, "confidence": 0.0-1.0}"""

CROSSING_PROMPT = """Is this wire crossing a connection point or just wires passing through each other?
- With junction dot: they ARE electrically connected
- Without junction dot: they are NOT connected

Respond in JSON format: {"connected": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""

TEXT_PROMPT = """Look at this schematic text label. What does it say? Consider:
- Common connector designations: X1, X2, Ш1, СНЦ144
- Pin numbers: 1, 2, 3, A, B
- Wire types: БПВЛ-0.35, МГТФ
- Voltage: +27В, -12В, GND

Respond in JSON format: {"text": "recognized text", "confidence": 0.0-1.0, "alternatives": ["option1", "option2"]}"""


@dataclass
class VLMConfig:
    """VLM arbitration configuration."""
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    roi_size: int = 256
    confidence_threshold: float = 0.65
    max_calls: int = 50
    temperature: float = 0.1
    max_tokens: int = 256


@dataclass
class ArbitrationResult:
    """Result from VLM arbitration."""
    resolved: bool
    connected: Optional[bool] = None
    selected_text: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    model: str = ""


@dataclass
class JunctionIssue:
    """Issue at wire junction requiring arbitration."""
    bbox: Tuple[int, int, int, int]
    issue_type: str  # 'suspicious_crossing', 'suspected_break', 'low_confidence'
    wire_count: int = 0
    has_dot: bool = False
    confidence: float = 0.0
    description: str = ""


class VLMWrapper:
    """
    Vision-Language Model wrapper for schematic arbitration.
    
    Supports:
    - Qwen2.5-VL
    - Gemma-4-VIT
    - Generic VLM via transformers
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load VLM model."""
        if self._loaded:
            return True
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            
            logger.info(f"Loading VLM: {self.config.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            # Load with quantization for memory efficiency
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
                device_map=self.config.device,
                trust_remote_code=True,
            )
            
            self._loaded = True
            logger.info("VLM loaded successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Transformers not installed: {e}")
            return self._load_mock()
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            return self._load_mock()
    
    def _load_mock(self) -> bool:
        """Load mock VLM for testing without actual model."""
        logger.warning("Using mock VLM - no actual inference")
        self._loaded = True
        self.model = None
        self.processor = None
        return True
    
    def query(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Query VLM with image and prompt.
        
        Args:
            image: Input image (H, W, 3)
            prompt: Text prompt
            
        Returns:
            Parsed response dict
        """
        if not self._loaded:
            self.load()
        
        if self.model is None:
            return self._mock_query(image, prompt)
        
        try:
            return self._real_query(image, prompt)
        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            return self._mock_query(image, prompt)
    
    def _real_query(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Run actual VLM inference."""
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        
        if self.config.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )
        
        # Decode
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        # Parse JSON response
        return self._parse_response(response)
    
    def _mock_query(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Mock VLM query for testing."""
        # Simple heuristic based on image analysis
        if "junction" in prompt.lower() or "crossing" in prompt.lower():
            # Check for junction dot using image analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Look for circular structure in center
            h, w = gray.shape
            center = gray[h//4:3*h//4, w//4:3*w//4]
            
            # Simple blob detection
            _, thresh = cv2.threshold(center, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            has_dot = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 50 < area < 500:  # Reasonable dot size
                    circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
                    if circularity > 0.5:  # Circular
                        has_dot = True
                        break
            
            return {
                "connected": has_dot,
                "confidence": 0.7 if has_dot else 0.6,
                "reasoning": "Mock VLM: detected junction dot" if has_dot else "Mock VLM: no dot found",
            }
        
        elif "text" in prompt.lower():
            # Mock OCR
            return {
                "text": "X1",
                "confidence": 0.5,
                "alternatives": ["X1", "X2", "Ш1"],
            }
        
        else:
            return {"connected": False, "confidence": 0.5}
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from VLM response."""
        import json
        import re
        
        # Try to extract JSON
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Try whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse VLM response: {response[:100]}...")
            return {"error": "parse_failed", "raw": response}


class ArbitrationEngine:
    """
    Orchestrates VLM arbitration for schematic ambiguities.
    """
    
    def __init__(self, vlm: Optional[VLMWrapper] = None):
        self.vlm = vlm or VLMWrapper()
        self.calls_made = 0
    
    def resolve_junction(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> ArbitrationResult:
        """
        Resolve whether a wire junction is connected.
        
        Args:
            image: Full image
            bbox: Junction bounding box
            
        Returns:
            ArbitrationResult
        """
        if self.calls_made >= self.vlm.config.max_calls:
            return ArbitrationResult(
                resolved=False,
                confidence=0.0,
                reasoning="Max VLM calls reached",
            )
        
        # Extract ROI
        roi = self._extract_roi(image, bbox)
        
        # Query VLM
        response = self.vlm.query(roi, JUNCTION_PROMPT)
        
        self.calls_made += 1
        
        return ArbitrationResult(
            resolved=True,
            connected=response.get("connected", False),
            confidence=response.get("confidence", 0.5),
            reasoning=response.get("reasoning", ""),
            model=self.vlm.config.model_name,
        )
    
    def resolve_crossing(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> ArbitrationResult:
        """Resolve whether wire crossing is connected or overlap."""
        if self.calls_made >= self.vlm.config.max_calls:
            return ArbitrationResult(
                resolved=False,
                confidence=0.0,
            )
        
        roi = self._extract_roi(image, bbox)
        response = self.vlm.query(roi, CROSSING_PROMPT)
        
        self.calls_made += 1
        
        return ArbitrationResult(
            resolved=True,
            connected=response.get("connected", False),
            confidence=response.get("confidence", 0.5),
            reasoning=response.get("reasoning", ""),
            model=self.vlm.config.model_name,
        )
    
    def resolve_text(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        hypotheses: List[str],
    ) -> ArbitrationResult:
        """Resolve ambiguous OCR text."""
        if self.calls_made >= self.vlm.config.max_calls:
            return ArbitrationResult(
                resolved=False,
                confidence=0.0,
            )
        
        roi = self._extract_roi(image, bbox)
        
        # Customize prompt with hypotheses
        prompt = TEXT_PROMPT + f"\nGiven options: {hypotheses}"
        response = self.vlm.query(roi, prompt)
        
        self.calls_made += 1
        
        return ArbitrationResult(
            resolved=True,
            selected_text=response.get("text", hypotheses[0] if hypotheses else ""),
            confidence=response.get("confidence", 0.5),
            model=self.vlm.config.model_name,
        )
    
    def _extract_roi(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        expand: float = 1.5,
    ) -> np.ndarray:
        """
        Extract and resize ROI from image.
        
        Args:
            image: Full image
            bbox: Bounding box
            expand: Expand ratio for context
            
        Returns:
            ROI image (roi_size x roi_size)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Expand bbox
        w = x_max - x_min
        h = y_max - y_min
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        
        new_w = int(w * expand)
        new_h = int(h * expand)
        
        x_min = max(0, cx - new_w // 2)
        y_min = max(0, cy - new_h // 2)
        x_max = min(image.shape[1], cx + new_w // 2)
        y_max = min(image.shape[0], cy + new_h // 2)
        
        # Crop
        crop = image[y_min:y_max, x_min:x_max]
        
        # Resize to standard size
        roi = cv2.resize(crop, (self.vlm.config.roi_size, self.vlm.config.roi_size))
        
        return roi
    
    def process_issues(
        self,
        image: np.ndarray,
        issues: List[JunctionIssue],
    ) -> List[ArbitrationResult]:
        """
        Process multiple arbitration issues.
        
        Args:
            image: Full image
            issues: List of issues to resolve
            
        Returns:
            List of ArbitrationResults
        """
        results = []
        
        for issue in issues:
            if issue.issue_type == "suspected_break":
                result = self.resolve_junction(image, issue.bbox)
            elif issue.issue_type == "suspicious_crossing":
                result = self.resolve_crossing(image, issue.bbox)
            elif issue.issue_type == "low_confidence_text":
                result = self.resolve_text(image, issue.bbox, issue.description.split())
            else:
                result = ArbitrationResult(resolved=False)
            
            results.append(result)
            
            if self.calls_made >= self.vlm.config.max_calls:
                logger.warning(f"Max VLM calls ({self.vlm.config.max_calls}) reached")
                break
        
        return results


def create_issues_from_detections(
    wire_junctions: List[Tuple[int, int]],
    junction_dots: List[Tuple[int, int, int, int]],
    low_confidence_texts: List[Tuple[int, int, int, int, float]],
    ambiguity_threshold: float = 0.65,
) -> List[JunctionIssue]:
    """
    Create arbitration issues from detection results.
    
    Args:
        wire_junctions: List of (x, y) junction coordinates
        junction_dots: List of (x_min, y_min, x_max, y_max) for detected dots
        low_confidence_texts: List of (bbox, confidence) for uncertain text
        
    Returns:
        List of JunctionIssues requiring arbitration
    """
    issues = []
    
    # Create junction dot lookup
    dot_set = set()
    for bbox in junction_dots:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        dot_set.add((cx, cy))
    
    # Check each wire junction
    for x, y in wire_junctions:
        # Check if there's a junction dot nearby
        has_dot = any(
            abs(cx - x) < 15 and abs(cy - y) < 15
            for cx, cy in dot_set
        )
        
        if not has_dot:
            # Suspicious - wires meet but no dot
            issues.append(JunctionIssue(
                bbox=(x - 50, y - 50, x + 50, y + 50),
                issue_type="suspicious_crossing",
                wire_count=3,  # Would be determined from graph
                has_dot=False,
                confidence=0.5,
                description="Wire junction without visible connection dot",
            ))
    
    # Add low confidence text issues
    for bbox, conf in low_confidence_texts:
        if conf < ambiguity_threshold:
            issues.append(JunctionIssue(
                bbox=bbox,
                issue_type="low_confidence_text",
                confidence=conf,
                description=f"Low confidence text (conf={conf:.2f})",
            ))
    
    return issues
