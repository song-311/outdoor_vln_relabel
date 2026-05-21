"""Structured schemas for VLM-generated multi-level navigation instructions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedLandmark:
    """A landmark referenced by the VLM in its instruction generation."""

    name: str
    category: str
    role: str
    relation: str
    bbox: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedLandmark":
        bbox = data.get("bbox")
        return cls(
            name=str(data.get("name", "")),
            category=str(data.get("category", "")),
            role=str(data.get("role", "")),
            relation=str(data.get("relation", "")),
            bbox=[int(v) for v in bbox] if bbox is not None else None,
        )


@dataclass
class VLMInstruction:
    """Three-level navigation instruction produced by the VLM."""

    extracted_landmarks: List[ExtractedLandmark] = field(default_factory=list)
    global_instruction: str = ""
    meso_instruction: str = ""
    micro_instruction: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_landmarks": [lm.to_dict() for lm in self.extracted_landmarks],
            "global_instruction": self.global_instruction,
            "meso_instruction": self.meso_instruction,
            "micro_instruction": self.micro_instruction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMInstruction":
        landmarks = data.get("extracted_landmarks", [])
        if isinstance(landmarks, list):
            parsed = [
                ExtractedLandmark.from_dict(item) if isinstance(item, dict) else item
                for item in landmarks
            ]
        else:
            parsed = []
        return cls(
            extracted_landmarks=parsed,
            global_instruction=str(data.get("global_instruction", "")),
            meso_instruction=str(data.get("meso_instruction", "")),
            micro_instruction=str(data.get("micro_instruction", "")),
        )
