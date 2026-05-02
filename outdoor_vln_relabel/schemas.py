"""Typed records used by the Outdoor-VLN relabeling pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _require(data: Dict[str, Any], key: str, cls_name: str) -> Any:
    """Return a required field or raise a clear schema error."""
    if key not in data:
        raise ValueError(f"{cls_name}.from_dict missing required field: {key}")
    return data[key]


@dataclass
class FrameRecord:
    """Single synchronized robot frame with optional sensor assets."""

    scene_id: str
    sequence_id: str
    frame_id: int
    timestamp: float
    rgb_path: str
    depth_path: Optional[str]
    lidar_path: Optional[str]
    semantic_path: Optional[str]
    position: List[float]
    yaw: float
    cmd_vel: Optional[List[float]]
    terrain: Optional[str] = None
    landmarks: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the frame record to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameRecord":
        """Build a frame record from a dictionary with clear missing-field errors."""
        return cls(
            scene_id=str(_require(data, "scene_id", cls.__name__)),
            sequence_id=str(_require(data, "sequence_id", cls.__name__)),
            frame_id=int(_require(data, "frame_id", cls.__name__)),
            timestamp=float(_require(data, "timestamp", cls.__name__)),
            rgb_path=str(_require(data, "rgb_path", cls.__name__)),
            depth_path=data.get("depth_path"),
            lidar_path=data.get("lidar_path"),
            semantic_path=data.get("semantic_path"),
            position=[float(v) for v in _require(data, "position", cls.__name__)],
            yaw=float(_require(data, "yaw", cls.__name__)),
            cmd_vel=(
                [float(v) for v in data["cmd_vel"]]
                if data.get("cmd_vel") is not None
                else None
            ),
            terrain=data.get("terrain"),
            landmarks=(
                [dict(item) for item in data["landmarks"]]
                if data.get("landmarks") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})) if data.get("metadata") else None,
        )


@dataclass
class PathSegment:
    """Local trajectory segment used as the geometric path in one sample."""

    scene_id: str
    sequence_id: str
    segment_id: str
    start_idx: int
    end_idx: int
    keyframe_indices: List[int]
    distance_m: float
    duration_s: float
    heading_change_deg: float
    motion: str
    trajectory_xy: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the path segment to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathSegment":
        """Build a path segment from a dictionary."""
        return cls(
            scene_id=str(_require(data, "scene_id", cls.__name__)),
            sequence_id=str(_require(data, "sequence_id", cls.__name__)),
            segment_id=str(_require(data, "segment_id", cls.__name__)),
            start_idx=int(_require(data, "start_idx", cls.__name__)),
            end_idx=int(_require(data, "end_idx", cls.__name__)),
            keyframe_indices=[
                int(v) for v in _require(data, "keyframe_indices", cls.__name__)
            ],
            distance_m=float(_require(data, "distance_m", cls.__name__)),
            duration_s=float(_require(data, "duration_s", cls.__name__)),
            heading_change_deg=float(
                _require(data, "heading_change_deg", cls.__name__)
            ),
            motion=str(_require(data, "motion", cls.__name__)),
            trajectory_xy=[
                [float(x), float(y)]
                for x, y in _require(data, "trajectory_xy", cls.__name__)
            ],
        )


@dataclass
class Landmark:
    """Outdoor landmark or terrain object aligned with a path segment."""

    name: str
    category: str
    role: str
    relation: str
    bbox: Optional[List[int]]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the landmark to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Landmark":
        """Build a landmark from a dictionary."""
        bbox = data.get("bbox")
        return cls(
            name=str(_require(data, "name", cls.__name__)),
            category=str(_require(data, "category", cls.__name__)),
            role=str(_require(data, "role", cls.__name__)),
            relation=str(_require(data, "relation", cls.__name__)),
            bbox=[int(v) for v in bbox] if bbox is not None else None,
            score=float(data.get("score", 1.0)),
        )


@dataclass
class StructuredLabel:
    """Structured navigation semantics before natural-language generation."""

    terrain: str
    motion: str
    goal_landmark: Optional[Landmark]
    constraint_landmarks: List[Landmark] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the structured label to a JSON-compatible dictionary."""
        return {
            "terrain": self.terrain,
            "motion": self.motion,
            "goal_landmark": (
                self.goal_landmark.to_dict() if self.goal_landmark else None
            ),
            "constraint_landmarks": [
                landmark.to_dict() for landmark in self.constraint_landmarks
            ],
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredLabel":
        """Build a structured label from a dictionary."""
        goal = data.get("goal_landmark")
        return cls(
            terrain=str(_require(data, "terrain", cls.__name__)),
            motion=str(_require(data, "motion", cls.__name__)),
            goal_landmark=Landmark.from_dict(goal) if goal is not None else None,
            constraint_landmarks=[
                Landmark.from_dict(item)
                for item in data.get("constraint_landmarks", [])
            ],
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class InstructionPathPair:
    """Final JSONL sample containing one instruction and one trajectory path."""

    scene_id: str
    sequence_id: str
    segment_id: str
    instruction_id: int
    instruction: str
    start_idx: int
    end_idx: int
    start_frame: Optional[str]
    goal_frame: Optional[str]
    terrain: Optional[str]
    motion: str
    landmarks: List[Dict[str, Any]]
    trajectory_xy: List[List[float]]
    distance_m: float
    duration_s: float
    heading_change_deg: float
    confidence: float
    version: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the instruction-path pair to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionPathPair":
        """Build an instruction-path pair from a dictionary."""
        return cls(
            scene_id=str(_require(data, "scene_id", cls.__name__)),
            sequence_id=str(_require(data, "sequence_id", cls.__name__)),
            segment_id=str(_require(data, "segment_id", cls.__name__)),
            instruction_id=int(_require(data, "instruction_id", cls.__name__)),
            instruction=str(_require(data, "instruction", cls.__name__)),
            start_idx=int(_require(data, "start_idx", cls.__name__)),
            end_idx=int(_require(data, "end_idx", cls.__name__)),
            start_frame=data.get("start_frame"),
            goal_frame=data.get("goal_frame"),
            terrain=data.get("terrain"),
            motion=str(_require(data, "motion", cls.__name__)),
            landmarks=list(data.get("landmarks", [])),
            trajectory_xy=[
                [float(x), float(y)]
                for x, y in _require(data, "trajectory_xy", cls.__name__)
            ],
            distance_m=float(_require(data, "distance_m", cls.__name__)),
            duration_s=float(_require(data, "duration_s", cls.__name__)),
            heading_change_deg=float(
                _require(data, "heading_change_deg", cls.__name__)
            ),
            confidence=float(data.get("confidence", 1.0)),
            version=str(data.get("version", "v0_motion")),
        )
