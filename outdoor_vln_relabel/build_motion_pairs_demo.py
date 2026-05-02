"""CLI demo for building terrain-aware instruction-path pairs from an npz file."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from outdoor_vln_relabel.io_utils import save_jsonl
from outdoor_vln_relabel.language.templates import (
    generate_landmark_instructions,
    generate_motion_instructions,
    generate_terrain_motion_instructions,
)
from outdoor_vln_relabel.perception.align import assign_landmark_roles
from outdoor_vln_relabel.perception.landmarks import (
    detect_landmarks_dummy,
    load_landmark_vocab,
)
from outdoor_vln_relabel.perception.terrain import (
    VALID_TERRAINS,
    classify_terrain_dummy,
    classify_terrain_from_metadata,
    load_terrain_taxonomy,
    normalize_terrain_name,
)
from outdoor_vln_relabel.schemas import InstructionPathPair
from outdoor_vln_relabel.trajectory.segment import segment_trajectory
from outdoor_vln_relabel.validation.checks import validate_pair


def load_npz_trajectory(path: str | Path) -> Tuple[object, object, object]:
    """Load positions, yaws, and timestamps arrays from a demo npz file."""
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Input npz file does not exist: {npz_path}")
    with np.load(npz_path) as data:
        required = ("positions", "yaws", "timestamps")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"{npz_path} missing required arrays: {', '.join(missing)}"
            )
        return data["positions"], data["yaws"], data["timestamps"]


def load_npz_terrain_labels(path: str | Path) -> Optional[object]:
    """Load optional per-frame or per-segment terrain labels from an npz file."""
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Input npz file does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "terrain_labels" not in data:
            return None
        return data["terrain_labels"].copy()


def build_motion_pairs(
    positions: Sequence[Sequence[float]],
    yaws: Sequence[float],
    timestamps: Sequence[float],
    scene_id: str,
    sequence_id: str = "demo_sequence",
    config: Dict[str, Any] | None = None,
    num_variants: int = 3,
    version: str = "v0_motion",
) -> List[InstructionPathPair]:
    """Build motion-only instruction-path pairs from trajectory arrays."""
    segment_config: Dict[str, Any] = dict(config or {})
    segment_config["scene_id"] = scene_id
    segment_config["sequence_id"] = sequence_id
    segments = segment_trajectory(positions, yaws, timestamps, segment_config)

    pairs: List[InstructionPathPair] = []
    instruction_id = 0
    for segment in segments:
        instructions = generate_motion_instructions(
            segment.motion, num_variants=num_variants
        )
        for instruction in instructions:
            instruction_id += 1
            pair = InstructionPathPair(
                scene_id=segment.scene_id,
                sequence_id=segment.sequence_id,
                segment_id=segment.segment_id,
                instruction_id=instruction_id,
                instruction=instruction,
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                start_frame=None,
                goal_frame=None,
                terrain=None,
                motion=segment.motion,
                landmarks=[],
                trajectory_xy=segment.trajectory_xy,
                distance_m=segment.distance_m,
                duration_s=segment.duration_s,
                heading_change_deg=segment.heading_change_deg,
                confidence=1.0,
                version=version,
            )
            if not validate_pair(pair):
                raise ValueError(
                    "Generated pair failed validation: "
                    f"segment_id={pair.segment_id}, instruction_id={pair.instruction_id}"
                )
            pairs.append(pair)
    return pairs


def _label_to_string(label: Any) -> str:
    """Convert numpy/string/bytes terrain labels to plain text."""
    if hasattr(label, "item"):
        label = label.item()
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def _safe_normalize_terrain(label: Any, taxonomy: Dict[str, Any], default: str) -> str:
    """Normalize one terrain label and return default when it is unknown."""
    fallback = normalize_terrain_name(default, taxonomy)
    if fallback not in VALID_TERRAINS:
        fallback = "dirt_trail"
    normalized = normalize_terrain_name(_label_to_string(label), taxonomy)
    if normalized not in VALID_TERRAINS:
        return fallback
    return normalized


def _terrain_from_labels(
    terrain_labels: object,
    segment: Any,
    segment_index: int,
    num_positions: int,
    num_segments: int,
    taxonomy: Dict[str, Any],
    default: str,
) -> str:
    """Resolve segment terrain from scalar, per-frame, or per-segment labels."""
    labels = np.asarray(terrain_labels)
    if labels.ndim == 0:
        return _safe_normalize_terrain(labels, taxonomy, default)

    label_count = len(labels)
    if label_count == num_positions:
        segment_labels = labels[segment.start_idx : segment.end_idx + 1]
        normalized = [
            _safe_normalize_terrain(label, taxonomy, default)
            for label in segment_labels.tolist()
        ]
        return Counter(normalized).most_common(1)[0][0]

    if label_count == num_segments:
        return _safe_normalize_terrain(labels[segment_index], taxonomy, default)

    if 0 <= segment.start_idx < label_count:
        return _safe_normalize_terrain(labels[segment.start_idx], taxonomy, default)

    return _safe_normalize_terrain(default, taxonomy, default)


def resolve_segment_terrain(
    segment: Any,
    segment_index: int,
    num_positions: int,
    num_segments: int,
    terrain_labels: Optional[object],
    terrain_mode: str,
    default_terrain: str,
    taxonomy: Dict[str, Any],
) -> str:
    """Resolve the terrain label for one segment using labels, metadata, or dummy mode."""
    if terrain_labels is not None:
        return _terrain_from_labels(
            terrain_labels=terrain_labels,
            segment=segment,
            segment_index=segment_index,
            num_positions=num_positions,
            num_segments=num_segments,
            taxonomy=taxonomy,
            default=default_terrain,
        )
    if terrain_mode == "metadata":
        if taxonomy:
            return _safe_normalize_terrain(default_terrain, taxonomy, default_terrain)
        return classify_terrain_from_metadata(
            {"terrain": default_terrain}, default=default_terrain
        )
    return classify_terrain_dummy(segment, default=default_terrain)


def build_terrain_motion_pairs(
    positions: Sequence[Sequence[float]],
    yaws: Sequence[float],
    timestamps: Sequence[float],
    scene_id: str,
    sequence_id: str = "demo_sequence",
    config: Dict[str, Any] | None = None,
    num_variants: int = 3,
    version: str = "terrain_motion_v1",
    default_terrain: str = "dirt_trail",
    terrain_mode: str = "dummy",
    terrain_taxonomy: Optional[str] = None,
    terrain_labels: Optional[object] = None,
    use_landmarks: bool = False,
    landmark_mode: str = "dummy",
    landmark_vocab: Optional[str] = None,
) -> List[InstructionPathPair]:
    """Build terrain-aware or landmark-aware instruction-path pairs."""
    if terrain_mode not in {"dummy", "metadata"}:
        raise ValueError("terrain_mode must be one of: dummy, metadata")
    if landmark_mode not in {"dummy", "metadata"}:
        raise ValueError("landmark_mode must be one of: dummy, metadata")
    if use_landmarks and landmark_mode != "dummy":
        raise NotImplementedError(
            "landmark_mode=metadata is reserved for future detector metadata support"
        )
    taxonomy = load_terrain_taxonomy(terrain_taxonomy)
    landmark_vocab_data = load_landmark_vocab(landmark_vocab) if use_landmarks else None
    segment_config: Dict[str, Any] = dict(config or {})
    segment_config["scene_id"] = scene_id
    segment_config["sequence_id"] = sequence_id
    segments = segment_trajectory(positions, yaws, timestamps, segment_config)

    pairs: List[InstructionPathPair] = []
    instruction_id = 0
    num_positions = len(positions)
    output_version = version
    if use_landmarks and output_version == "terrain_motion_v1":
        output_version = "landmark_terrain_motion_v2"

    for segment_index, segment in enumerate(segments):
        terrain = resolve_segment_terrain(
            segment=segment,
            segment_index=segment_index,
            num_positions=num_positions,
            num_segments=len(segments),
            terrain_labels=terrain_labels,
            terrain_mode=terrain_mode,
            default_terrain=default_terrain,
            taxonomy=taxonomy,
        )
        landmarks = []
        confidence = 1.0
        if use_landmarks:
            landmarks = detect_landmarks_dummy(
                segment, terrain=terrain, vocab=landmark_vocab_data
            )
            structured_label = assign_landmark_roles(
                landmarks, segment=segment, terrain=terrain
            )
            instructions = generate_landmark_instructions(
                segment.motion,
                terrain,
                structured_label,
                num_variants=num_variants,
            )
            confidence = structured_label.confidence
        else:
            instructions = generate_terrain_motion_instructions(
                segment.motion, terrain, num_variants=num_variants
            )

        landmark_dicts = [landmark.to_dict() for landmark in landmarks]
        for instruction in instructions:
            instruction_id += 1
            pair = InstructionPathPair(
                scene_id=segment.scene_id,
                sequence_id=segment.sequence_id,
                segment_id=segment.segment_id,
                instruction_id=instruction_id,
                instruction=instruction,
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                start_frame=None,
                goal_frame=None,
                terrain=terrain,
                motion=segment.motion,
                landmarks=landmark_dicts,
                trajectory_xy=segment.trajectory_xy,
                distance_m=segment.distance_m,
                duration_s=segment.duration_s,
                heading_change_deg=segment.heading_change_deg,
                confidence=confidence,
                version=output_version,
            )
            if not validate_pair(pair):
                raise ValueError(
                    "Generated terrain pair failed validation: "
                    f"segment_id={pair.segment_id}, instruction_id={pair.instruction_id}"
                )
            pairs.append(pair)
    return pairs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Stage-2 demo."""
    parser = argparse.ArgumentParser(
        description="Build terrain-aware Outdoor-VLN instruction-path pairs from npz."
    )
    parser.add_argument("--input_npz", required=True, help="Path to input trajectory npz")
    parser.add_argument(
        "--output_jsonl", required=True, help="Path to output instruction JSONL"
    )
    parser.add_argument("--scene_id", required=True, help="Scene id for generated pairs")
    parser.add_argument(
        "--sequence_id",
        default="demo_sequence",
        help="Sequence id for generated pairs",
    )
    parser.add_argument(
        "--default_terrain",
        default="dirt_trail",
        help="Default terrain label when metadata is unavailable",
    )
    parser.add_argument(
        "--terrain_mode",
        choices=["dummy", "metadata"],
        default="dummy",
        help="Terrain source mode. npz terrain_labels are used when present.",
    )
    parser.add_argument(
        "--terrain_taxonomy",
        default=None,
        help="Optional path to terrain taxonomy YAML",
    )
    parser.add_argument(
        "--use_landmarks",
        action="store_true",
        help="Enable dummy landmark-aware instruction generation",
    )
    parser.add_argument(
        "--landmark_mode",
        choices=["dummy", "metadata"],
        default="dummy",
        help="Landmark source mode. Stage 3 supports dummy for npz inputs.",
    )
    parser.add_argument(
        "--landmark_vocab",
        default=None,
        help="Optional path to landmark vocabulary YAML",
    )
    return parser.parse_args()


def main() -> None:
    """Run the terrain-aware npz-to-JSONL demo pipeline."""
    args = parse_args()
    positions, yaws, timestamps = load_npz_trajectory(args.input_npz)
    terrain_labels = load_npz_terrain_labels(args.input_npz)
    pairs = build_terrain_motion_pairs(
        positions=positions,
        yaws=yaws,
        timestamps=timestamps,
        scene_id=args.scene_id,
        sequence_id=args.sequence_id,
        default_terrain=args.default_terrain,
        terrain_mode=args.terrain_mode,
        terrain_taxonomy=args.terrain_taxonomy,
        terrain_labels=terrain_labels,
        use_landmarks=args.use_landmarks,
        landmark_mode=args.landmark_mode,
        landmark_vocab=args.landmark_vocab,
    )
    save_jsonl(pairs, args.output_jsonl)
    print(f"Generated {len(pairs)} instruction-path pairs at {args.output_jsonl}")


if __name__ == "__main__":
    main()
