"""Dataset-building CLI entry point for Outdoor-VLN relabeling."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from outdoor_vln_relabel.build_motion_pairs_demo import build_terrain_motion_pairs
from outdoor_vln_relabel.dataset_adapters.base_adapter import NpzTrajectoryAdapter
from outdoor_vln_relabel.dataset_adapters.botanicgarden_adapter import (
    BotanicGardenAdapter,
)
from outdoor_vln_relabel.dataset_adapters.manifest_adapter import ManifestDatasetAdapter
from outdoor_vln_relabel.dataset_adapters.rellis_adapter import RellisAdapter
from outdoor_vln_relabel.dataset_adapters.rosbag_adapter import RosbagAdapter
from outdoor_vln_relabel.dataset_adapters.tartanground_adapter import (
    TartanGroundAdapter,
)
from outdoor_vln_relabel.io_utils import save_jsonl
from outdoor_vln_relabel.language.templates import (
    generate_landmark_instructions,
    generate_terrain_motion_instructions,
)
from outdoor_vln_relabel.perception.align import assign_landmark_roles
from outdoor_vln_relabel.perception.landmarks import (
    detect_landmarks_dummy,
    detect_landmarks_from_metadata,
    detect_landmarks_from_semantic_mask_dummy,
    load_landmark_vocab,
)
from outdoor_vln_relabel.perception.landmarks_from_mask import (
    detect_landmarks_from_segment_masks,
)
from outdoor_vln_relabel.perception.semantic_mask import load_label_map
from outdoor_vln_relabel.perception.terrain import (
    VALID_TERRAINS,
    classify_terrain_from_frames,
    classify_terrain_from_semantic_mask_dummy,
    load_terrain_taxonomy,
    normalize_terrain_name,
)
from outdoor_vln_relabel.perception.terrain_from_mask import (
    classify_terrain_from_segment_masks,
)
from outdoor_vln_relabel.schemas import FrameRecord, InstructionPathPair, Landmark
from outdoor_vln_relabel.trajectory.segment import segment_trajectory
from outdoor_vln_relabel.validation.checks import validate_pair


def load_config(path: str | None) -> Dict[str, Any]:
    """Load an optional YAML configuration file."""
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    try:
        import yaml
    except ImportError:
        data = _load_simple_yaml(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return data


def _parse_scalar(value: str) -> Any:
    """Parse a small YAML scalar subset used by Stage-1 config files."""
    value = value.strip()
    if value == "":
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid inline list value in config: {value}") from exc
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Load a minimal indentation-based YAML mapping without third-party deps."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if stripped.startswith("- "):
                raise ValueError(
                    "Simple YAML fallback only supports mappings and inline lists; "
                    f"found block list on line {line_number} of {path}"
                )
            if ":" not in stripped:
                raise ValueError(f"Invalid config line {line_number} in {path}: {raw_line}")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if raw_value.strip() == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_scalar(raw_value)
    return root


def make_adapter(
    adapter_name: str, input_dir: str, scene_id: str, sequence_id: str
):
    """Construct a dataset adapter by name."""
    if adapter_name in {"npz", "demo"}:
        return NpzTrajectoryAdapter(input_dir, scene_id, sequence_id)
    if adapter_name == "manifest":
        return ManifestDatasetAdapter(input_dir, scene_id, sequence_id)
    if adapter_name == "botanicgarden":
        return BotanicGardenAdapter(input_dir, scene_id, sequence_id)
    if adapter_name == "rellis":
        return RellisAdapter(input_dir, scene_id, sequence_id)
    if adapter_name == "tartanground":
        return TartanGroundAdapter(input_dir, scene_id, sequence_id)
    if adapter_name == "rosbag":
        return RosbagAdapter(input_dir, scene_id, sequence_id)
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Build Outdoor-VLN instruction-path pairs from a dataset adapter."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        choices=[
            "demo",
            "npz",
            "manifest",
            "botanicgarden",
            "rellis",
            "tartanground",
            "rosbag",
        ],
        help="Source dataset adapter",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Input directory or direct npz path for the selected adapter",
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Manifest dataset root containing trajectory.csv",
    )
    parser.add_argument(
        "--output_jsonl", required=True, help="Path to output instruction JSONL"
    )
    parser.add_argument("--scene_id", required=True, help="Scene id for generated pairs")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument(
        "--sequence_id",
        default="default_sequence",
        help="Sequence id for generated pairs",
    )
    parser.add_argument(
        "--default_terrain",
        default="dirt_trail",
        help="Default terrain label when metadata is unavailable",
    )
    parser.add_argument(
        "--terrain_mode",
        choices=["dummy", "metadata", "mask", "metadata_or_mask"],
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
        choices=["dummy", "metadata", "mask", "metadata_or_mask", "metadata_or_dummy"],
        default="dummy",
        help="Landmark source mode. Manifest supports metadata and mask evidence.",
    )
    parser.add_argument(
        "--landmark_vocab",
        default=None,
        help="Optional path to landmark vocabulary YAML",
    )
    parser.add_argument(
        "--semantic_label_map",
        default=None,
        help="Optional semantic mask label-map YAML for real mask parsing",
    )
    return parser.parse_args()


def _adapter_input_path(args: argparse.Namespace) -> str:
    """Resolve adapter input path from --dataset_root or --input_dir."""
    if args.adapter == "manifest":
        input_path = args.dataset_root or args.input_dir
        if not input_path:
            raise ValueError("--adapter manifest requires --dataset_root")
        return str(input_path)
    if not args.input_dir:
        raise ValueError(f"--adapter {args.adapter} requires --input_dir")
    return str(args.input_dir)


def _metadata_terrain_optional(
    frames: List[FrameRecord],
    start_idx: int,
    end_idx: int,
    taxonomy: Dict[str, Any],
) -> Optional[str]:
    """Return majority metadata terrain, or None when no frame has terrain evidence."""
    labels: List[str] = []
    for frame in frames[start_idx : end_idx + 1]:
        raw = frame.terrain
        if not raw and frame.metadata:
            raw = (
                frame.metadata.get("terrain")
                or frame.metadata.get("terrain_label")
                or frame.metadata.get("terrain_class")
            )
        if raw:
            normalized = normalize_terrain_name(str(raw), taxonomy)
            if normalized in VALID_TERRAINS:
                labels.append(normalized)
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _mask_terrain_optional(
    frames: List[FrameRecord],
    start_idx: int,
    end_idx: int,
    taxonomy: Dict[str, Any],
    semantic_label_map: Optional[Dict[str, Any]] = None,
    default_terrain: str = "dirt_trail",
) -> Optional[str]:
    """Return terrain from real semantic masks or filename evidence."""
    has_existing_mask = any(
        frame.semantic_path and Path(str(frame.semantic_path)).is_file()
        for frame in frames[start_idx : end_idx + 1]
    )
    if semantic_label_map and has_existing_mask:
        return classify_terrain_from_segment_masks(
            frames,
            start_idx,
            end_idx,
            semantic_label_map,
            default=default_terrain,
        )

    labels: List[str] = []
    for frame in frames[start_idx : end_idx + 1]:
        if not frame.semantic_path:
            continue
        terrain = classify_terrain_from_semantic_mask_dummy(
            frame.semantic_path, taxonomy
        )
        if terrain:
            labels.append(terrain)
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _resolve_manifest_terrain(
    frames: List[FrameRecord],
    start_idx: int,
    end_idx: int,
    terrain_mode: str,
    default_terrain: str,
    taxonomy: Dict[str, Any],
    semantic_label_map: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve terrain for a manifest segment using metadata and mask evidence."""
    fallback = normalize_terrain_name(default_terrain, taxonomy)
    if fallback not in VALID_TERRAINS:
        fallback = "dirt_trail"
    if terrain_mode == "metadata":
        return classify_terrain_from_frames(
            frames, start_idx, end_idx, default=fallback, taxonomy=taxonomy
        )
    if terrain_mode == "mask":
        return (
            _mask_terrain_optional(
                frames,
                start_idx,
                end_idx,
                taxonomy,
                semantic_label_map=semantic_label_map,
                default_terrain=fallback,
            )
            or fallback
        )
    if terrain_mode == "metadata_or_mask":
        return (
            _metadata_terrain_optional(frames, start_idx, end_idx, taxonomy)
            or _mask_terrain_optional(
                frames,
                start_idx,
                end_idx,
                taxonomy,
                semantic_label_map=semantic_label_map,
                default_terrain=fallback,
            )
            or fallback
        )
    return fallback


def _resolve_manifest_landmarks(
    frames: List[FrameRecord],
    start_idx: int,
    end_idx: int,
    terrain: str,
    landmark_mode: str,
    vocab: Dict[str, Any],
    segment: Any,
    semantic_label_map: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Landmark], str]:
    """Resolve landmarks for a manifest segment and return evidence source."""
    metadata_landmarks = detect_landmarks_from_metadata(frames, start_idx, end_idx, vocab)
    has_existing_mask = any(
        frame.semantic_path and Path(str(frame.semantic_path)).is_file()
        for frame in frames[start_idx : end_idx + 1]
    )
    if semantic_label_map and has_existing_mask:
        mask_landmarks = detect_landmarks_from_segment_masks(
            frames, start_idx, end_idx, semantic_label_map, terrain
        )
        if not mask_landmarks:
            mask_landmarks = detect_landmarks_from_semantic_mask_dummy(
                frames, start_idx, end_idx, terrain, vocab
            )
    else:
        mask_landmarks = detect_landmarks_from_semantic_mask_dummy(
            frames, start_idx, end_idx, terrain, vocab
        )

    if landmark_mode == "metadata":
        return metadata_landmarks, "metadata"
    if landmark_mode == "mask":
        return mask_landmarks, "mask"
    if landmark_mode == "metadata_or_mask":
        if metadata_landmarks:
            merged = _merge_landmarks(metadata_landmarks, mask_landmarks)
            return merged, "metadata"
        if mask_landmarks:
            return mask_landmarks, "mask"
        return detect_landmarks_dummy(segment, terrain=terrain, vocab=vocab), "dummy"
    if landmark_mode == "metadata_or_dummy":
        if metadata_landmarks:
            return metadata_landmarks, "metadata"
        return detect_landmarks_dummy(segment, terrain=terrain, vocab=vocab), "dummy"
    return detect_landmarks_dummy(segment, terrain=terrain, vocab=vocab), "dummy"


def _merge_landmarks(primary: List[Landmark], secondary: List[Landmark]) -> List[Landmark]:
    """Merge landmarks by normalized name, preserving primary metadata first."""
    merged = list(primary)
    existing = {landmark.name.lower() for landmark in merged}
    for landmark in secondary:
        if landmark.name.lower() not in existing:
            merged.append(landmark)
            existing.add(landmark.name.lower())
    return merged


def _build_manifest_pairs(
    frames: List[FrameRecord],
    scene_id: str,
    sequence_id: str,
    config: Dict[str, Any],
    num_variants: int,
    default_terrain: str,
    terrain_mode: str,
    terrain_taxonomy: Optional[str],
    use_landmarks: bool,
    landmark_mode: str,
    landmark_vocab_path: Optional[str],
    semantic_label_map_path: Optional[str] = None,
) -> List[InstructionPathPair]:
    """Build instruction-path pairs from manifest FrameRecord evidence."""
    taxonomy = load_terrain_taxonomy(terrain_taxonomy)
    vocab = load_landmark_vocab(landmark_vocab_path)
    semantic_label_map = (
        load_label_map(semantic_label_map_path) if semantic_label_map_path else None
    )
    positions = [frame.position for frame in frames]
    yaws = [frame.yaw for frame in frames]
    timestamps = [frame.timestamp for frame in frames]
    segment_config: Dict[str, Any] = dict(config or {})
    segment_config["scene_id"] = scene_id
    segment_config["sequence_id"] = sequence_id
    segments = segment_trajectory(positions, yaws, timestamps, segment_config)

    pairs: List[InstructionPathPair] = []
    instruction_id = 0
    for segment in segments:
        terrain = _resolve_manifest_terrain(
            frames,
            segment.start_idx,
            segment.end_idx,
            terrain_mode,
            default_terrain,
            taxonomy,
            semantic_label_map=semantic_label_map,
        )
        landmarks: List[Landmark] = []
        confidence = 1.0
        version = "terrain_motion_v1"
        if use_landmarks:
            landmarks, source = _resolve_manifest_landmarks(
                frames,
                segment.start_idx,
                segment.end_idx,
                terrain,
                landmark_mode,
                vocab,
                segment,
                semantic_label_map=semantic_label_map,
            )
            structured_label = assign_landmark_roles(
                landmarks, segment=segment, terrain=terrain
            )
            instructions = generate_landmark_instructions(
                segment.motion, terrain, structured_label, num_variants=num_variants
            )
            confidence = structured_label.confidence
            version = (
                "landmark_terrain_motion_v2"
                if source == "dummy"
                else "evidence_landmark_terrain_motion_v3"
            )
        else:
            instructions = generate_terrain_motion_instructions(
                segment.motion, terrain, num_variants=num_variants
            )

        landmark_dicts = [landmark.to_dict() for landmark in landmarks]
        for instruction in instructions:
            instruction_id += 1
            pair = InstructionPathPair(
                scene_id=scene_id,
                sequence_id=sequence_id,
                segment_id=segment.segment_id,
                instruction_id=instruction_id,
                instruction=instruction,
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                start_frame=frames[segment.start_idx].rgb_path,
                goal_frame=frames[segment.end_idx].rgb_path,
                terrain=terrain,
                motion=segment.motion,
                landmarks=landmark_dicts,
                trajectory_xy=segment.trajectory_xy,
                distance_m=segment.distance_m,
                duration_s=segment.duration_s,
                heading_change_deg=segment.heading_change_deg,
                confidence=confidence,
                version=version,
            )
            if not validate_pair(pair):
                raise ValueError(
                    "Generated manifest pair failed validation: "
                    f"segment_id={pair.segment_id}, instruction_id={pair.instruction_id}"
                )
            pairs.append(pair)
    return pairs


def main() -> None:
    """Run a configured dataset adapter and save generated JSONL records."""
    args = parse_args()
    config = load_config(args.config)
    sequence_id = str(config.get("sequence_id", args.sequence_id))
    input_path = _adapter_input_path(args)
    adapter = make_adapter(args.adapter, input_path, args.scene_id, sequence_id)
    positions, yaws, timestamps = adapter.load_trajectory_arrays()
    terrain_labels = adapter.load_terrain_labels()
    language_config = config.get("language", {})
    num_variants = int(language_config.get("instruction_variants", 3))
    version = str(config.get("version", "terrain_motion_v1"))
    default_terrain = args.default_terrain
    terrain_mode = args.terrain_mode
    terrain_config = config.get("terrain", {})
    terrain_taxonomy = args.terrain_taxonomy or terrain_config.get("taxonomy")
    landmark_config = config.get("landmark", {})
    landmark_vocab = args.landmark_vocab or landmark_config.get("vocab")
    semantic_config = config.get("semantic", {})
    semantic_label_map = args.semantic_label_map or semantic_config.get("label_map")

    if args.adapter == "manifest":
        pairs = _build_manifest_pairs(
            frames=adapter.load_frames(),
            scene_id=args.scene_id,
            sequence_id=sequence_id,
            config=config,
            num_variants=num_variants,
            default_terrain=default_terrain,
            terrain_mode=terrain_mode,
            terrain_taxonomy=terrain_taxonomy,
            use_landmarks=args.use_landmarks,
            landmark_mode=args.landmark_mode,
            landmark_vocab_path=landmark_vocab,
            semantic_label_map_path=semantic_label_map,
        )
    else:
        pairs = build_terrain_motion_pairs(
            positions=positions,
            yaws=yaws,
            timestamps=timestamps,
            scene_id=args.scene_id,
            sequence_id=sequence_id,
            config=config,
            num_variants=num_variants,
            version=version,
            default_terrain=default_terrain,
            terrain_mode=terrain_mode,
            terrain_taxonomy=terrain_taxonomy,
            terrain_labels=terrain_labels,
            use_landmarks=args.use_landmarks,
            landmark_mode=args.landmark_mode,
            landmark_vocab=landmark_vocab,
        )
    save_jsonl(pairs, args.output_jsonl)
    print(
        f"Generated {len(pairs)} instruction-path pairs with adapter "
        f"'{args.adapter}' at {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
