"""Perception placeholders for terrain and landmark-aware relabeling."""

from .align import assign_landmark_roles
from .landmarks import detect_landmarks_dummy
from .landmarks import (
    detect_landmarks_from_metadata,
    detect_landmarks_from_semantic_mask_dummy,
    get_default_role_for_category,
    infer_landmark_category,
    load_landmark_vocab,
    normalize_landmark_name,
)
from .landmarks_from_mask import (
    detect_landmarks_from_mask,
    detect_landmarks_from_segment_masks,
)
from .semantic_mask import load_label_map, parse_mask_labels, read_semantic_mask
from .terrain import (
    classify_terrain_from_frames,
    classify_terrain_from_semantic_mask_dummy,
    get_terrain_properties,
    classify_terrain_dummy,
    classify_terrain_from_metadata,
    load_terrain_taxonomy,
    normalize_terrain_name,
)
from .terrain_from_mask import (
    classify_terrain_from_mask,
    classify_terrain_from_segment_masks,
)

__all__ = [
    "assign_landmark_roles",
    "get_terrain_properties",
    "classify_terrain_dummy",
    "classify_terrain_from_frames",
    "classify_terrain_from_metadata",
    "classify_terrain_from_semantic_mask_dummy",
    "detect_landmarks_from_metadata",
    "detect_landmarks_from_mask",
    "detect_landmarks_from_semantic_mask_dummy",
    "detect_landmarks_from_segment_masks",
    "detect_landmarks_dummy",
    "get_default_role_for_category",
    "infer_landmark_category",
    "classify_terrain_from_mask",
    "classify_terrain_from_segment_masks",
    "load_landmark_vocab",
    "load_label_map",
    "load_terrain_taxonomy",
    "normalize_landmark_name",
    "normalize_terrain_name",
    "parse_mask_labels",
    "read_semantic_mask",
]
