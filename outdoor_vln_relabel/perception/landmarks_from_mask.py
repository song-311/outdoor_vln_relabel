"""Landmark extraction from semantic mask evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from outdoor_vln_relabel.perception.semantic_mask import (
    parse_mask_labels,
    read_semantic_mask,
)
from outdoor_vln_relabel.schemas import Landmark


VALID_ROLES = {
    "follow",
    "avoid",
    "turn_at",
    "pass_between",
    "stop_near",
    "go_toward",
}


def _clean_name(name: str) -> str:
    """Normalize a semantic label name for rule matching."""
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _infer_category_and_role(name: str, outdoor_group: str, raw_role: str) -> Tuple[str, str]:
    """Map semantic label metadata to an Outdoor-VLN landmark category and role."""
    cleaned = _clean_name(name)
    role = str(raw_role or "").strip()
    if role == "cautious_follow":
        role = "follow"

    if any(token in cleaned for token in ("road", "trail", "path")):
        return "path_like", "follow"
    if cleaned in {"grass", "grass field", "meadow", "open area", "clearing"}:
        return "region_like", "go_toward" if role not in VALID_ROLES else role
    if any(token in cleaned for token in ("bush", "shrub", "vegetation")):
        return "obstacle_like", "avoid"
    if any(token in cleaned for token in ("mud", "puddle", "water", "wet")):
        return "obstacle_like", "avoid"
    if any(token in cleaned for token in ("rock", "tree trunk", "fallen branch")):
        return "obstacle_like", "avoid"
    if any(token in cleaned for token in ("gravel", "rough", "slope", "uneven")):
        return "geometry_like", "follow"

    if outdoor_group in {"dirt_trail"}:
        return "path_like", "follow"
    if outdoor_group == "grass":
        return "region_like", "go_toward"
    if outdoor_group in {"vegetation", "mud_water"}:
        return "obstacle_like", "avoid"
    if outdoor_group == "rough_terrain":
        return "geometry_like", "follow"
    if role in VALID_ROLES:
        return "unknown", role
    return "unknown", "go_toward"


def _relation_from_center(center: List[float], width: int, height: int) -> str:
    """Infer a coarse left/right/ahead relation from a mask component center."""
    cx, cy = float(center[0]), float(center[1])
    x_ratio = cx / max(float(width), 1.0)
    y_ratio = cy / max(float(height), 1.0)
    if y_ratio >= 0.65 and x_ratio < 0.4:
        return "front_left"
    if y_ratio >= 0.65 and x_ratio > 0.6:
        return "front_right"
    if x_ratio < 0.4:
        return "left"
    if x_ratio > 0.6:
        return "right"
    return "ahead"


def _landmark_name(name: str, outdoor_group: str) -> str:
    """Return a human-friendly landmark name from a semantic label."""
    cleaned = _clean_name(name)
    if cleaned == "grass":
        return "grass field"
    if cleaned in {"dirt road", "dirt path", "trail", "path"}:
        return cleaned
    if cleaned == "vegetation":
        return "dense vegetation"
    if outdoor_group == "rough_terrain" and cleaned == "gravel":
        return "gravel path"
    return cleaned


def detect_landmarks_from_mask(
    mask_path: str,
    label_map: Dict[str, Any],
    min_area_ratio: float = 0.005,
) -> List[Landmark]:
    """Detect coarse landmarks from one semantic mask.

    This is not an object detector. It converts sufficiently large semantic
    regions into grounded evidence records with bbox, relation, role, and score.
    """
    mask = read_semantic_mask(mask_path)
    parsed = parse_mask_labels(mask, label_map)
    width = int(parsed["width"])
    height = int(parsed["height"])
    landmarks: List[Landmark] = []
    for label in parsed["labels"]:
        outdoor_group = str(label.get("outdoor_group", "unknown"))
        role = str(label.get("role", "ignore"))
        if role == "ignore" or outdoor_group == "unknown":
            continue
        area_ratio = float(label.get("area_ratio", 0.0))
        if area_ratio < min_area_ratio:
            continue
        name = _landmark_name(str(label.get("name", "unknown")), outdoor_group)
        category, landmark_role = _infer_category_and_role(name, outdoor_group, role)
        relation = _relation_from_center(label.get("center", [0.0, 0.0]), width, height)
        if category == "path_like" and landmark_role == "follow":
            relation = "ahead"
        score = min(0.95, 0.5 + area_ratio * 5.0)
        landmarks.append(
            Landmark(
                name=name,
                category=category,
                role=landmark_role,
                relation=relation,
                bbox=[int(value) for value in label.get("bbox", [])],
                score=round(float(score), 6),
            )
        )
    return sorted(landmarks, key=lambda landmark: -landmark.score)


def _merge_landmark(existing: Landmark, incoming: Landmark) -> Landmark:
    """Merge two same-name/relation landmarks by preserving the stronger evidence."""
    score = max(float(existing.score), float(incoming.score))
    bbox = incoming.bbox if float(incoming.score) >= float(existing.score) else existing.bbox
    role = existing.role if existing.role else incoming.role
    category = existing.category if existing.category else incoming.category
    return Landmark(
        name=existing.name,
        category=category,
        role=role,
        relation=existing.relation,
        bbox=bbox,
        score=round(score, 6),
    )


def detect_landmarks_from_segment_masks(
    frames: List[Any],
    start_idx: int,
    end_idx: int,
    label_map: Dict[str, Any],
    terrain: str,
) -> List[Landmark]:
    """Detect and merge semantic-mask landmarks across a trajectory segment."""
    del terrain
    grouped: Dict[Tuple[str, str], Landmark] = {}
    for frame in frames[start_idx : end_idx + 1]:
        semantic_path = getattr(frame, "semantic_path", None)
        if not semantic_path or not Path(str(semantic_path)).is_file():
            continue
        for landmark in detect_landmarks_from_mask(str(semantic_path), label_map):
            key = (_clean_name(landmark.name), str(landmark.relation))
            if key in grouped:
                grouped[key] = _merge_landmark(grouped[key], landmark)
            else:
                grouped[key] = landmark
    ranked = sorted(grouped.values(), key=lambda landmark: -float(landmark.score))
    return ranked[:5]
