"""Terrain classification from semantic mask evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from outdoor_vln_relabel.perception.semantic_mask import (
    parse_mask_labels,
    read_semantic_mask,
)
from outdoor_vln_relabel.perception.terrain import VALID_TERRAINS


def _empty_distribution() -> Dict[str, float]:
    """Return a zero-filled terrain distribution over the five classes."""
    return {terrain: 0.0 for terrain in sorted(VALID_TERRAINS)}


def _safe_roi_bounds(
    height: int, width: int, ground_roi: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    """Convert fractional ROI bounds into clamped pixel coordinates."""
    y0_frac, y1_frac, x0_frac, x1_frac = ground_roi
    y0 = max(0, min(height, int(round(height * y0_frac))))
    y1 = max(y0 + 1, min(height, int(round(height * y1_frac))))
    x0 = max(0, min(width, int(round(width * x0_frac))))
    x1 = max(x0 + 1, min(width, int(round(width * x1_frac))))
    return y0, y1, x0, x1


def classify_terrain_from_mask(
    mask_path: str,
    label_map: Dict[str, Any],
    ground_roi: Tuple[float, float, float, float] = (0.45, 1.0, 0.15, 0.85),
    default: str = "dirt_trail",
) -> Dict[str, Any]:
    """Classify dominant terrain from the lower-center ground ROI of a mask.

    The returned distribution is expressed over the five Outdoor-VLN terrain
    classes. Unknown/background pixels are ignored in the named distribution,
    so the ratios may sum to less than one if a mask contains large ignored
    regions.
    """
    mask = read_semantic_mask(mask_path)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    y0, y1, x0, x1 = _safe_roi_bounds(height, width, ground_roi)
    roi_mask = mask[y0:y1, x0:x1]
    parsed = parse_mask_labels(roi_mask, label_map)
    distribution = _empty_distribution()
    for label in parsed["labels"]:
        group = str(label.get("outdoor_group", "unknown"))
        if group in distribution:
            distribution[group] += float(label.get("area_ratio", 0.0))
    dominant = max(distribution, key=distribution.get)
    confidence = float(distribution[dominant])
    if confidence <= 0.0:
        dominant = default if default in VALID_TERRAINS else "dirt_trail"
    return {
        "dominant_terrain": dominant,
        "terrain_distribution": distribution,
        "confidence": confidence,
    }


def classify_terrain_from_segment_masks(
    frames: List[Any],
    start_idx: int,
    end_idx: int,
    label_map: Dict[str, Any],
    default: str = "dirt_trail",
) -> str:
    """Classify terrain by averaging mask distributions over a segment."""
    accumulated = _empty_distribution()
    used_masks = 0
    for frame in frames[start_idx : end_idx + 1]:
        semantic_path = getattr(frame, "semantic_path", None)
        if not semantic_path or not Path(str(semantic_path)).is_file():
            continue
        result = classify_terrain_from_mask(
            str(semantic_path), label_map=label_map, default=default
        )
        distribution = result.get("terrain_distribution", {})
        for terrain in accumulated:
            accumulated[terrain] += float(distribution.get(terrain, 0.0))
        used_masks += 1

    fallback = default if default in VALID_TERRAINS else "dirt_trail"
    if used_masks == 0:
        return fallback
    averaged = {
        terrain: value / used_masks for terrain, value in accumulated.items()
    }
    dominant = max(averaged, key=averaged.get)
    if averaged[dominant] <= 0.0:
        return fallback
    return dominant
