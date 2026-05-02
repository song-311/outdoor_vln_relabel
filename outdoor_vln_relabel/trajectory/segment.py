"""Trajectory slicing utilities for motion-only Outdoor-VLN pairs."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from outdoor_vln_relabel.schemas import PathSegment

DEFAULT_SEGMENT_CONFIG: Dict[str, float] = {
    "min_dist": 3.0,
    "max_dist": 12.0,
    "min_duration": 4.0,
    "max_duration": 15.0,
    "max_heading_change_deg": 70.0,
}


def wrap_angle(angle: float) -> float:
    """Wrap a radian angle to the [-pi, pi) interval."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _as_xy_list(positions: Sequence[Sequence[float]]) -> List[List[float]]:
    """Convert position-like input into a validated list of [x, y] floats."""
    xy: List[List[float]] = []
    for idx, position in enumerate(positions):
        if len(position) < 2:
            raise ValueError(f"positions[{idx}] must contain at least x and y values")
        xy.append([float(position[0]), float(position[1])])
    return xy


def path_length(positions: Sequence[Sequence[float]]) -> float:
    """Compute cumulative 2D path length in meters."""
    xy = _as_xy_list(positions)
    if len(xy) < 2:
        return 0.0
    total = 0.0
    for prev, curr in zip(xy[:-1], xy[1:]):
        total += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
    return float(total)


def relative_trajectory_xy(
    positions: Sequence[Sequence[float]], start_idx: int, end_idx: int
) -> List[List[float]]:
    """Return a start-relative [x, y] trajectory for an inclusive index range."""
    xy = _as_xy_list(positions)
    if start_idx < 0 or end_idx >= len(xy) or start_idx > end_idx:
        raise ValueError(
            "Invalid trajectory range: "
            f"start_idx={start_idx}, end_idx={end_idx}, num_positions={len(xy)}"
        )
    origin_x, origin_y = xy[start_idx]
    return [
        [round(x - origin_x, 6), round(y - origin_y, 6)]
        for x, y in xy[start_idx : end_idx + 1]
    ]


def classify_motion(heading_change_deg: float) -> str:
    """Classify a segment into one of five coarse motion types."""
    heading_change_deg = float(heading_change_deg)
    if heading_change_deg > 20.0:
        return "turn_left"
    if 8.0 < heading_change_deg <= 20.0:
        return "forward_left"
    if heading_change_deg < -20.0:
        return "turn_right"
    if -20.0 <= heading_change_deg < -8.0:
        return "forward_right"
    return "forward"


def _merged_segment_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge user config with segment defaults."""
    merged: Dict[str, Any] = dict(DEFAULT_SEGMENT_CONFIG)
    if not config:
        return merged
    segment_config = config.get("segment", config)
    for key in DEFAULT_SEGMENT_CONFIG:
        if key in segment_config:
            merged[key] = float(segment_config[key])
    return merged


def _keyframes(start_idx: int, end_idx: int) -> List[int]:
    """Select stable start/middle/end keyframes for a segment."""
    mid_idx = (start_idx + end_idx) // 2
    frames = [start_idx, mid_idx, end_idx]
    deduped: List[int] = []
    for frame in frames:
        if frame not in deduped:
            deduped.append(frame)
    return deduped


def _validate_arrays(
    positions: Sequence[Sequence[float]],
    yaws: Sequence[float],
    timestamps: Sequence[float],
) -> None:
    """Validate trajectory arrays before segmentation."""
    num_positions = len(positions)
    if num_positions == 0:
        raise ValueError("positions must not be empty")
    if len(yaws) != num_positions:
        raise ValueError(
            f"yaws length ({len(yaws)}) must match positions length ({num_positions})"
        )
    if len(timestamps) != num_positions:
        raise ValueError(
            "timestamps length "
            f"({len(timestamps)}) must match positions length ({num_positions})"
        )


def segment_trajectory(
    positions: Sequence[Sequence[float]],
    yaws: Sequence[float],
    timestamps: Sequence[float],
    config: Dict[str, Any] | None = None,
) -> List[PathSegment]:
    """Split a full trajectory into local PathSegment records.

    The default policy greedily creates non-overlapping local segments once both
    the minimum distance and minimum duration are satisfied, while enforcing
    maximum distance, duration, and heading-change limits.
    """
    _validate_arrays(positions, yaws, timestamps)
    xy = _as_xy_list(positions)
    yaws_float = [float(yaw) for yaw in yaws]
    timestamps_float = [float(timestamp) for timestamp in timestamps]
    cfg = _merged_segment_config(config)

    scene_id = str((config or {}).get("scene_id", "unknown_scene"))
    sequence_id = str((config or {}).get("sequence_id", "default_sequence"))

    segments: List[PathSegment] = []
    start_idx = 0
    segment_count = 0
    num_positions = len(xy)

    while start_idx < num_positions - 1:
        chosen: Dict[str, Any] | None = None
        last_viable_end = None

        for end_idx in range(start_idx + 1, num_positions):
            segment_positions = xy[start_idx : end_idx + 1]
            distance_m = path_length(segment_positions)
            duration_s = timestamps_float[end_idx] - timestamps_float[start_idx]

            if distance_m <= cfg["max_dist"] and duration_s <= cfg["max_duration"]:
                last_viable_end = end_idx

            if distance_m > cfg["max_dist"] or duration_s > cfg["max_duration"]:
                break

            if distance_m < cfg["min_dist"] or duration_s < cfg["min_duration"]:
                continue

            heading_change_rad = wrap_angle(yaws_float[end_idx] - yaws_float[start_idx])
            heading_change_deg = math.degrees(heading_change_rad)
            if abs(heading_change_deg) > cfg["max_heading_change_deg"]:
                break

            chosen = {
                "end_idx": end_idx,
                "distance_m": distance_m,
                "duration_s": duration_s,
                "heading_change_deg": heading_change_deg,
            }

        if chosen is not None:
            end_idx = int(chosen["end_idx"])
            segment_count += 1
            segment_id = f"{sequence_id}_{segment_count:06d}"
            segments.append(
                PathSegment(
                    scene_id=scene_id,
                    sequence_id=sequence_id,
                    segment_id=segment_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    keyframe_indices=_keyframes(start_idx, end_idx),
                    distance_m=round(float(chosen["distance_m"]), 6),
                    duration_s=round(float(chosen["duration_s"]), 6),
                    heading_change_deg=round(
                        float(chosen["heading_change_deg"]), 6
                    ),
                    motion=classify_motion(float(chosen["heading_change_deg"])),
                    trajectory_xy=relative_trajectory_xy(xy, start_idx, end_idx),
                )
            )
            start_idx = end_idx
        else:
            if last_viable_end is not None and last_viable_end > start_idx:
                start_idx = last_viable_end
            else:
                start_idx += 1

    return segments
