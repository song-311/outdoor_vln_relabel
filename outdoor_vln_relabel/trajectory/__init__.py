"""Trajectory segmentation and motion-label utilities."""

from .segment import (
    classify_motion,
    path_length,
    relative_trajectory_xy,
    segment_trajectory,
    wrap_angle,
)

__all__ = [
    "classify_motion",
    "path_length",
    "relative_trajectory_xy",
    "segment_trajectory",
    "wrap_angle",
]

