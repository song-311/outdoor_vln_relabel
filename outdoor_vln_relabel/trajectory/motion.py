"""Motion label helpers shared by generation and validation code."""

from __future__ import annotations

from typing import Set

from .segment import classify_motion

VALID_MOTIONS: Set[str] = {
    "forward",
    "forward_left",
    "forward_right",
    "turn_left",
    "turn_right",
}


def is_valid_motion(motion: str) -> bool:
    """Return True when a motion label belongs to the Stage-1 taxonomy."""
    return motion in VALID_MOTIONS


__all__ = ["VALID_MOTIONS", "classify_motion", "is_valid_motion"]

