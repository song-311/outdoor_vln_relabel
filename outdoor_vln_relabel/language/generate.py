"""Language generation entry points."""

from __future__ import annotations

from typing import List

from outdoor_vln_relabel.schemas import StructuredLabel

from .templates import (
    generate_landmark_instructions,
    generate_motion_instructions,
    generate_terrain_motion_instructions,
)


def generate_instructions_from_label(
    label: StructuredLabel, num_variants: int = 3
) -> List[str]:
    """Generate instructions from a structured label.

    Stage 2 uses terrain-aware templates when a terrain label is present.
    Landmark-aware text can be added here without changing the downstream JSONL
    schema.
    """
    if label.terrain and (
        label.goal_landmark is not None or label.constraint_landmarks
    ):
        return generate_landmark_instructions(
            label.motion,
            label.terrain,
            label,
            num_variants=num_variants,
        )
    if label.terrain:
        return generate_terrain_motion_instructions(
            label.motion, label.terrain, num_variants=num_variants
        )
    return generate_motion_instructions(label.motion, num_variants=num_variants)
