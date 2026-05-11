"""Deterministic instruction templates for motion and terrain-aware samples."""

from __future__ import annotations

from itertools import islice
from typing import Dict, List, Optional

from outdoor_vln_relabel.schemas import Landmark, StructuredLabel

MOTION_TEMPLATES: Dict[str, List[str]] = {
    "forward": [
        "Move forward along the path.",
        "Continue straight ahead.",
        "Drive forward toward the area ahead.",
    ],
    "forward_left": [
        "Move forward and slightly left.",
        "Continue ahead while keeping slightly left.",
        "Follow the path forward with a slight left turn.",
    ],
    "forward_right": [
        "Move forward and slightly right.",
        "Continue ahead while keeping slightly right.",
        "Follow the path forward with a slight right turn.",
    ],
    "turn_left": [
        "Turn left and continue forward.",
        "Make a left turn along the path.",
        "Head left toward the next part of the route.",
    ],
    "turn_right": [
        "Turn right and continue forward.",
        "Make a right turn along the path.",
        "Head right toward the next part of the route.",
    ],
}

TERRAIN_MOTION_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "dirt_trail": {
        "forward": [
            "Follow the dirt trail ahead.",
            "Continue forward along the dirt path.",
            "Drive straight along the unpaved trail.",
        ],
        "forward_left": [
            "Follow the dirt trail while keeping slightly left.",
            "Continue along the trail with a slight left turn.",
            "Move forward on the dirt path and bear slightly left.",
        ],
        "forward_right": [
            "Follow the dirt trail while keeping slightly right.",
            "Continue along the trail with a slight right turn.",
            "Move forward on the dirt path and bear slightly right.",
        ],
        "turn_left": [
            "Turn left and follow the dirt trail.",
            "Make a left turn along the dirt path.",
            "Head left onto the next part of the unpaved trail.",
        ],
        "turn_right": [
            "Turn right and follow the dirt trail.",
            "Make a right turn along the dirt path.",
            "Head right onto the next part of the unpaved trail.",
        ],
    },
    "grass": {
        "forward": [
            "Move forward across the grass.",
            "Continue through the grassy area ahead.",
            "Drive straight across the grass field.",
        ],
        "forward_left": [
            "Move forward across the grass while keeping slightly left.",
            "Continue through the grassy area with a slight left turn.",
            "Drive across the grass field and bear slightly left.",
        ],
        "forward_right": [
            "Move forward across the grass while keeping slightly right.",
            "Continue through the grassy area with a slight right turn.",
            "Drive across the grass field and bear slightly right.",
        ],
        "turn_left": [
            "Turn left across the grass.",
            "Make a left turn through the grassy area.",
            "Head left across the grass field.",
        ],
        "turn_right": [
            "Turn right across the grass.",
            "Make a right turn through the grassy area.",
            "Head right across the grass field.",
        ],
    },
    "vegetation": {
        "forward": [
            "Move forward while avoiding dense vegetation.",
            "Continue ahead and stay clear of the bushes.",
            "Drive carefully through the sparse vegetation.",
        ],
        "forward_left": [
            "Move forward and slightly left while avoiding dense vegetation.",
            "Continue ahead with a slight left turn and stay clear of the bushes.",
            "Drive carefully through the vegetation while bearing slightly left.",
        ],
        "forward_right": [
            "Move forward and slightly right while avoiding dense vegetation.",
            "Continue ahead with a slight right turn and stay clear of the bushes.",
            "Drive carefully through the vegetation while bearing slightly right.",
        ],
        "turn_left": [
            "Turn left while avoiding dense vegetation.",
            "Make a left turn and stay clear of the bushes.",
            "Head left carefully through the sparse vegetation.",
        ],
        "turn_right": [
            "Turn right while avoiding dense vegetation.",
            "Make a right turn and stay clear of the bushes.",
            "Head right carefully through the sparse vegetation.",
        ],
    },
    "mud_water": {
        "forward": [
            "Move forward while avoiding the muddy area.",
            "Continue ahead and stay clear of puddles.",
            "Drive forward without entering the wet ground.",
        ],
        "forward_left": [
            "Move forward and slightly left while avoiding the muddy area.",
            "Continue ahead with a slight left turn and stay clear of puddles.",
            "Drive forward-left without entering the wet ground.",
        ],
        "forward_right": [
            "Move forward and slightly right while avoiding the muddy area.",
            "Continue ahead with a slight right turn and stay clear of puddles.",
            "Drive forward-right without entering the wet ground.",
        ],
        "turn_left": [
            "Turn left while avoiding the muddy area.",
            "Make a left turn and stay clear of puddles.",
            "Head left carefully without entering the wet ground.",
        ],
        "turn_right": [
            "Turn right while avoiding the muddy area.",
            "Make a right turn and stay clear of puddles.",
            "Head right carefully without entering the wet ground.",
        ],
    },
    "rough_terrain": {
        "forward": [
            "Move forward carefully over the rough terrain.",
            "Continue ahead across the uneven ground.",
            "Drive slowly along the rocky path.",
        ],
        "forward_left": [
            "Move forward and slightly left over the rough terrain.",
            "Continue across the uneven ground with a slight left turn.",
            "Drive slowly along the rocky path while bearing slightly left.",
        ],
        "forward_right": [
            "Move forward and slightly right over the rough terrain.",
            "Continue across the uneven ground with a slight right turn.",
            "Drive slowly along the rocky path while bearing slightly right.",
        ],
        "turn_left": [
            "Turn left carefully over the rough terrain.",
            "Make a left turn across the uneven ground.",
            "Head left along the rocky path.",
        ],
        "turn_right": [
            "Turn right carefully over the rough terrain.",
            "Make a right turn across the uneven ground.",
            "Head right along the rocky path.",
        ],
    },
}


def _repeat_templates(templates: List[str]):
    """Yield templates in order, cycling when more variants are requested."""
    while True:
        for template in templates:
            yield template


def generate_motion_instructions(motion: str, num_variants: int = 3) -> List[str]:
    """Generate deterministic English motion instructions for a motion label."""
    if motion not in MOTION_TEMPLATES:
        valid = ", ".join(sorted(MOTION_TEMPLATES))
        raise ValueError(f"Unknown motion '{motion}'. Expected one of: {valid}")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")
    return list(islice(_repeat_templates(MOTION_TEMPLATES[motion]), num_variants))


def generate_terrain_motion_instructions(
    motion: str, terrain: str, num_variants: int = 3
) -> List[str]:
    """Generate deterministic terrain-aware instructions for a motion label."""
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")
    terrain_templates = TERRAIN_MOTION_TEMPLATES.get(terrain, {})
    templates = terrain_templates.get(motion)
    if not templates:
        return generate_motion_instructions(motion, num_variants=num_variants)
    return list(islice(_repeat_templates(templates), num_variants))


TERRAIN_PHRASES: Dict[str, str] = {
    "dirt_trail": "dirt trail",
    "grass": "grass",
    "vegetation": "vegetation",
    "mud_water": "muddy ground",
    "rough_terrain": "rough terrain",
}


def _relation_phrase(relation: str) -> str:
    """Convert a relation label to natural instruction wording."""
    phrases = {
        "left": "left",
        "right": "right",
        "ahead": "ahead",
        "front_left": "front-left",
        "front_right": "front-right",
        "near": "nearby",
    }
    return phrases.get(relation, relation.replace("_", " "))


def _avoid_location_phrase(relation: str) -> str:
    """Return natural location wording for avoid constraints."""
    if relation == "left":
        return "on your left"
    if relation == "right":
        return "on your right"
    if relation == "ahead":
        return "ahead"
    if relation == "front_left":
        return "ahead on your front-left"
    if relation == "front_right":
        return "ahead on your front-right"
    if relation == "near":
        return "nearby"
    return _relation_phrase(relation)


def _motion_prefix(motion: str) -> str:
    """Return a direction-aware prefix for landmark instructions."""
    if motion == "turn_left":
        return "Turn left"
    if motion == "turn_right":
        return "Turn right"
    if motion == "forward_left":
        return "Move forward and slightly left"
    if motion == "forward_right":
        return "Move forward and slightly right"
    return "Move forward"


def _primary_constraint(
    label: StructuredLabel, terrain: Optional[str] = None
) -> Optional[Landmark]:
    """Return the most terrain-relevant avoid landmark.

    For mud/water terrain, prefer puddle/mud/water hazards over generic
    obstacles such as trees or bushes. This keeps generated instructions
    consistent with the terrain label.
    """
    avoid_landmarks = [
        landmark
        for landmark in label.constraint_landmarks
        if landmark.role == "avoid"
    ]
    if not avoid_landmarks:
        return None

    terrain_hazards = {
        "mud_water": {"puddle", "mud", "water", "muddy area", "wet ground"},
        "vegetation": {"bush", "bushes", "shrub", "tree", "vegetation"},
        "rough_terrain": {"rock", "rocks", "rubble", "barrier", "log", "fence"},
    }

    preferred_names = terrain_hazards.get(str(terrain or ""))
    if preferred_names:
        preferred = [
            landmark
            for landmark in avoid_landmarks
            if str(landmark.name).lower() in preferred_names
        ]
        if preferred:
            return max(preferred, key=lambda landmark: float(landmark.score))

    return max(avoid_landmarks, key=lambda landmark: float(landmark.score))


def _format_goal_only(motion: str, goal: Landmark) -> List[str]:
    """Build templates for a single goal landmark."""
    goal_name = goal.name
    if goal.role == "turn_at" and motion in {"turn_left", "forward_left"}:
        return [
            f"Turn left near the {goal_name}.",
            f"Make a left turn at the {goal_name}.",
            f"Bear left after passing the {goal_name}.",
        ]
    if goal.role == "turn_at" and motion in {"turn_right", "forward_right"}:
        return [
            f"Turn right near the {goal_name}.",
            f"Make a right turn at the {goal_name}.",
            f"Bear right after passing the {goal_name}.",
        ]
    if goal.role == "pass_between":
        return [
            f"Pass through the {goal_name}.",
            f"Move through the {goal_name} ahead.",
            "Continue through the narrow passage.",
        ]
    prefix = _motion_prefix(motion)
    if motion in {"turn_left", "turn_right", "forward_left", "forward_right"}:
        return [
            f"{prefix} toward the {goal_name}.",
            f"{prefix} and continue along the {goal_name}.",
            f"{prefix} with the {goal_name} ahead.",
        ]
    return [
        f"Follow the {goal_name} ahead.",
        f"Move toward the {goal_name}.",
        f"Continue along the {goal_name}.",
    ]


def _format_goal_avoid(
    motion: str, terrain: str, goal: Landmark, obstacle: Landmark
) -> List[str]:
    """Build templates that mention a goal and an avoid constraint."""
    goal_name = goal.name
    obstacle_name = obstacle.name
    location = _avoid_location_phrase(obstacle.relation)
    prefix = _motion_prefix(motion)
    terrain_phrase = TERRAIN_PHRASES.get(terrain, terrain.replace("_", " "))

    if motion in {"turn_left", "turn_right", "forward_left", "forward_right"}:
        return [
            f"{prefix} along the {goal_name} and avoid the {obstacle_name} {location}.",
            f"{prefix} toward the {goal_name} while staying clear of the {obstacle_name} {location}.",
            f"{prefix} across the {terrain_phrase}, keeping away from the {obstacle_name} {location}.",
        ]
    return [
        f"Follow the {goal_name} and avoid the {obstacle_name} {location}.",
        f"Move toward the {goal_name} while staying clear of the {obstacle_name} {location}.",
        f"Continue along the {goal_name}, keeping away from the {obstacle_name} {location}.",
    ]


def generate_landmark_instructions(
    motion: str,
    terrain: str,
    structured_label: StructuredLabel,
    num_variants: int = 3,
) -> List[str]:
    """Generate landmark-aware instructions with terrain/motion fallbacks."""
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")

    goal = structured_label.goal_landmark
    if goal is None:
        return generate_terrain_motion_instructions(
            motion, terrain, num_variants=num_variants
        )

    obstacle = _primary_constraint(structured_label, terrain=terrain)
    if obstacle is not None:
        templates = _format_goal_avoid(motion, terrain, goal, obstacle)
    else:
        templates = _format_goal_only(motion, goal)

    if not templates:
        return generate_terrain_motion_instructions(
            motion, terrain, num_variants=num_variants
        )
    return list(islice(_repeat_templates(templates), num_variants))
