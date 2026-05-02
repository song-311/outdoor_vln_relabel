"""Safe deterministic rewrite backend for Outdoor-VLN instructions."""

from __future__ import annotations

from itertools import islice
from typing import List, Optional

from .constraints import check_rewrite_constraints, extract_constraints


def _relation_phrase(relation: str) -> str:
    """Return natural wording for landmark relations."""
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
    return relation.replace("_", " ")


def _motion_phrase(motion: str) -> str:
    """Return a safe phrase for the existing motion."""
    phrases = {
        "forward": "Move forward",
        "forward_left": "Move forward and slightly left",
        "forward_right": "Move forward and slightly right",
        "turn_left": "Turn left",
        "turn_right": "Turn right",
    }
    return phrases.get(motion, "Move forward")


def _terrain_phrase(terrain: str) -> str:
    """Return natural terrain wording."""
    phrases = {
        "dirt_trail": "dirt trail",
        "grass": "grass",
        "vegetation": "vegetation",
        "mud_water": "muddy ground",
        "rough_terrain": "rough terrain",
    }
    return phrases.get(terrain, terrain.replace("_", " "))


def _first_by_role(constraints: dict, roles: set[str]) -> Optional[dict]:
    """Return the first landmark matching any role."""
    for landmark in constraints["landmarks"]:
        if landmark.get("role") in roles:
            return landmark
    return None


def _all_avoid(constraints: dict) -> List[dict]:
    """Return avoid-role landmarks."""
    return [
        landmark
        for landmark in constraints["landmarks"]
        if landmark.get("role") == "avoid"
    ]


def _goal_landmark(constraints: dict) -> Optional[dict]:
    """Return a goal-like landmark."""
    return _first_by_role(
        constraints, {"follow", "go_toward", "pass_between", "turn_at"}
    )


def _all_goals(constraints: dict) -> List[dict]:
    """Return all goal-like landmarks in their constrained order."""
    return [
        landmark
        for landmark in constraints["landmarks"]
        if landmark.get("role") in {"follow", "go_toward", "pass_between", "turn_at"}
    ]


def _goal_text(primary: dict, goals: List[dict]) -> str:
    """Return goal wording that preserves every goal-like landmark."""
    primary_name = primary["name"]
    extras = []
    seen = {primary_name.lower()}
    for goal in goals:
        name = str(goal.get("name", "")).strip()
        if name and name.lower() not in seen:
            extras.append(name)
            seen.add(name.lower())
    if not extras:
        return primary_name
    if len(extras) == 1:
        return f"{primary_name} toward the {extras[0]}"
    return f"{primary_name} toward the " + " and the ".join(extras)


def _avoid_phrase(obstacle: dict) -> str:
    """Return an avoid phrase for one obstacle landmark."""
    return f"the {obstacle['name']} {_relation_phrase(obstacle.get('relation', 'ahead'))}"


def _avoid_join(obstacles: List[dict]) -> str:
    """Join avoid landmarks into one readable phrase."""
    phrases = [_avoid_phrase(obstacle) for obstacle in obstacles]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _goal_avoid_templates(
    record: dict, goal: dict, goals: List[dict], obstacles: List[dict]
) -> List[str]:
    """Generate templates that preserve a goal and obstacle constraint."""
    motion = str(record.get("motion", ""))
    terrain = str(record.get("terrain", ""))
    prefix = _motion_phrase(motion)
    goal_name = _goal_text(goal, goals)
    avoid_text = _avoid_join(obstacles)

    if terrain == "mud_water":
        return [
            f"{prefix} along the {goal_name} while staying clear of {avoid_text}.",
            f"{prefix} on the {goal_name} and avoid {avoid_text}.",
            f"{prefix} while following the {goal_name}, staying away from {avoid_text}.",
        ]
    if terrain == "dirt_trail":
        return [
            f"Follow the {goal_name} and stay clear of {avoid_text}.",
            f"Continue along the {goal_name}, keeping away from {avoid_text}.",
            f"{prefix} on the {goal_name} while avoiding {avoid_text}.",
        ]
    if terrain == "vegetation":
        return [
            f"{prefix} through the {goal_name} and stay clear of {avoid_text}.",
            f"Continue through the {goal_name}, avoiding {avoid_text}.",
            f"Pass through the {goal_name} while keeping away from {avoid_text}.",
        ]
    return [
        f"{prefix} toward the {goal_name} while avoiding {avoid_text}.",
        f"Continue along the {goal_name}, keeping away from {avoid_text}.",
        f"{prefix} across the {_terrain_phrase(terrain)} and stay clear of {avoid_text}.",
    ]


def _goal_only_templates(record: dict, goal: dict, goals: List[dict]) -> List[str]:
    """Generate templates for records with no avoid landmark."""
    motion = str(record.get("motion", ""))
    prefix = _motion_phrase(motion)
    goal_name = _goal_text(goal, goals)
    role = goal.get("role")
    if role == "pass_between":
        return [
            f"{prefix} through the {goal_name}.",
            f"Continue through the {goal_name} ahead.",
            f"Pass through the {goal_name} and keep moving forward.",
        ]
    if role == "turn_at":
        return [
            f"{prefix} near the {goal_name}.",
            f"{prefix} at the {goal_name}.",
            f"{prefix} after passing the {goal_name}.",
        ]
    return [
        f"Continue along the {goal_name}.",
        f"{prefix} toward the {goal_name}.",
        f"Keep following the {goal_name} ahead.",
    ]


def _fallback_templates(record: dict) -> List[str]:
    """Generate conservative fallback rewrites from existing fields."""
    instruction = str(record.get("instruction", "")).strip()
    terrain = _terrain_phrase(str(record.get("terrain", "")))
    prefix = _motion_phrase(str(record.get("motion", "")))
    return [
        instruction,
        f"{prefix} across the {terrain}.",
        f"{prefix} and continue ahead through the {terrain}.",
    ]


def _dedupe(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key not in seen:
            output.append(item)
            seen.add(key)
    return output


def rule_based_rewrite(record: dict, num_variants: int = 3) -> List[str]:
    """Generate safe deterministic rewrite variants for one record."""
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")
    constraints = extract_constraints(record)
    goal = _goal_landmark(constraints)
    goals = _all_goals(constraints)
    avoid_landmarks = _all_avoid(constraints)

    if goal and avoid_landmarks:
        candidates = _goal_avoid_templates(record, goal, goals, avoid_landmarks)
    elif goal:
        candidates = _goal_only_templates(record, goal, goals)
    else:
        candidates = _fallback_templates(record)

    # Add the original instruction as a late fallback, but still validate it.
    original = str(record.get("instruction", "")).strip()
    if original:
        candidates.append(original)

    valid = [
        candidate
        for candidate in _dedupe(candidates)
        if not check_rewrite_constraints(record, candidate)
    ]
    return list(islice(valid, num_variants))
