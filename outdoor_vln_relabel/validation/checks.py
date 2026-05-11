"""Basic validation checks for generated instruction-path pairs."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set

from outdoor_vln_relabel.perception.terrain import VALID_TERRAINS


def _get(pair: Any, field: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or dictionary."""
    if isinstance(pair, dict):
        return pair.get(field, default)
    return getattr(pair, field, default)


def _landmark_get(landmark: Any, field: str, default: Any = None) -> Any:
    """Read a field from either a Landmark object or dictionary."""
    if isinstance(landmark, dict):
        return landmark.get(field, default)
    return getattr(landmark, field, default)


def _landmarks(pair: Any) -> List[Any]:
    """Return pair landmarks as a list."""
    landmarks = _get(pair, "landmarks", [])
    if landmarks is None:
        return []
    if isinstance(landmarks, list):
        return landmarks
    return list(landmarks)


def _is_landmark_pair(pair: Any) -> bool:
    """Return True when validation should enforce landmark-aware checks."""
    version = str(_get(pair, "version", ""))
    return version.startswith("landmark_") or version.startswith("evidence_landmark_")


def check_trajectory_non_empty(pair: Any) -> bool:
    """Return True when the pair contains a non-empty trajectory_xy list."""
    trajectory = _get(pair, "trajectory_xy", [])
    return isinstance(trajectory, list) and len(trajectory) > 0


def check_instruction_non_empty(pair: Any) -> bool:
    """Return True when the pair contains non-empty instruction text."""
    instruction = _get(pair, "instruction", "")
    return isinstance(instruction, str) and bool(instruction.strip())


def check_motion_instruction_consistency(pair: Any) -> bool:
    """Return True when instruction keywords are compatible with the motion."""
    motion = _get(pair, "motion", "")
    instruction = str(_get(pair, "instruction", "")).lower()
    forward_words = (
        "forward",
        "straight",
        "ahead",
        "continue",
        "follow",
        "drive",
        "move",
        "toward",
    )
    if motion == "turn_left":
        return "left" in instruction
    if motion == "turn_right":
        return "right" in instruction
    if motion == "forward_left":
        return any(word in instruction for word in forward_words) and "left" in instruction
    if motion == "forward_right":
        return any(word in instruction for word in forward_words) and "right" in instruction
    if motion == "forward":
        return any(word in instruction for word in forward_words)
    return False


def check_terrain_valid(
    pair: Any, allowed_terrains: Optional[Iterable[str]] = None
) -> bool:
    """Return True when the pair terrain is absent or belongs to the taxonomy."""
    terrain = _get(pair, "terrain", None)
    if terrain is None:
        return True
    allowed: Set[str] = set(allowed_terrains or VALID_TERRAINS)
    return str(terrain) in allowed


def _contains_any(text: str, words: Iterable[str]) -> bool:
    """Return True when any keyword appears in text."""
    return any(word in text for word in words)


def check_terrain_instruction_consistency(pair: Any) -> bool:
    """Return True when terrain wording is compatible with the terrain label."""
    terrain = _get(pair, "terrain", None)
    if terrain is None:
        return True
    instruction = str(_get(pair, "instruction", "")).lower()
    if terrain == "dirt_trail":
        return _contains_any(
            instruction, ("trail", "path", "road", "dirt", "unpaved")
        )
    if terrain == "grass":
        return _contains_any(
            instruction, ("grass", "grassy", "meadow", "field")
        )
    if terrain == "vegetation":
        return _contains_any(
            instruction,
            ("avoid", "stay clear", "carefully", "vegetation", "bush", "shrub"),
        )
    if terrain == "mud_water":
        bad_follow = (
            "follow the puddle",
            "follow puddles",
            "follow the water",
            "follow water",
        )
        if _contains_any(instruction, bad_follow):
            return False
        return _contains_any(
            instruction,
            (
                "avoid",
                "stay clear",
                "carefully",
                "without entering",
                "mud",
                "puddle",
                "wet ground",
            ),
        )
    if terrain == "rough_terrain":
        return _contains_any(
            instruction,
            ("rough", "uneven", "rock", "rocky", "gravel", "slope", "carefully"),
        )
    return False


def check_landmarks_non_empty(pair: Any) -> bool:
    """Return True when the pair has at least one landmark."""
    return len(_landmarks(pair)) > 0


def check_avoid_role_consistency(pair: Any) -> bool:
    """Return True when avoid wording and avoid-role landmarks agree."""
    instruction = str(_get(pair, "instruction", "")).lower()
    landmarks = _landmarks(pair)
    avoid_words = ("avoid", "stay clear", "keep away", "keeping away")
    has_avoid_wording = _contains_any(instruction, avoid_words)
    has_avoid_landmark = any(
        str(_landmark_get(landmark, "role", "")) == "avoid"
        for landmark in landmarks
    )
    if has_avoid_wording and not has_avoid_landmark:
        return False

    unsafe_follow = (
        "follow the puddle",
        "follow puddle",
        "follow the mud",
        "follow mud",
        "follow the water",
        "follow water",
    )
    if _contains_any(instruction, unsafe_follow):
        return False

    if _get(pair, "terrain", None) == "mud_water":
        unsafe_names = {"puddle", "mud", "water", "muddy area"}
        for landmark in landmarks:
            name = str(_landmark_get(landmark, "name", "")).lower()
            role = str(_landmark_get(landmark, "role", ""))
            if name in unsafe_names and role != "avoid":
                return False
    return True


def check_follow_role_consistency(pair: Any) -> bool:
    """Return True when follow wording does not target unsafe landmarks."""
    instruction = str(_get(pair, "instruction", "")).lower()
    if not _contains_any(instruction, ("follow", "continue along", "move toward")):
        return True
    unsafe_targets = (
        "follow the puddle",
        "follow the mud",
        "follow the water",
    )
    if _contains_any(instruction, unsafe_targets):
        return False
    landmarks = _landmarks(pair)
    if not landmarks:
        return True
    return any(
        str(_landmark_get(landmark, "role", ""))
        in {"follow", "go_toward", "pass_between", "turn_at"}
        for landmark in landmarks
    )


def check_landmark_instruction_consistency(pair: Any) -> bool:
    """Return True when landmark-aware wording is compatible with landmarks."""
    landmarks = _landmarks(pair)
    if not landmarks:
        return False
    instruction = str(_get(pair, "instruction", "")).lower()

    if "on the left" in instruction or "your left" in instruction:
        if not any(
            "left" in str(_landmark_get(landmark, "relation", ""))
            for landmark in landmarks
        ):
            return False
    if "on the right" in instruction or "your right" in instruction:
        if not any(
            "right" in str(_landmark_get(landmark, "relation", ""))
            for landmark in landmarks
        ):
            return False

    names = [str(_landmark_get(landmark, "name", "")).lower() for landmark in landmarks]
    if any(name and name in instruction for name in names):
        return True
    if _contains_any(
        instruction,
        (
            "bush",
            "shrub",
            "tree",
            "rock",
            "puddle",
            "mud",
            "water",
            "passage",
            "path",
            "trail",
            "vegetation",
            "dense vegetation",
            "sparse vegetation",
            "person",
            "pedestrian",
            "fence",
        ),
    ):
        return True
    return False


def validate_pair(pair: Any) -> bool:
    """Run Stage-1/2/3/4 validity checks for one instruction-path pair."""
    base_ok = (
        check_trajectory_non_empty(pair)
        and check_instruction_non_empty(pair)
        and check_motion_instruction_consistency(pair)
        and check_terrain_valid(pair)
        and check_terrain_instruction_consistency(pair)
    )
    if not base_ok:
        return False
    if not _is_landmark_pair(pair):
        return True
    return (
        check_landmarks_non_empty(pair)
        and check_landmark_instruction_consistency(pair)
        and check_avoid_role_consistency(pair)
        and check_follow_role_consistency(pair)
    )


def validate_pair_verbose(pair: Any) -> List[str]:
    """Return human-readable validation issue strings for one pair."""
    issues: List[str] = []
    if not check_trajectory_non_empty(pair):
        issues.append("empty_trajectory")
    if not check_instruction_non_empty(pair):
        issues.append("empty_instruction")
    if not check_motion_instruction_consistency(pair):
        issues.append("motion_instruction_inconsistent")
    if not check_terrain_valid(pair):
        issues.append("invalid_terrain")
    if not check_terrain_instruction_consistency(pair):
        issues.append("terrain_instruction_inconsistent")
    if _is_landmark_pair(pair):
        if not check_landmarks_non_empty(pair):
            issues.append("missing_landmarks_when_v2_or_v3")
        if not check_landmark_instruction_consistency(pair):
            issues.append("landmark_instruction_inconsistent")
        if not check_avoid_role_consistency(pair):
            issues.append("avoid_role_inconsistent")
        if not check_follow_role_consistency(pair):
            issues.append("follow_role_inconsistent")
    return issues
