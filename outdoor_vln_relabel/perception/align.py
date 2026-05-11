"""Physical alignment placeholders for landmarks and path segments."""

from __future__ import annotations

from typing import Iterable, List, Optional

from outdoor_vln_relabel.schemas import Landmark, PathSegment, StructuredLabel


GOAL_ROLE_PRIORITY = {
    "follow": 0,
    "go_toward": 1,
    "pass_between": 2,
}


def _infer_role(landmark: Landmark, terrain: str) -> str:
    """Infer a safe role when metadata did not provide one."""
    role = str(landmark.role or "").strip()
    name = str(landmark.name or "").lower()
    category = str(landmark.category or "")
    if terrain == "mud_water" and name in {"mud", "puddle", "water", "muddy area"}:
        return "avoid"
    if role:
        return role
    if category == "obstacle_like":
        return "avoid"
    if category == "path_like":
        return "follow"
    if category == "geometry_like" and ("narrow" in name or "gap" in name):
        return "pass_between"
    return "go_toward"


def _normalize_landmark(landmark: Landmark, terrain: str) -> Landmark:
    """Fill missing or unsafe landmark role/relation fields."""
    role = _infer_role(landmark, terrain)
    relation = str(landmark.relation or "").strip() or "ahead"
    return Landmark(
        name=landmark.name,
        category=landmark.category or "unknown",
        role=role,
        relation=relation,
        bbox=landmark.bbox,
        score=float(landmark.score),
    )


def _select_goal_landmark(landmarks: List[Landmark]) -> Optional[Landmark]:
    """Select the most likely goal landmark by role priority and score.

    Only followable / goal-like landmarks can be selected as goals.
    Avoid-only landmarks such as tree, bush, rock, mud, and puddle must not
    become navigation goals.
    """
    candidates = [
        landmark
        for landmark in landmarks
        if landmark.role in GOAL_ROLE_PRIORITY
    ]
    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda landmark: (
            GOAL_ROLE_PRIORITY.get(landmark.role, 3),
            -float(landmark.score),
        ),
    )
    return ranked[0]

def assign_landmark_roles(
    landmarks: Iterable[Landmark],
    segment: PathSegment,
    terrain: str = "dirt_trail",
) -> StructuredLabel:
    """Assign landmark roles and return a structured path label.

    Future versions will use trajectory geometry, image detections, depth, and
    LiDAR to decide follow, avoid, turn_at, pass_between, stop_near, and
    go_toward roles. Stage 3 uses the rule-generated landmark roles directly.
    """
    landmark_list: List[Landmark] = [
        _normalize_landmark(landmark, terrain) for landmark in landmarks
    ]
    goal_landmark = _select_goal_landmark(landmark_list)
    constraints = [
        landmark
        for landmark in landmark_list
        if landmark.role == "avoid" and landmark is not goal_landmark
    ]
    if landmark_list:
        confidence = sum(float(landmark.score) for landmark in landmark_list) / len(
            landmark_list
        )
    else:
        confidence = 0.5
    return StructuredLabel(
        terrain=terrain,
        motion=segment.motion,
        goal_landmark=goal_landmark,
        constraint_landmarks=constraints,
        confidence=round(confidence, 6),
    )
