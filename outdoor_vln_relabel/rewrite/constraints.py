"""Hard constraints for safe Outdoor-VLN instruction rewrites."""

from __future__ import annotations

import re
from typing import Any, Dict, List

AVOID_WORDS = (
    "avoid",
    "avoiding",
    "stay clear",
    "staying clear",
    "keep away",
    "keeping away",
    "steer clear",
    "stay away",
    "staying away",
)

FORBIDDEN_FOLLOW_LANDMARKS = ["mud", "water", "puddle", "muddy area"]


def _landmarks(record: dict) -> List[dict]:
    """Return landmark dictionaries from a record."""
    landmarks = record.get("landmarks") or []
    return [landmark for landmark in landmarks if isinstance(landmark, dict)]


def _norm(text: Any) -> str:
    """Normalize text for matching."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _contains_any(text: str, phrases) -> bool:
    """Return True when text contains any phrase."""
    return any(phrase in text for phrase in phrases)


def _simple_name_variants(name: str) -> List[str]:
    """Return simple accepted textual variants for one landmark name."""
    cleaned = _norm(name)
    variants = {cleaned}
    if cleaned.endswith("s"):
        variants.add(cleaned[:-1])
    else:
        variants.add(cleaned + "s")
    replacements = {
        "bush": ["bushes"],
        "bushes": ["bush"],
        "dirt trail": ["trail", "dirt path"],
        "open path": ["path"],
        "puddle": ["puddles", "wet ground"],
        "muddy area": ["mud", "wet ground"],
        "narrow passage": ["passage", "gap", "narrow opening"],
        "rock": ["rocks"],
        "rocks": ["rock", "rocky area"],
    }
    variants.update(replacements.get(cleaned, []))
    return sorted(variant for variant in variants if variant)


def extract_constraints(record: dict) -> dict:
    """Extract rewrite constraints from one instruction-path record."""
    landmarks = []
    must_keep_roles: Dict[str, str] = {}
    must_keep_relations: Dict[str, str] = {}
    avoid_landmarks = []
    goal_landmarks = []

    for landmark in _landmarks(record):
        name = str(landmark.get("name", "")).strip()
        if not name:
            continue
        role = str(landmark.get("role", "")).strip()
        relation = str(landmark.get("relation", "")).strip()
        item = {"name": name, "role": role, "relation": relation}
        landmarks.append(item)
        must_keep_roles[name] = role
        must_keep_relations[name] = relation
        if role == "avoid":
            avoid_landmarks.append(name)
        if role in {"follow", "go_toward", "pass_between", "turn_at"}:
            goal_landmarks.append(name)

    return {
        "terrain": record.get("terrain"),
        "motion": record.get("motion"),
        "landmarks": landmarks,
        "must_keep_landmarks": [item["name"] for item in landmarks],
        "must_keep_roles": must_keep_roles,
        "must_keep_relations": must_keep_relations,
        "avoid_landmarks": avoid_landmarks,
        "goal_landmarks": goal_landmarks,
        "forbidden_follow_landmarks": list(FORBIDDEN_FOLLOW_LANDMARKS),
        "forbidden_phrases": [
            f"follow the {name}" for name in FORBIDDEN_FOLLOW_LANDMARKS
        ]
        + [f"follow {name}" for name in FORBIDDEN_FOLLOW_LANDMARKS],
    }


def _has_name_variant(text: str, name: str) -> bool:
    """Return True if text contains a landmark name or simple variant."""
    return any(variant in text for variant in _simple_name_variants(name))


def _opposite_relation(relation: str) -> str:
    """Return opposite left/right relation when one exists."""
    if "left" in relation:
        return "right"
    if "right" in relation:
        return "left"
    return ""


def _contains_forbidden_follow(text: str, landmark_name: str) -> bool:
    """Return True if text says to follow a forbidden landmark."""
    for variant in _simple_name_variants(landmark_name):
        if f"follow the {variant}" in text or f"follow {variant}" in text:
            return True
    return False


def check_rewrite_constraints(
    original_record: dict, rewritten_instruction: str
) -> List[str]:
    """Return hard-constraint issues for a candidate rewritten instruction."""
    issues: List[str] = []
    text = _norm(rewritten_instruction)
    original_instruction = _norm(original_record.get("instruction", ""))
    constraints = extract_constraints(original_record)

    if not text:
        return ["empty_rewrite"]

    original_has_avoid = _contains_any(original_instruction, AVOID_WORDS)
    if original_has_avoid and not _contains_any(text, AVOID_WORDS):
        issues.append("missing_avoid_wording")

    for name, role in constraints["must_keep_roles"].items():
        name_norm = _norm(name)
        if role == "avoid" and name_norm in FORBIDDEN_FOLLOW_LANDMARKS:
            if _contains_forbidden_follow(text, name_norm):
                issues.append(f"forbidden_follow_landmark:{name}")

    for forbidden in FORBIDDEN_FOLLOW_LANDMARKS:
        if _contains_forbidden_follow(text, forbidden):
            issues.append(f"forbidden_follow_landmark:{forbidden}")

    for name, relation in constraints["must_keep_relations"].items():
        relation_norm = _norm(relation)
        opposite = _opposite_relation(relation_norm)
        if opposite and _has_name_variant(text, name):
            forbidden_patterns = (
                f"on your {opposite}",
                f"on the {opposite}",
                f"to your {opposite}",
                f"{opposite}-side",
            )
            if _contains_any(text, forbidden_patterns):
                issues.append(f"relation_changed:{name}:{relation}->{opposite}")

    motion = constraints.get("motion")
    if motion == "turn_left" and "turn right" in text:
        issues.append("motion_changed:turn_left_to_right")
    if motion == "turn_right" and "turn left" in text:
        issues.append("motion_changed:turn_right_to_left")
    if motion == "forward_left" and ("turn right" in text or "bear right" in text):
        issues.append("motion_changed:forward_left_to_right")
    if motion == "forward_right" and ("turn left" in text or "bear left" in text):
        issues.append("motion_changed:forward_right_to_left")

    if constraints.get("terrain") == "mud_water":
        for forbidden in FORBIDDEN_FOLLOW_LANDMARKS:
            if _contains_forbidden_follow(text, forbidden):
                issues.append(f"mud_water_follow_error:{forbidden}")

    for name in constraints["must_keep_landmarks"]:
        if not _has_name_variant(text, name):
            issues.append(f"missing_landmark:{name}")

    for avoid_name in constraints["avoid_landmarks"]:
        if not _has_name_variant(text, avoid_name):
            issues.append(f"missing_avoid_landmark:{avoid_name}")

    return issues
