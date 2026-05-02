"""Statistics and issue discovery for Outdoor-VLN JSONL datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List

from outdoor_vln_relabel.io_utils import load_jsonl


def load_pairs(jsonl_path: str) -> List[dict]:
    """Load instruction-path pairs from a JSONL file."""
    records = load_jsonl(jsonl_path)
    return [dict(record) for record in records]


def _counter_dict(values: Iterable[Any]) -> Dict[str, int]:
    """Return a sorted dictionary from non-empty values."""
    counter = Counter(str(value) for value in values if value not in (None, ""))
    return dict(sorted(counter.items()))


def _summary(values: List[float]) -> Dict[str, float | None]:
    """Return min/max/mean/median summary for numeric values."""
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
    }


def _instruction_length(instruction: Any) -> int:
    """Return word-count style instruction length."""
    if not isinstance(instruction, str):
        return 0
    return len(instruction.split())


def _landmarks(record: dict) -> List[dict]:
    """Return landmarks as dictionaries."""
    landmarks = record.get("landmarks") or []
    return [landmark for landmark in landmarks if isinstance(landmark, dict)]


def compute_basic_stats(records: List[dict]) -> dict:
    """Compute high-level dataset statistics from JSONL records."""
    instruction_lengths = [
        _instruction_length(record.get("instruction", "")) for record in records
    ]
    confidences = [
        float(record["confidence"])
        for record in records
        if record.get("confidence") is not None
    ]

    all_landmarks = [landmark for record in records for landmark in _landmarks(record)]
    landmark_name_counts = _counter_dict(
        landmark.get("name") for landmark in all_landmarks
    )
    landmark_role_counts = _counter_dict(
        landmark.get("role") for landmark in all_landmarks
    )
    landmark_relation_counts = _counter_dict(
        landmark.get("relation") for landmark in all_landmarks
    )

    return {
        "total_pairs": len(records),
        "unique_scenes": len({record.get("scene_id") for record in records}),
        "unique_segments": len({record.get("segment_id") for record in records}),
        "version_counts": _counter_dict(record.get("version") for record in records),
        "rewrite_source_counts": _counter_dict(
            record.get("rewrite_source") for record in records
        ),
        "rewrite_counts": {
            "original": sum(1 for record in records if "original_instruction" not in record),
            "rewritten": sum(1 for record in records if "original_instruction" in record),
        },
        "terrain_counts": _counter_dict(record.get("terrain") for record in records),
        "motion_counts": _counter_dict(record.get("motion") for record in records),
        "instruction_length": _summary(instruction_lengths),
        "landmark_counts": {
            "total_landmarks": len(all_landmarks),
            "avg_landmarks_per_pair": (
                round(len(all_landmarks) / len(records), 6) if records else 0.0
            ),
            "landmark_name_counts": landmark_name_counts,
            "landmark_role_counts": landmark_role_counts,
            "landmark_relation_counts": landmark_relation_counts,
        },
        "confidence": _summary(confidences),
    }


def _issue(record: dict, index: int, issue_type: str, message: str) -> dict:
    """Create a normalized issue dictionary."""
    return {
        "index": index,
        "scene_id": str(record.get("scene_id", "")),
        "segment_id": str(record.get("segment_id", "")),
        "issue_type": issue_type,
        "message": message,
        "instruction": str(record.get("instruction", "")),
    }


def _has_avoid_landmark(record: dict) -> bool:
    """Return True when a record contains an avoid-role landmark."""
    return any(landmark.get("role") == "avoid" for landmark in _landmarks(record))


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    """Return True when text contains any phrase."""
    return any(phrase in text for phrase in phrases)


def _landmark_relations(record: dict) -> set[str]:
    """Return landmark relation labels for a record."""
    return {
        str(landmark.get("relation", ""))
        for landmark in _landmarks(record)
        if landmark.get("relation")
    }


def find_potential_issues(records: List[dict]) -> List[dict]:
    """Find lightweight potential issues in generated instruction-path pairs."""
    issues: List[dict] = []
    for index, record in enumerate(records):
        instruction = str(record.get("instruction", ""))
        lower_instruction = instruction.lower()
        landmarks = _landmarks(record)
        version = str(record.get("version", ""))
        terrain = record.get("terrain")
        instruction_len = _instruction_length(instruction)

        if not instruction.strip():
            issues.append(_issue(record, index, "empty_instruction", "Instruction is empty."))
        if not record.get("trajectory_xy"):
            issues.append(_issue(record, index, "empty_trajectory", "trajectory_xy is empty."))
        if terrain in (None, ""):
            issues.append(_issue(record, index, "missing_terrain", "Terrain is missing."))
        if (
            version.startswith("landmark_") or version.startswith("evidence_landmark_")
        ) and not landmarks:
            issues.append(
                _issue(
                    record,
                    index,
                    "missing_landmarks_when_v2_or_v3",
                    "Landmark-aware record has no landmarks.",
                )
            )
        if _contains_any(lower_instruction, ("avoid", "stay clear", "keep away")):
            if not _has_avoid_landmark(record):
                issues.append(
                    _issue(
                        record,
                        index,
                        "avoid_instruction_without_avoid_landmark",
                        "Instruction contains avoid wording but no avoid landmark.",
                    )
                )
        if _contains_any(
            lower_instruction,
            (
                "follow the puddle",
                "follow puddle",
                "follow the mud",
                "follow mud",
                "follow the water",
                "follow water",
            ),
        ):
            issues.append(
                _issue(
                    record,
                    index,
                    "mud_water_follow_error",
                    "Instruction appears to follow mud, puddle, or water.",
                )
            )
        if terrain == "vegetation" and _contains_any(
            lower_instruction,
            ("follow the vegetation", "follow vegetation", "follow the bush"),
        ):
            issues.append(
                _issue(
                    record,
                    index,
                    "vegetation_follow_error",
                    "Instruction appears to follow vegetation or bushes.",
                )
            )
        relations = _landmark_relations(record)
        mentions_left_relation = (
            "on your left" in lower_instruction
            or "on the left" in lower_instruction
            or "to your left" in lower_instruction
        )
        mentions_right_relation = (
            "on your right" in lower_instruction
            or "on the right" in lower_instruction
            or "to your right" in lower_instruction
        )
        if (
            mentions_left_relation and not any("left" in rel for rel in relations)
        ) or (
            mentions_right_relation and not any("right" in rel for rel in relations)
        ):
            issues.append(
                _issue(
                    record,
                    index,
                    "left_right_mismatch_warning",
                    "Instruction mentions left/right but landmark relations do not match.",
                )
            )
        if 0 < instruction_len < 3:
            issues.append(
                _issue(
                    record,
                    index,
                    "very_short_instruction",
                    f"Instruction has only {instruction_len} words.",
                )
            )
        if instruction_len > 40:
            issues.append(
                _issue(
                    record,
                    index,
                    "very_long_instruction",
                    f"Instruction has {instruction_len} words.",
                )
            )
        confidence = record.get("confidence")
        if confidence is not None and float(confidence) < 0.6:
            issues.append(
                _issue(
                    record,
                    index,
                    "low_confidence",
                    f"Confidence is low: {float(confidence):.3f}.",
                )
            )
    return issues


def group_issues_by_type(issues: List[dict]) -> Dict[str, int]:
    """Return issue counts grouped by issue_type."""
    grouped = defaultdict(int)
    for issue in issues:
        grouped[str(issue.get("issue_type", "unknown"))] += 1
    return dict(sorted(grouped.items()))
