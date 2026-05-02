"""Dataset-level constrained rewrite orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import List

from outdoor_vln_relabel.validation.checks import validate_pair

from .constraints import check_rewrite_constraints
from .rule_rewriter import rule_based_rewrite


def _rewritten_version(version: str) -> str:
    """Return the rewritten dataset version string."""
    mapping = {
        "evidence_landmark_terrain_motion_v3": "rewritten_evidence_landmark_terrain_motion_v4",
        "landmark_terrain_motion_v2": "rewritten_landmark_terrain_motion_v4",
        "terrain_motion_v1": "rewritten_terrain_motion_v4",
        "v0_motion": "rewritten_v0_motion_v4",
    }
    return mapping.get(version, f"rewritten_{version}")


def _generate_rewrites(record: dict, backend: str, num_variants: int) -> List[str]:
    """Generate rewrite candidates with a selected backend."""
    if backend == "rule_based":
        return rule_based_rewrite(record, num_variants=num_variants)
    if backend == "llm_stub":
        return []
    raise ValueError(f"Unsupported rewrite backend: {backend}")


def rewrite_records(
    records: List[dict],
    backend: str = "rule_based",
    num_variants: int = 3,
    keep_original: bool = True,
) -> List[dict]:
    """Rewrite records, dropping candidates that violate constraints."""
    if backend not in {"rule_based", "llm_stub"}:
        raise ValueError("backend must be one of: rule_based, llm_stub")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive")

    output: List[dict] = []
    for record_index, record in enumerate(records):
        if keep_original:
            output.append(deepcopy(record))
        rewrites = _generate_rewrites(record, backend, num_variants)
        for rewrite_id, rewrite in enumerate(rewrites):
            issues = check_rewrite_constraints(record, rewrite)
            if issues:
                continue
            rewritten = deepcopy(record)
            rewritten["original_instruction"] = record.get("instruction", "")
            rewritten["instruction"] = rewrite
            rewritten["rewrite_source"] = backend
            rewritten["rewrite_id"] = rewrite_id
            rewritten["rewrite_record_index"] = record_index
            rewritten["rewrite_constraint_issues"] = []
            rewritten["version"] = _rewritten_version(str(record.get("version", "")))
            if not validate_pair(rewritten):
                rewritten["rewrite_constraint_issues"] = ["validate_pair_failed"]
                continue
            output.append(rewritten)
    return output

