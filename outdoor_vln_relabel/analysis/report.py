"""Markdown reporting for Outdoor-VLN dataset QA."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from outdoor_vln_relabel.io_utils import ensure_dir


def _table(mapping: Dict[str, Any], key_name: str, value_name: str = "count") -> str:
    """Render a small markdown table from a mapping."""
    lines = [f"| {key_name} | {value_name} |", "|---|---:|"]
    if not mapping:
        lines.append("| _none_ | 0 |")
    else:
        for key, value in mapping.items():
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _summary_line(summary: Dict[str, Any]) -> str:
    """Render a numeric summary as one markdown line."""
    return (
        f"- min: {summary.get('min')}\n"
        f"- max: {summary.get('max')}\n"
        f"- mean: {summary.get('mean')}\n"
        f"- median: {summary.get('median')}"
    )


def _issue_counts(issues: List[dict]) -> Dict[str, int]:
    """Count issues by type."""
    counter = Counter(str(issue.get("issue_type", "unknown")) for issue in issues)
    return dict(sorted(counter.items()))


def _issues_by_type(issues: List[dict]) -> Dict[str, List[dict]]:
    """Group issues by issue_type."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for issue in issues:
        grouped[str(issue.get("issue_type", "unknown"))].append(issue)
    return dict(sorted(grouped.items()))


def write_markdown_report(stats: dict, issues: List[dict], output_path: str) -> None:
    """Write a Markdown QA report to output_path."""
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        ensure_dir(path.parent)

    landmark_counts = stats.get("landmark_counts", {})
    issue_counts = _issue_counts(issues)
    issue_groups = _issues_by_type(issues)

    lines = [
        "# Outdoor-VLN Dataset QA Report",
        "",
        "## Basic Statistics",
        f"- total pairs: {stats.get('total_pairs', 0)}",
        f"- scenes: {stats.get('unique_scenes', 0)}",
        f"- segments: {stats.get('unique_segments', 0)}",
        "",
        "### Versions",
        _table(stats.get("version_counts", {}), "version"),
        "",
        "### Rewrite Sources",
        _table(stats.get("rewrite_source_counts", {}), "rewrite_source"),
        "",
        "### Rewrite Counts",
        _table(stats.get("rewrite_counts", {}), "kind"),
        "",
        "## Terrain Distribution",
        _table(stats.get("terrain_counts", {}), "terrain"),
        "",
        "## Motion Distribution",
        _table(stats.get("motion_counts", {}), "motion"),
        "",
        "## Landmark Role Distribution",
        _table(landmark_counts.get("landmark_role_counts", {}), "role"),
        "",
        "## Landmark Relation Distribution",
        _table(landmark_counts.get("landmark_relation_counts", {}), "relation"),
        "",
        "## Confidence Summary",
        _summary_line(stats.get("confidence", {})),
        "",
        "## Instruction Length Summary",
        _summary_line(stats.get("instruction_length", {})),
        "",
        "## Landmark Summary",
        f"- total landmarks: {landmark_counts.get('total_landmarks', 0)}",
        f"- avg landmarks per pair: {landmark_counts.get('avg_landmarks_per_pair', 0)}",
        "",
        "### Landmark Names",
        _table(landmark_counts.get("landmark_name_counts", {}), "landmark"),
        "",
        "## Potential Issues",
        f"- total issues: {len(issues)}",
        "",
        _table(issue_counts, "issue_type"),
        "",
    ]

    if issues:
        lines.append("### Issue Examples")
        shown = 0
        for issue_type, group in issue_groups.items():
            lines.append("")
            lines.append(f"#### {issue_type}")
            for issue in group[:20]:
                if shown >= 20:
                    break
                lines.append(
                    "- "
                    f"index={issue.get('index')} "
                    f"scene={issue.get('scene_id')} "
                    f"segment={issue.get('segment_id')}: "
                    f"{issue.get('message')} "
                    f"`{issue.get('instruction')}`"
                )
                shown += 1
            if shown >= 20:
                break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
