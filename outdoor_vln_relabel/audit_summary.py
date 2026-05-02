"""Summarize manually filled Outdoor-VLN audit CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from outdoor_vln_relabel.io_utils import ensure_dir


AUDIT_FIELDS = [
    "human_instruction_correct",
    "human_landmarks_visible",
    "human_motion_correct",
    "human_terrain_correct",
]


TRUE_VALUES = {"true", "yes", "1", "y", "t"}
FALSE_VALUES = {"false", "no", "0", "n", "f"}


def _parse_bool(value: Any) -> Optional[bool]:
    """Parse a manual audit boolean value or return None for blank/unknown."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded rate or None when the denominator is zero."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _read_audit_rows(path: str) -> List[dict]:
    """Read manual audit CSV rows."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manual audit CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manual audit CSV has no header: {csv_path}")
        return list(reader)


def summarize_audit(audit_csv: str) -> Dict[str, Any]:
    """Summarize human audit fields from a manual audit CSV."""
    rows = _read_audit_rows(audit_csv)
    field_stats: Dict[str, Dict[str, Any]] = {}
    parsed_by_row: List[Dict[str, Optional[bool]]] = []
    for row in rows:
        parsed = {field: _parse_bool(row.get(field)) for field in AUDIT_FIELDS}
        parsed_by_row.append(parsed)

    for field in AUDIT_FIELDS:
        values = [parsed[field] for parsed in parsed_by_row if parsed[field] is not None]
        correct = sum(1 for value in values if value is True)
        incorrect = sum(1 for value in values if value is False)
        field_stats[field] = {
            "num_rated": len(values),
            "num_true": correct,
            "num_false": incorrect,
            "rate": _rate(correct, len(values)),
        }

    audited_rows = [
        parsed for parsed in parsed_by_row if any(value is not None for value in parsed.values())
    ]
    all_fields_rows = [
        parsed for parsed in parsed_by_row if all(parsed[field] is not None for field in AUDIT_FIELDS)
    ]
    all_correct = sum(
        1 for parsed in all_fields_rows if all(parsed[field] is True for field in AUDIT_FIELDS)
    )

    failure_notes = Counter()
    for row, parsed in zip(rows, parsed_by_row):
        has_failure = any(value is False for value in parsed.values())
        note = str(row.get("human_notes", "") or "").strip()
        if has_failure and note:
            failure_notes[note] += 1

    return {
        "audit_csv": audit_csv,
        "num_rows": len(rows),
        "num_audited": len(audited_rows),
        "instruction_correct_rate": field_stats["human_instruction_correct"]["rate"],
        "landmarks_visible_rate": field_stats["human_landmarks_visible"]["rate"],
        "motion_correct_rate": field_stats["human_motion_correct"]["rate"],
        "terrain_correct_rate": field_stats["human_terrain_correct"]["rate"],
        "all_correct_rate": _rate(all_correct, len(all_fields_rows)),
        "all_correct_denominator": len(all_fields_rows),
        "field_stats": field_stats,
        "common_failure_notes": [
            {"note": note, "count": count}
            for note, count in failure_notes.most_common(20)
        ],
    }


def _write_markdown(summary: Dict[str, Any], output_path: str) -> None:
    """Write audit summary as Markdown."""
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        ensure_dir(path.parent)
    lines = [
        "# Outdoor-VLN Manual Audit Summary",
        "",
        f"- audit csv: {summary.get('audit_csv')}",
        f"- rows: {summary.get('num_rows')}",
        f"- audited rows: {summary.get('num_audited')}",
        f"- instruction correct rate: {summary.get('instruction_correct_rate')}",
        f"- landmarks visible rate: {summary.get('landmarks_visible_rate')}",
        f"- motion correct rate: {summary.get('motion_correct_rate')}",
        f"- terrain correct rate: {summary.get('terrain_correct_rate')}",
        f"- all correct rate: {summary.get('all_correct_rate')}",
        "",
        "| field | rated | true | false | rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, stats in summary.get("field_stats", {}).items():
        lines.append(
            f"| {field} | {stats.get('num_rated')} | {stats.get('num_true')} | "
            f"{stats.get('num_false')} | {stats.get('rate')} |"
        )
    lines.extend(["", "## Common Failure Notes", ""])
    notes = summary.get("common_failure_notes", [])
    if not notes:
        lines.append("- none")
    else:
        for item in notes:
            lines.append(f"- {item.get('count')}: {item.get('note')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_summary(audit_csv: str, output_json: str, output_md: str) -> Dict[str, Any]:
    """Summarize an audit CSV and write JSON/Markdown outputs."""
    summary = summarize_audit(audit_csv)
    json_path = Path(output_json)
    if json_path.parent and str(json_path.parent) != ".":
        ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(summary, output_md)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Summarize manual audit CSV results.")
    parser.add_argument("--audit_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    return parser.parse_args()


def main() -> None:
    """Run audit summary generation."""
    args = parse_args()
    summary = write_audit_summary(args.audit_csv, args.output_json, args.output_md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
