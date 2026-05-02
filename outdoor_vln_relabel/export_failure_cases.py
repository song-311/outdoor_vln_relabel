"""Export potential failure cases with optional trajectory visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from outdoor_vln_relabel.analysis.visualize import make_sample_card
from outdoor_vln_relabel.io_utils import ensure_dir, load_jsonl, save_jsonl


def _record_for_issue(records: List[dict], issue: dict) -> Dict[str, Any]:
    """Return the record referenced by an issue index, or an empty dict."""
    try:
        index = int(issue.get("index", -1))
    except (TypeError, ValueError):
        return {}
    if 0 <= index < len(records):
        return records[index]
    return {}


def export_failure_cases(
    input_jsonl: str,
    issues_jsonl: str,
    output_dir: str,
    max_cases: int = 50,
) -> Dict[str, Any]:
    """Export issue-linked records and a Markdown failure case index."""
    records = [dict(record) for record in load_jsonl(input_jsonl)]
    issues = [dict(issue) for issue in load_jsonl(issues_jsonl)]
    output = ensure_dir(output_dir)
    case_rows: List[Dict[str, Any]] = []
    lines = [
        "# Outdoor-VLN Failure Cases",
        "",
        f"- input jsonl: {input_jsonl}",
        f"- issues jsonl: {issues_jsonl}",
        f"- total issues: {len(issues)}",
        f"- exported cases: {min(len(issues), max_cases)}",
        "",
    ]

    for case_id, issue in enumerate(issues[: max(0, max_cases)], start=1):
        record = _record_for_issue(records, issue)
        image_name = f"failure_{case_id:03d}_idx_{issue.get('index')}.png"
        image_path = output / image_name
        if record:
            frame_paths = [
                path
                for path in [record.get("start_frame"), record.get("goal_frame")]
                if path
            ]
            make_sample_card(record, image_paths=frame_paths, output_path=str(image_path))
        case = {
            "case_id": case_id,
            "issue": issue,
            "record": record,
            "visualization": str(image_path) if record else "",
        }
        case_rows.append(case)
        lines.extend(
            [
                f"## Case {case_id}",
                "",
                f"- issue_type: {issue.get('issue_type')}",
                f"- index: {issue.get('index')}",
                f"- scene_id: {issue.get('scene_id')}",
                f"- segment_id: {issue.get('segment_id')}",
                f"- message: {issue.get('message')}",
                f"- instruction: {issue.get('instruction')}",
                f"- visualization: {image_name if record else 'none'}",
                "",
            ]
        )

    if not issues:
        lines.append("No potential issues were found.")

    save_jsonl(case_rows, output / "failure_cases.jsonl")
    (output / "failure_cases.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    summary = {
        "input_jsonl": input_jsonl,
        "issues_jsonl": issues_jsonl,
        "output_dir": str(output),
        "total_issues": len(issues),
        "exported_cases": len(case_rows),
        "failure_cases_jsonl": str(output / "failure_cases.jsonl"),
        "failure_cases_md": str(output / "failure_cases.md"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export QA failure cases.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--issues_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Run failure case export."""
    args = parse_args()
    summary = export_failure_cases(
        args.input_jsonl,
        args.issues_jsonl,
        args.output_dir,
        max_cases=args.max_cases,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
