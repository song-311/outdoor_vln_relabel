"""Export paper-ready CSV/JSON summary tables for Outdoor-VLN datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

from outdoor_vln_relabel.analysis.stats import compute_basic_stats, load_pairs
from outdoor_vln_relabel.io_utils import ensure_dir


def _write_distribution_csv(mapping: Dict[str, Any], output_path: Path, key_name: str) -> None:
    """Write a distribution mapping to CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[key_name, "count"])
        writer.writeheader()
        for key, value in mapping.items():
            writer.writerow({key_name: key, "count": value})


def export_paper_tables(
    input_jsonl: str,
    output_dir: str,
    qa_report: str | None = None,
) -> Dict[str, Any]:
    """Export dataset distributions and summary statistics for paper tables."""
    records = load_pairs(input_jsonl)
    stats = compute_basic_stats(records)
    output = ensure_dir(output_dir)
    landmark_counts = stats.get("landmark_counts", {})

    tables = {
        "terrain_distribution": (
            stats.get("terrain_counts", {}),
            "terrain",
            output / "terrain_distribution.csv",
        ),
        "motion_distribution": (
            stats.get("motion_counts", {}),
            "motion",
            output / "motion_distribution.csv",
        ),
        "landmark_role_distribution": (
            landmark_counts.get("landmark_role_counts", {}),
            "role",
            output / "landmark_role_distribution.csv",
        ),
        "landmark_relation_distribution": (
            landmark_counts.get("landmark_relation_counts", {}),
            "relation",
            output / "landmark_relation_distribution.csv",
        ),
        "version_distribution": (
            stats.get("version_counts", {}),
            "version",
            output / "version_distribution.csv",
        ),
    }
    for mapping, key_name, path in tables.values():
        _write_distribution_csv(mapping, path, key_name)

    summary = {
        "input_jsonl": input_jsonl,
        "qa_report": qa_report,
        "output_dir": str(output),
        "stats": stats,
        "tables": {name: str(path) for name, (_, _, path) in tables.items()},
    }
    (output / "summary_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export paper-ready dataset tables.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--qa_report", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Run paper table export."""
    args = parse_args()
    summary = export_paper_tables(
        input_jsonl=args.input_jsonl,
        qa_report=args.qa_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
