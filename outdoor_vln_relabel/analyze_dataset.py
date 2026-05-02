"""CLI for Outdoor-VLN JSONL QA statistics and Markdown reports."""

from __future__ import annotations

import argparse

from outdoor_vln_relabel.analysis.report import write_markdown_report
from outdoor_vln_relabel.analysis.stats import (
    compute_basic_stats,
    find_potential_issues,
    load_pairs,
)
from outdoor_vln_relabel.io_utils import save_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze an Outdoor-VLN JSONL dataset.")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL dataset")
    parser.add_argument("--output_report", required=True, help="Output Markdown report")
    parser.add_argument("--output_issues", required=True, help="Output issues JSONL")
    return parser.parse_args()


def main() -> None:
    """Run dataset QA and write report artifacts."""
    args = parse_args()
    records = load_pairs(args.input_jsonl)
    stats = compute_basic_stats(records)
    issues = find_potential_issues(records)
    write_markdown_report(stats, issues, args.output_report)
    save_jsonl(issues, args.output_issues)
    print(f"Loaded {len(records)} records")
    print(f"Found {len(issues)} potential issues")
    print(f"Wrote report to {args.output_report}")


if __name__ == "__main__":
    main()

