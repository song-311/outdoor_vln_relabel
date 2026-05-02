"""CLI for visualizing random Outdoor-VLN JSONL samples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from outdoor_vln_relabel.analysis.stats import load_pairs
from outdoor_vln_relabel.analysis.visualize import make_sample_card
from outdoor_vln_relabel.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize random Outdoor-VLN samples.")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL dataset")
    parser.add_argument("--output_dir", required=True, help="Output visualization directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _sample_records(records: List[dict], num_samples: int, seed: int) -> List[tuple[int, dict]]:
    """Return indexed random samples without replacement."""
    indexed = list(enumerate(records))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    return indexed[: max(0, min(num_samples, len(indexed)))]


def _landmarks_text(record: dict) -> str:
    """Return compact landmark text for Markdown."""
    items = []
    for landmark in record.get("landmarks") or []:
        if isinstance(landmark, dict):
            items.append(
                f"{landmark.get('name')} ({landmark.get('role')}, {landmark.get('relation')})"
            )
    return ", ".join(items) if items else "none"


def _write_samples_md(samples: List[tuple[int, dict]], output_dir: Path, image_names: List[str]) -> None:
    """Write a Markdown index for visualized samples."""
    lines = ["# Outdoor-VLN Sample Visualizations", ""]
    for (index, record), image_name in zip(samples, image_names):
        lines.extend(
            [
                f"## Sample {index}",
                "",
                f"![sample]({image_name})",
                "",
                f"- instruction: {record.get('instruction', '')}",
                f"- terrain: {record.get('terrain')}",
                f"- motion: {record.get('motion')}",
                f"- confidence: {record.get('confidence')}",
                f"- landmarks: {_landmarks_text(record)}",
                "",
            ]
        )
    (output_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate sample visualization PNGs and a Markdown index."""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    records = load_pairs(args.input_jsonl)
    samples = _sample_records(records, args.num_samples, args.seed)
    image_names: List[str] = []
    for sample_number, (index, record) in enumerate(samples, start=1):
        image_name = f"sample_{sample_number:03d}_idx_{index:06d}.png"
        output_path = Path(output_dir) / image_name
        image_paths = [
            path for path in [record.get("start_frame"), record.get("goal_frame")] if path
        ]
        make_sample_card(record, image_paths=image_paths, output_path=str(output_path))
        image_names.append(image_name)
    _write_samples_md(samples, Path(output_dir), image_names)
    print(f"Loaded {len(records)} records")
    print(f"Visualized {len(samples)} samples")
    print(f"Wrote samples to {output_dir}")


if __name__ == "__main__":
    main()

