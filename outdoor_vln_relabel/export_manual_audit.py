"""Export a sampled manual audit CSV for generated Outdoor-VLN pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from outdoor_vln_relabel.io_utils import ensure_dir, load_jsonl


AUDIT_COLUMNS = [
    "index",
    "scene_id",
    "segment_id",
    "instruction",
    "terrain",
    "motion",
    "landmarks",
    "trajectory_summary",
    "confidence",
    "version",
    "image_start",
    "image_goal",
    "human_instruction_correct",
    "human_landmarks_visible",
    "human_motion_correct",
    "human_terrain_correct",
    "human_notes",
]


def _sample_records(records: List[dict], num_samples: int, seed: int) -> List[Tuple[int, dict]]:
    """Return deterministic random sampled records with original indices."""
    indexed = list(enumerate(records))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    return indexed[: max(0, min(num_samples, len(indexed)))]


def _landmarks_json(record: Dict[str, Any]) -> str:
    """Serialize landmarks compactly for manual audit."""
    landmarks = record.get("landmarks") or []
    return json.dumps(landmarks, ensure_ascii=False)


def _trajectory_summary(record: Dict[str, Any]) -> str:
    """Return a concise trajectory summary string."""
    trajectory = record.get("trajectory_xy") or []
    if trajectory:
        start = trajectory[0]
        goal = trajectory[-1]
        start_text = f"({float(start[0]):.2f},{float(start[1]):.2f})"
        goal_text = f"({float(goal[0]):.2f},{float(goal[1]):.2f})"
    else:
        start_text = "(n/a)"
        goal_text = "(n/a)"
    return (
        f"points={len(trajectory)}; start={start_text}; goal={goal_text}; "
        f"distance_m={record.get('distance_m')}; duration_s={record.get('duration_s')}; "
        f"heading_change_deg={record.get('heading_change_deg')}"
    )


def export_manual_audit(
    input_jsonl: str,
    output_csv: str,
    num_samples: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sample generated records and write a manual audit CSV."""
    records = load_jsonl(input_jsonl)
    samples = _sample_records(records, num_samples, seed)
    output_path = Path(output_csv)
    if output_path.parent and str(output_path.parent) != ".":
        ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for index, record in samples:
            writer.writerow(
                {
                    "index": index,
                    "scene_id": record.get("scene_id", ""),
                    "segment_id": record.get("segment_id", ""),
                    "instruction": record.get("instruction", ""),
                    "terrain": record.get("terrain", ""),
                    "motion": record.get("motion", ""),
                    "landmarks": _landmarks_json(record),
                    "trajectory_summary": _trajectory_summary(record),
                    "confidence": record.get("confidence", ""),
                    "version": record.get("version", ""),
                    "image_start": record.get("start_frame", ""),
                    "image_goal": record.get("goal_frame", ""),
                    "human_instruction_correct": "",
                    "human_landmarks_visible": "",
                    "human_motion_correct": "",
                    "human_terrain_correct": "",
                    "human_notes": "",
                }
            )
    return {
        "input_jsonl": input_jsonl,
        "output_csv": str(output_path),
        "num_records": len(records),
        "num_samples": len(samples),
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export a manual audit CSV.")
    parser.add_argument("--input_jsonl", required=True, help="Input generated JSONL")
    parser.add_argument("--output_csv", required=True, help="Output audit CSV")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run manual audit export."""
    args = parse_args()
    result = export_manual_audit(
        args.input_jsonl,
        args.output_csv,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
