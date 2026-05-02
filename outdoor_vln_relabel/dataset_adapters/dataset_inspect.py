"""Inspect unified manifest datasets for basic ingestion readiness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


def _path_exists(value: str, dataset_root: Path) -> bool:
    """Return True for an existing non-empty path."""
    if not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return path.is_file()


def _as_float(value: str) -> float | None:
    """Parse float, returning None when invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_rows(dataset_root: Path) -> List[dict]:
    """Read trajectory.csv rows."""
    trajectory_csv = dataset_root / "trajectory.csv"
    if not trajectory_csv.is_file():
        raise FileNotFoundError(f"Missing trajectory.csv: {trajectory_csv}")
    with trajectory_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"trajectory.csv has no header: {trajectory_csv}")
        return list(reader)


def inspect_manifest(dataset_root: str) -> Dict[str, Any]:
    """Inspect a unified manifest dataset and return a summary dictionary."""
    root = Path(dataset_root)
    rows = _read_rows(root)
    total = len(rows)
    timestamps = [_as_float(row.get("timestamp", "")) for row in rows]
    xs = [_as_float(row.get("x", "")) for row in rows]
    ys = [_as_float(row.get("y", "")) for row in rows]
    yaws = [_as_float(row.get("yaw", "")) for row in rows]
    frame_ids = [row.get("frame_id", "") for row in rows]

    semantic_values = [row.get("semantic_path", "") for row in rows]
    terrain_values = [row.get("terrain", "") for row in rows]
    landmark_values = [row.get("landmarks", "") for row in rows]
    rgb_values = [row.get("rgb_path", "") for row in rows]

    timestamp_increasing = all(
        timestamps[i] is not None
        and timestamps[i + 1] is not None
        and timestamps[i] <= timestamps[i + 1]
        for i in range(max(0, len(timestamps) - 1))
    )
    duplicate_frame_ids = total - len(set(frame_ids))
    missing_images = sum(1 for path in rgb_values if not _path_exists(path, root))
    missing_masks = sum(
        1 for path in semantic_values if path and not _path_exists(path, root)
    )

    return {
        "dataset_root": str(root),
        "trajectory_exists": (root / "trajectory.csv").is_file(),
        "total_frames": total,
        "rgb_path_exists_ratio": (
            round((total - missing_images) / total, 6) if total else 0.0
        ),
        "semantic_path_present_ratio": (
            round(sum(1 for value in semantic_values if value) / total, 6) if total else 0.0
        ),
        "terrain_present_ratio": (
            round(sum(1 for value in terrain_values if value) / total, 6) if total else 0.0
        ),
        "landmarks_present_ratio": (
            round(sum(1 for value in landmark_values if value) / total, 6) if total else 0.0
        ),
        "timestamp_increasing": timestamp_increasing,
        "position_valid": all(x is not None and y is not None for x, y in zip(xs, ys)),
        "yaw_valid": all(yaw is not None for yaw in yaws),
        "duplicate_frame_ids": duplicate_frame_ids,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print inspect summary to terminal."""
    for key, value in summary.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect a unified manifest dataset.")
    parser.add_argument("--dataset_root", required=True, help="Manifest dataset root")
    return parser.parse_args()


def main() -> None:
    """Run manifest inspection."""
    args = parse_args()
    _print_summary(inspect_manifest(args.dataset_root))


if __name__ == "__main__":
    main()
