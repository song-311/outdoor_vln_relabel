"""Simple folder+poses.csv to unified manifest conversion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

from outdoor_vln_relabel.io_utils import ensure_dir


def _find_frame_path(frames_dir: Path, frame_id: int) -> Path:
    """Find a frame image by zero-padded frame id."""
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = frames_dir / f"{frame_id:06d}{suffix}"
        if candidate.is_file():
            return candidate
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = frames_dir / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    return frames_dir / f"{frame_id:06d}.jpg"


def convert_folder_pose_dataset(
    input_root: str | Path,
    output_dir: str | Path,
    scene_id: str,
    sequence_id: str,
    default_terrain: str = "dirt_trail",
) -> Dict[str, object]:
    """Convert input_root/frames plus input_root/poses.csv into a manifest."""
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    poses_path = root / "poses.csv"
    frames_dir = root / "frames"
    if not poses_path.is_file():
        raise FileNotFoundError(f"Missing poses.csv: {poses_path}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")

    output = ensure_dir(output_dir)
    trajectory_csv = Path(output) / "trajectory.csv"
    required = {"frame_id", "timestamp", "x", "y", "yaw"}
    count = 0
    with poses_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"poses.csv has no header: {poses_path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"poses.csv missing required columns: {', '.join(sorted(missing))}")
        with trajectory_csv.open("w", encoding="utf-8", newline="") as target:
            fieldnames = [
                "frame_id",
                "timestamp",
                "rgb_path",
                "semantic_path",
                "x",
                "y",
                "yaw",
                "terrain",
                "landmarks",
            ]
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                try:
                    frame_id = int(row["frame_id"])
                    timestamp = float(row["timestamp"])
                    x = float(row["x"])
                    y = float(row["y"])
                    yaw = float(row["yaw"])
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value in poses.csv row {row_number}: {exc}") from exc
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 6),
                        "rgb_path": str(_find_frame_path(frames_dir, frame_id)),
                        "semantic_path": "",
                        "x": round(x, 6),
                        "y": round(y, 6),
                        "yaw": round(yaw, 8),
                        "terrain": default_terrain,
                        "landmarks": "",
                    }
                )
                count += 1

    info = {
        "source_root": str(root),
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "dataset_name": "folder_pose",
        "num_frames": count,
        "copy_files": False,
    }
    with (Path(output) / "manifest_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)
    return {"output_dir": str(output), "trajectory_csv": str(trajectory_csv), **info}
