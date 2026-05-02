"""Create a tiny manifest-style dataset for evidence-grounding demos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _write_placeholder_image(path: Path) -> None:
    """Write a tiny image file using PIL when available, otherwise bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        if path.suffix.lower() == ".png":
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
                b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
                b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
            )
        else:
            path.write_bytes(
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
                b"\x00\x01\x00\x00\xff\xdb\x00C\x00" + b"\x08" * 64 + b"\xff\xd9"
            )
        return

    color = (210, 220, 210) if path.suffix.lower() == ".jpg" else (0, 0, 0)
    image = Image.new("RGB", (4, 4), color=color)
    image.save(path)


def _write_index_mask(path: Path, frame_id: int) -> None:
    """Write a small indexed semantic mask with real label ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("PIL/Pillow is required for --with_index_masks") from exc

    height, width = 96, 96
    mask = np.ones((height, width), dtype=np.uint8)
    if frame_id < 50:
        # Mostly dirt road, with a right/front-right bush obstacle.
        mask[52:88, 60:88] = 3
    else:
        # Keep a visible path above, but make the ground ROI dominated by puddle.
        mask[44:96, 10:86] = 5
        mask[58:96, 40:56] = 1
    Image.fromarray(mask, mode="L").save(path)


def _landmarks_for_frame(frame_id: int) -> list[dict]:
    """Return sparse frame-level landmark metadata for the demo manifest."""
    if frame_id in {5, 10, 15, 20}:
        return [
            {
                "name": "dirt trail",
                "category": "path_like",
                "relation": "ahead",
                "role": "follow",
                "score": 0.9,
            }
        ]
    if frame_id in {30, 35, 40}:
        return [
            {
                "name": "bushes",
                "category": "obstacle_like",
                "relation": "right",
                "role": "avoid",
                "score": 0.82,
            },
            {
                "name": "dirt trail",
                "category": "path_like",
                "relation": "ahead",
                "role": "follow",
                "score": 0.88,
            },
        ]
    if frame_id in {70, 75, 80}:
        return [
            {
                "name": "open path",
                "category": "path_like",
                "relation": "ahead",
                "role": "follow",
                "score": 0.84,
            }
        ]
    return []


def _mask_name(frame_id: int) -> str:
    """Return mask filenames that encode lightweight semantic evidence."""
    if frame_id < 25:
        return f"{frame_id:06d}_trail.png"
    if frame_id < 50:
        return f"{frame_id:06d}_bush.png"
    if frame_id < 75:
        return f"{frame_id:06d}_puddle.png"
    return f"{frame_id:06d}_water.png"


def _write_rows(output_dir: Path, rows: Iterable[dict]) -> None:
    """Write trajectory.csv rows."""
    csv_path = output_dir / "trajectory.csv"
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
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_demo_manifest(output_dir: str | Path, with_index_masks: bool = False) -> None:
    """Generate a small manifest dataset with trajectory, frames, masks, and metadata."""
    root = Path(output_dir)
    frames_dir = root / "frames"
    masks_dir = root / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    num_frames = 100
    timestamps = np.arange(num_frames) * 0.25
    x_values = np.linspace(0, 20, num_frames)
    y_values = np.zeros(num_frames)
    y_values[50:] = np.linspace(0, 5, num_frames - 50)
    yaws = np.zeros(num_frames)
    yaws[50:] = np.linspace(0, 0.35, num_frames - 50)

    rows = []
    for frame_id in range(num_frames):
        rgb_rel = f"frames/{frame_id:06d}.jpg"
        mask_rel = (
            f"masks/{frame_id:06d}.png"
            if with_index_masks
            else f"masks/{_mask_name(frame_id)}"
        )
        _write_placeholder_image(root / rgb_rel)
        if with_index_masks:
            _write_index_mask(root / mask_rel, frame_id)
        else:
            _write_placeholder_image(root / mask_rel)

        terrain = "dirt_trail" if frame_id < 50 else "mud_water"
        # Leave a subset blank so metadata_or_mask exercises the mask fallback.
        if 58 <= frame_id <= 99:
            terrain = ""
        landmarks = _landmarks_for_frame(frame_id)
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp": round(float(timestamps[frame_id]), 6),
                "rgb_path": rgb_rel,
                "semantic_path": mask_rel,
                "x": round(float(x_values[frame_id]), 6),
                "y": round(float(y_values[frame_id]), 6),
                "yaw": round(float(yaws[frame_id]), 6),
                "terrain": terrain,
                "landmarks": json.dumps(landmarks) if landmarks else "",
            }
        )

    _write_rows(root, rows)
    metadata = {
        "description": "Tiny Outdoor-VLN-Relabel demo manifest",
        "num_frames": num_frames,
    }
    with (root / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create an Outdoor-VLN demo manifest.")
    parser.add_argument("--output_dir", required=True, help="Output manifest directory")
    parser.add_argument(
        "--with_index_masks",
        action="store_true",
        help="Generate real indexed semantic masks using generic_outdoor label ids",
    )
    return parser.parse_args()


def main() -> None:
    """Run the demo manifest generator."""
    args = parse_args()
    make_demo_manifest(args.output_dir, with_index_masks=args.with_index_masks)
    print(f"Wrote demo manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
