"""Inspect one semantic mask with an Outdoor-VLN label map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from outdoor_vln_relabel.perception.landmarks_from_mask import detect_landmarks_from_mask
from outdoor_vln_relabel.perception.semantic_mask import (
    load_label_map,
    parse_mask_labels,
    read_semantic_mask,
)
from outdoor_vln_relabel.perception.terrain_from_mask import classify_terrain_from_mask


DEFAULT_PALETTE: Dict[str, tuple[int, int, int]] = {
    "unknown": (20, 20, 20),
    "dirt_trail": (138, 93, 45),
    "grass": (54, 150, 65),
    "vegetation": (27, 95, 45),
    "mud_water": (55, 92, 132),
    "rough_terrain": (145, 145, 145),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect an Outdoor-VLN semantic mask.")
    parser.add_argument("--mask_path", required=True, help="Input semantic mask path")
    parser.add_argument(
        "--semantic_label_map", required=True, help="Semantic label-map YAML path"
    )
    parser.add_argument("--output_json", required=True, help="Output JSON path")
    parser.add_argument(
        "--output_vis",
        default=None,
        help="Optional PNG visualization of parsed outdoor groups",
    )
    return parser.parse_args()


def _save_visualization(mask: np.ndarray, label_map: Dict[str, Any], output_path: str) -> None:
    """Save a simple color visualization for indexed or color masks."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("PIL/Pillow is required for --output_vis") from exc

    if label_map.get("mask_type") == "color" and mask.ndim == 3:
        vis = mask[..., :3].astype(np.uint8)
    else:
        indexed = mask[..., 0] if mask.ndim == 3 else mask
        vis = np.zeros((indexed.shape[0], indexed.shape[1], 3), dtype=np.uint8)
        for label_id, entry in label_map.get("labels", {}).items():
            color = entry.get("color")
            if color is None:
                color = DEFAULT_PALETTE.get(
                    str(entry.get("outdoor_group", "unknown")),
                    DEFAULT_PALETTE["unknown"],
                )
            vis[indexed == int(label_id)] = np.asarray(color[:3], dtype=np.uint8)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis, mode="RGB").save(path)


def inspect_mask(mask_path: str, semantic_label_map: str) -> Dict[str, Any]:
    """Return label stats, terrain classification, and mask-derived landmarks."""
    label_map = load_label_map(semantic_label_map)
    mask = read_semantic_mask(mask_path)
    parsed = parse_mask_labels(mask, label_map)
    terrain = classify_terrain_from_mask(mask_path, label_map)
    landmarks = [
        landmark.to_dict()
        for landmark in detect_landmarks_from_mask(mask_path, label_map)
    ]
    return {
        "mask_path": mask_path,
        "semantic_label_map": semantic_label_map,
        "label_statistics": parsed,
        "terrain": terrain,
        "landmarks": landmarks,
    }


def main() -> None:
    """Run semantic mask inspection and save JSON output."""
    args = parse_args()
    label_map = load_label_map(args.semantic_label_map)
    result = inspect_mask(args.mask_path, args.semantic_label_map)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_vis:
        _save_visualization(read_semantic_mask(args.mask_path), label_map, args.output_vis)
    print(
        "Semantic mask labels: "
        f"{len(result['label_statistics']['labels'])}; "
        f"terrain={result['terrain']['dominant_terrain']}; "
        f"landmarks={len(result['landmarks'])}; "
        f"output={output_json}"
    )


if __name__ == "__main__":
    main()
