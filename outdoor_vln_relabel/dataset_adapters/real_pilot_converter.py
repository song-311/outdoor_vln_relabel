"""Convert configured real-pilot datasets into unified Outdoor-VLN manifests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from outdoor_vln_relabel.dataset_adapters.configurable_manifest_converter import (
    load_dataset_config,
)
from outdoor_vln_relabel.dataset_adapters.raw_dataset_inspect import (
    _column_name,
    _read_pose_rows,
    _resolve_path,
    _resolve_root,
    _section,
    _source_path_from_row,
    inspect_raw_dataset,
)
from outdoor_vln_relabel.io_utils import ensure_dir


def _convert_units(value: float, unit: str, kind: str) -> float:
    """Convert configured timestamp/yaw units to seconds/radians."""
    lowered = unit.lower()
    if kind == "timestamp" and lowered in {"ms", "millisecond", "milliseconds"}:
        return value / 1000.0
    if kind == "yaw" and lowered in {"deg", "degree", "degrees"}:
        return math.radians(value)
    return value


def _row_float(row: Dict[str, str], column: Optional[str], field: str, row_number: int) -> float:
    """Read a required finite float from a pose row."""
    if not column:
        raise ValueError(f"Missing configured column for {field}")
    value = row.get(column)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for {field} on pose row {row_number}: {value}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite value for {field} on pose row {row_number}: {value}")
    return parsed


def _row_text(row: Dict[str, str], column: Optional[str]) -> str:
    """Read an optional text value from a pose row."""
    if not column:
        return ""
    return str(row.get(column, "") or "").strip()


def _summary_errors(summary: Dict[str, Any], allow_missing_masks: bool) -> List[str]:
    """Return conversion-blocking errors from an inspect summary."""
    errors = list(summary.get("errors") or [])
    if not allow_missing_masks and int(summary.get("missing_masks", 0) or 0) > 0:
        errors.append(
            f"{summary.get('missing_masks')} pose rows could not be matched to semantic masks"
        )
    return errors


def _write_manifest_info(output: Path, info: Dict[str, Any]) -> None:
    """Write manifest_info.json."""
    with (output / "manifest_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)


def convert_real_pilot_to_manifest(
    config_path: str,
    output_dir: Optional[str] = None,
    allow_missing_masks: bool = False,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert a real-pilot dataset config into a unified manifest directory.

    Conversion stops when required pose data, required columns, or RGB image
    matches are missing. This prevents accidental creation of formal VLN samples
    from fabricated or incomplete trajectories.
    """
    summary = inspect_raw_dataset(config_path)
    blocking_errors = _summary_errors(summary, allow_missing_masks)
    if blocking_errors:
        raise ValueError(
            "Raw dataset is not ready for manifest conversion:\n- "
            + "\n- ".join(blocking_errors)
        )

    config = load_dataset_config(config_path)
    dataset = _section(config, "dataset")
    paths = _section(config, "paths")
    pose_config = _section(config, "pose_file")
    columns = _section(pose_config, "columns")
    matching = _section(config, "matching")
    defaults = _section(config, "defaults")
    semantic = _section(config, "semantic")
    pilot = _section(config, "pilot")

    root = _resolve_root(config, Path(config_path), [])
    output = Path(output_dir or dataset.get("output_dir") or paths.get("output_dir") or "outputs/real_pilot_manifest")
    ensure_dir(output)

    pose_path = _resolve_path(root, paths.get("pose_file"))
    if pose_path is None or not pose_path.is_file():
        raise FileNotFoundError(f"Real pilot pose file does not exist: {pose_path}")
    delimiter = str(pose_config.get("delimiter", ","))
    _, rows = _read_pose_rows(pose_path, delimiter)

    frame_limit = max_frames
    if frame_limit is None and pilot.get("max_frames") not in (None, ""):
        frame_limit = int(pilot.get("max_frames"))
    if frame_limit is not None:
        rows = rows[: max(0, frame_limit)]

    image_dir = _resolve_path(root, paths.get("image_dir"))
    mask_dir = _resolve_path(root, paths.get("mask_dir"))
    image_mode = str(matching.get("image_mode", "by_frame_id"))
    image_pattern = str(matching.get("image_pattern", "{frame_id:06d}.jpg"))
    mask_mode = str(matching.get("mask_mode", "by_frame_id"))
    mask_pattern = str(matching.get("mask_pattern", "{frame_id:06d}.png"))

    frame_id_col = _column_name(columns, "frame_id")
    timestamp_col = _column_name(columns, "timestamp")
    x_col = _column_name(columns, "x")
    y_col = _column_name(columns, "y")
    yaw_col = _column_name(columns, "yaw")
    rgb_col = _column_name(columns, "rgb_path")
    semantic_col = _column_name(columns, "semantic_path")
    terrain_col = _column_name(columns, "terrain")
    landmarks_col = _column_name(columns, "landmarks")

    output_csv = output / "trajectory.csv"
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
    matched_images = 0
    matched_masks = 0
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_number, row in enumerate(rows, start=2):
            frame_id_raw = _row_text(row, frame_id_col)
            if not frame_id_raw:
                raise ValueError(f"Pose row {row_number} missing frame_id")
            frame_id = int(frame_id_raw)
            timestamp = _convert_units(
                _row_float(row, timestamp_col, "timestamp", row_number),
                str(defaults.get("timestamp_unit", "sec")),
                "timestamp",
            )
            x = _row_float(row, x_col, "x", row_number)
            y = _row_float(row, y_col, "y", row_number)
            yaw = _convert_units(
                _row_float(row, yaw_col, "yaw", row_number),
                str(defaults.get("yaw_unit", "rad")),
                "yaw",
            )

            rgb_path = _source_path_from_row(
                row, root, image_dir, image_mode, image_pattern, frame_id_raw, rgb_col
            )
            if rgb_path is None or not rgb_path.is_file():
                raise FileNotFoundError(
                    f"Pose row {row_number} RGB image is missing for frame_id={frame_id_raw}: {rgb_path}"
                )
            matched_images += 1

            semantic_path = _source_path_from_row(
                row, root, mask_dir, mask_mode, mask_pattern, frame_id_raw, semantic_col
            )
            semantic_value = ""
            if semantic_path is not None:
                if semantic_path.is_file():
                    semantic_value = str(semantic_path)
                    matched_masks += 1
                elif not allow_missing_masks:
                    raise FileNotFoundError(
                        "Pose row "
                        f"{row_number} semantic mask is missing for frame_id={frame_id_raw}: {semantic_path}"
                    )

            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 6),
                    "rgb_path": str(rgb_path),
                    "semantic_path": semantic_value,
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "yaw": round(yaw, 8),
                    "terrain": _row_text(row, terrain_col) if terrain_col else "",
                    "landmarks": _row_text(row, landmarks_col),
                }
            )

    info = {
        "source_dataset": dataset.get("name", "real_pilot"),
        "source_root": str(root),
        "scene_id": dataset.get("scene_id", "real_scene_000"),
        "sequence_id": dataset.get("sequence_id", "seq_000"),
        "num_frames": len(rows),
        "num_matched_images": matched_images,
        "num_matched_masks": matched_masks,
        "config_path": str(Path(config_path)),
        "semantic_label_map": semantic.get("label_map"),
        "synthetic_pose": False,
        "demo_only": bool(str(dataset.get("name", "")).startswith("demo")),
        "raw_inspect": summary,
    }
    _write_manifest_info(output, info)
    return {"output_dir": str(output), "trajectory_csv": str(output_csv), **info}
