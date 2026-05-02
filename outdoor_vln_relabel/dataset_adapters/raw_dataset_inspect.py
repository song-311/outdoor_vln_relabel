"""Inspect raw image/mask/pose datasets before manifest conversion."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from outdoor_vln_relabel.dataset_adapters.configurable_manifest_converter import (
    load_dataset_config,
)


REQUIRED_POSE_KEYS = ("frame_id", "timestamp", "x", "y", "yaw")


def _section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a config mapping section or an empty mapping."""
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _resolve_root(config: Dict[str, Any], config_path: Path, errors: List[str]) -> Path:
    """Resolve the dataset root and record a clear error if it is not configured."""
    dataset = _section(config, "dataset")
    root_value = str(dataset.get("root", "") or "").strip()
    if not root_value:
        errors.append("dataset.root is empty; configure the real dataset root")
        return config_path.resolve().parent
    root = Path(root_value)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _resolve_path(root: Path, value: Any) -> Optional[Path]:
    """Resolve a configured file or directory path relative to root."""
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _format_frame_pattern(pattern: str, frame_id: str) -> str:
    """Format a frame pattern using numeric frame ids when possible."""
    try:
        return pattern.format(frame_id=int(frame_id))
    except (TypeError, ValueError):
        return pattern.format(frame_id=frame_id)


def _read_pose_rows(pose_path: Path, delimiter: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV pose file and return fieldnames plus rows."""
    with pose_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _column_name(columns: Dict[str, Any], key: str) -> Optional[str]:
    """Return a configured column name."""
    value = columns.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    """Parse a finite float value or return None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _source_path_from_row(
    row: Dict[str, str],
    root: Path,
    data_dir: Optional[Path],
    mode: str,
    pattern: str,
    frame_id: str,
    column_name: Optional[str],
) -> Optional[Path]:
    """Resolve one source image or mask path from matching config and pose row."""
    if mode == "none":
        return None
    if mode == "from_pose_file" and column_name:
        value = str(row.get(column_name, "") or "").strip()
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        base = data_dir if data_dir is not None else root
        return (base / path).resolve()
    if mode in {"by_frame_id", "from_pose_file"}:
        relative = _format_frame_pattern(pattern, frame_id)
        base = data_dir if data_dir is not None else root
        return (base / relative).resolve()
    return None


def _count_files(directory: Optional[Path], extension: str) -> int:
    """Count files with a configured extension under a directory."""
    if directory is None or not directory.is_dir():
        return 0
    pattern = f"*{extension}" if extension else "*"
    return sum(1 for path in directory.rglob(pattern) if path.is_file())


def inspect_raw_dataset(config_path: str) -> Dict[str, Any]:
    """Inspect a real pilot dataset config for ingestion readiness.

    The function does not fabricate missing pose data. Missing pose files or
    required pose columns are returned as hard errors so conversion can stop
    before producing invalid VLN samples.
    """
    config_file = Path(config_path)
    config = load_dataset_config(str(config_file))
    errors: List[str] = []
    warnings: List[str] = []

    dataset = _section(config, "dataset")
    paths = _section(config, "paths")
    pose_config = _section(config, "pose_file")
    columns = _section(pose_config, "columns")
    matching = _section(config, "matching")

    root = _resolve_root(config, config_file, errors)
    if not root.is_dir():
        errors.append(f"dataset.root does not exist or is not a directory: {root}")

    image_dir = _resolve_path(root, paths.get("image_dir"))
    mask_dir = _resolve_path(root, paths.get("mask_dir"))
    pose_path = _resolve_path(root, paths.get("pose_file"))
    image_ext = str(paths.get("image_ext", ".jpg") or "")
    mask_ext = str(paths.get("mask_ext", ".png") or "")

    if image_dir is None:
        errors.append("paths.image_dir is empty")
    elif not image_dir.is_dir():
        errors.append(f"paths.image_dir does not exist: {image_dir}")
    if mask_dir is None:
        warnings.append("paths.mask_dir is empty; mask evidence will be unavailable")
    elif not mask_dir.is_dir():
        warnings.append(f"paths.mask_dir does not exist: {mask_dir}")
    if pose_path is None:
        errors.append("paths.pose_file is empty; real pilot conversion requires pose data")
    elif not pose_path.is_file():
        errors.append(f"paths.pose_file does not exist: {pose_path}")

    fieldnames: List[str] = []
    rows: List[Dict[str, str]] = []
    if pose_path is not None and pose_path.is_file():
        delimiter = str(pose_config.get("delimiter", ","))
        fieldnames, rows = _read_pose_rows(pose_path, delimiter)
        if not fieldnames:
            errors.append(f"pose file has no CSV header: {pose_path}")
        if not rows:
            errors.append(f"pose file contains no pose rows: {pose_path}")

    missing_columns = [
        key
        for key in REQUIRED_POSE_KEYS
        if not _column_name(columns, key)
        or _column_name(columns, key) not in fieldnames
    ]
    required_columns_ok = not missing_columns
    if missing_columns:
        errors.append(
            "pose file missing required configured columns: "
            + ", ".join(missing_columns)
        )

    image_mode = str(matching.get("image_mode", "by_frame_id"))
    image_pattern = str(matching.get("image_pattern", "{frame_id:06d}.jpg"))
    mask_mode = str(matching.get("mask_mode", "by_frame_id"))
    mask_pattern = str(matching.get("mask_pattern", "{frame_id:06d}.png"))
    rgb_col = _column_name(columns, "rgb_path")
    semantic_col = _column_name(columns, "semantic_path")
    frame_id_col = _column_name(columns, "frame_id")

    if image_mode == "from_pose_file" and not rgb_col:
        errors.append("matching.image_mode=from_pose_file requires columns.rgb_path")
    if mask_mode == "from_pose_file" and not semantic_col:
        warnings.append("matching.mask_mode=from_pose_file has no columns.semantic_path")

    matched_images = 0
    matched_masks = 0
    missing_images = 0
    missing_masks = 0
    timestamps: List[Optional[float]] = []
    pose_valid = True

    if required_columns_ok:
        timestamp_col = _column_name(columns, "timestamp")
        x_col = _column_name(columns, "x")
        y_col = _column_name(columns, "y")
        yaw_col = _column_name(columns, "yaw")
        for row in rows:
            frame_id = str(row.get(frame_id_col or "", "")).strip()
            image_path = _source_path_from_row(
                row,
                root,
                image_dir,
                image_mode,
                image_pattern,
                frame_id,
                rgb_col,
            )
            if image_path is not None and image_path.is_file():
                matched_images += 1
            else:
                missing_images += 1

            mask_path = _source_path_from_row(
                row,
                root,
                mask_dir,
                mask_mode,
                mask_pattern,
                frame_id,
                semantic_col,
            )
            if mask_path is not None:
                if mask_path.is_file():
                    matched_masks += 1
                else:
                    missing_masks += 1

            timestamp = _safe_float(row.get(timestamp_col or ""))
            x = _safe_float(row.get(x_col or ""))
            y = _safe_float(row.get(y_col or ""))
            yaw = _safe_float(row.get(yaw_col or ""))
            timestamps.append(timestamp)
            if timestamp is None or x is None or y is None or yaw is None:
                pose_valid = False

    if missing_images:
        errors.append(f"{missing_images} pose rows could not be matched to RGB images")
    if missing_masks:
        warnings.append(f"{missing_masks} pose rows could not be matched to semantic masks")

    timestamp_ok = all(
        timestamps[index] is not None
        and timestamps[index + 1] is not None
        and timestamps[index] <= timestamps[index + 1]
        for index in range(max(0, len(timestamps) - 1))
    )
    if timestamps and not timestamp_ok:
        warnings.append("pose timestamps are not monotonically non-decreasing")
    if rows and not pose_valid:
        errors.append("pose file contains invalid timestamp/x/y/yaw numeric values")

    return {
        "config_path": str(config_file),
        "dataset_name": dataset.get("name", ""),
        "dataset_root": str(root),
        "image_dir": str(image_dir) if image_dir else "",
        "mask_dir": str(mask_dir) if mask_dir else "",
        "pose_file": str(pose_path) if pose_path else "",
        "num_images": _count_files(image_dir, image_ext),
        "num_masks": _count_files(mask_dir, mask_ext),
        "num_poses": len(rows),
        "matched_images": matched_images,
        "matched_masks": matched_masks,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "required_columns_ok": required_columns_ok,
        "timestamp_ok": timestamp_ok,
        "pose_ok": pose_valid and required_columns_ok and bool(rows),
        "warnings": warnings,
        "errors": errors,
        "pose_header": fieldnames,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a raw dataset summary."""
    for key, value in summary.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for raw dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect a raw real-pilot dataset.")
    parser.add_argument("--config", required=True, help="Real pilot YAML config")
    return parser.parse_args()


def main() -> None:
    """Run raw dataset inspection from the command line."""
    args = parse_args()
    _print_summary(inspect_raw_dataset(args.config))


if __name__ == "__main__":
    main()
