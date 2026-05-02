"""Configurable conversion from image+pose datasets to unified manifests."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from outdoor_vln_relabel.io_utils import ensure_dir


def _parse_scalar(value: str) -> Any:
    """Parse a small YAML scalar subset used by dataset config templates."""
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Load a minimal indentation-based YAML mapping without third-party deps."""
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.lstrip("\ufeff").split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                raise ValueError(
                    "Dataset config fallback loader supports mappings only; "
                    f"found list on line {line_number} of {path}"
                )
            if ":" not in stripped:
                raise ValueError(f"Invalid config line {line_number} in {path}: {raw_line}")
            indent = len(line) - len(line.lstrip(" "))
            key, raw_value = stripped.split(":", 1)
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            key = key.strip()
            if raw_value.strip() == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_scalar(raw_value)
    return root


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """Load a dataset conversion config from YAML."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {path}")
    try:
        import yaml
    except ImportError:
        config = _load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must contain a mapping: {path}")
    return config


def _require_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a required mapping config section."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset config missing mapping section: {key}")
    return value


def _column(columns: Dict[str, Any], key: str, required: bool = True) -> Optional[str]:
    """Return a configured column name."""
    value = columns.get(key)
    if value in (None, ""):
        if required:
            raise ValueError(f"Dataset config missing required column mapping: {key}")
        return None
    return str(value)


def _row_value(row: Dict[str, str], column_name: Optional[str], key: str, required: bool = True) -> str:
    """Read a row value using a configured column name."""
    if not column_name:
        if required:
            raise ValueError(f"Missing required column mapping for {key}")
        return ""
    if column_name not in row:
        raise ValueError(f"Pose file missing configured column '{column_name}' for {key}")
    value = row[column_name]
    if required and (value is None or str(value).strip() == ""):
        raise ValueError(f"Pose row missing required value for {key}")
    return str(value).strip()


def _resolve_source_path(root: Path, subdir: Optional[str], value: str) -> Path:
    """Resolve a source path relative to root and optional subdir."""
    path = Path(value)
    if path.is_absolute():
        return path
    if subdir:
        candidate = root / subdir / path
        if candidate.exists():
            return candidate
    return root / path


def _format_frame_pattern(pattern: str, frame_id: int | str) -> str:
    """Format a frame pattern with numeric frame_id when possible."""
    try:
        numeric = int(frame_id)
        return pattern.format(frame_id=numeric)
    except (ValueError, TypeError):
        return pattern.format(frame_id=frame_id)


def _resolve_rgb_path(
    row: Dict[str, str],
    columns: Dict[str, Any],
    root: Path,
    image_dir: Optional[str],
    image_matching: Dict[str, Any],
    frame_id: int | str,
) -> Path:
    """Resolve an RGB image path using config matching rules."""
    mode = str(image_matching.get("mode") or "from_pose_file")
    rgb_column = _column(columns, "rgb_path", required=False)
    if mode == "from_pose_file" and rgb_column:
        value = _row_value(row, rgb_column, "rgb_path", required=True)
        return _resolve_source_path(root, image_dir, value)
    if mode in {"by_frame_id", "from_pose_file"}:
        pattern = str(image_matching.get("frame_pattern", "{frame_id:06d}.jpg"))
        return _resolve_source_path(root, image_dir, _format_frame_pattern(pattern, frame_id))
    raise ValueError(f"Unsupported image_matching.mode: {mode}")


def _resolve_semantic_path(
    row: Dict[str, str],
    columns: Dict[str, Any],
    root: Path,
    mask_dir: Optional[str],
    semantic_matching: Dict[str, Any],
    frame_id: int | str,
) -> str:
    """Resolve an optional semantic path using config matching rules."""
    mode = str(semantic_matching.get("mode") or "none")
    semantic_column = _column(columns, "semantic_path", required=False)
    if mode == "none":
        return ""
    if mode == "from_pose_file" and semantic_column:
        value = _row_value(row, semantic_column, "semantic_path", required=False)
        return str(_resolve_source_path(root, mask_dir, value)) if value else ""
    if mode in {"by_frame_id", "from_pose_file"}:
        pattern = str(semantic_matching.get("frame_pattern", "{frame_id:06d}.png"))
        path = _resolve_source_path(root, mask_dir, _format_frame_pattern(pattern, frame_id))
        return str(path)
    raise ValueError(f"Unsupported semantic_matching.mode: {mode}")


def _convert_units(value: float, unit: str, kind: str) -> float:
    """Convert timestamp/yaw units to seconds/radians."""
    unit = unit.lower()
    if kind == "timestamp" and unit in {"ms", "millisecond", "milliseconds"}:
        return value / 1000.0
    if kind == "yaw" and unit in {"deg", "degree", "degrees"}:
        return math.radians(value)
    return value


def _copy_if_requested(source: Path, output_dir: Path, subdir: str, copy_files: bool) -> str:
    """Return manifest path, optionally copying source into output_dir/subdir."""
    if not copy_files:
        return str(source)
    target_dir = ensure_dir(output_dir / subdir)
    target = target_dir / source.name
    if source.is_file():
        shutil.copy2(source, target)
    return str(Path(subdir) / source.name)


def _read_pose_rows(pose_path: Path, delimiter: str) -> List[Dict[str, str]]:
    """Read pose CSV rows with clear errors."""
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose file does not exist: {pose_path}")
    with pose_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Pose file has no header: {pose_path}")
        return list(reader)


def convert_dataset_to_manifest(
    config_path: str, output_dir: str | None = None, copy_files: Optional[bool] = None
) -> Dict[str, Any]:
    """Convert a configured source dataset into a unified manifest directory."""
    config = load_dataset_config(config_path)
    dataset = _require_mapping(config, "dataset")
    paths = _require_mapping(config, "paths")
    pose_file = _require_mapping(config, "pose_file")
    columns = _require_mapping(pose_file, "columns")
    image_matching = config.get("image_matching", {}) or {}
    semantic_matching = config.get("semantic_matching", {}) or {}
    defaults = config.get("defaults", {}) or {}

    root_value = str(dataset.get("root", "") or "").strip()
    root = Path(root_value).resolve() if root_value else Path(config_path).resolve().parent
    output = Path(output_dir or paths.get("output_dir") or "outputs/manifest_dataset")
    ensure_dir(output)

    pose_rel = str(paths.get("pose_file") or "poses.csv")
    pose_path = Path(pose_rel)
    if not pose_path.is_absolute():
        pose_path = root / pose_path
    delimiter = str(pose_file.get("delimiter", ","))
    rows = _read_pose_rows(pose_path, delimiter)

    image_dir = paths.get("image_dir")
    mask_dir = paths.get("mask_dir")
    should_copy_files = bool(paths.get("copy_files", False)) if copy_files is None else copy_files
    frame_id_col = _column(columns, "frame_id")
    timestamp_col = _column(columns, "timestamp")
    x_col = _column(columns, "x")
    y_col = _column(columns, "y")
    yaw_col = _column(columns, "yaw")
    terrain_col = _column(columns, "terrain", required=False)
    landmarks_col = _column(columns, "landmarks", required=False)
    default_terrain = str(defaults.get("terrain", "") or "")

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
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows, start=2):
            try:
                frame_id_raw = _row_value(row, frame_id_col, "frame_id")
                frame_id = int(frame_id_raw)
                timestamp = _convert_units(
                    float(_row_value(row, timestamp_col, "timestamp")),
                    str(defaults.get("timestamp_unit", "sec")),
                    "timestamp",
                )
                x = float(_row_value(row, x_col, "x"))
                y = float(_row_value(row, y_col, "y"))
                yaw = _convert_units(
                    float(_row_value(row, yaw_col, "yaw")),
                    str(defaults.get("yaw_unit", "rad")),
                    "yaw",
                )
            except ValueError as exc:
                raise ValueError(f"Invalid pose row {row_index}: {exc}") from exc

            rgb_source = _resolve_rgb_path(
                row, columns, root, image_dir, image_matching, frame_id
            )
            semantic_value = _resolve_semantic_path(
                row, columns, root, mask_dir, semantic_matching, frame_id
            )
            semantic_path = Path(semantic_value) if semantic_value else None
            terrain = (
                _row_value(row, terrain_col, "terrain", required=False)
                if terrain_col
                else default_terrain
            )
            landmarks = (
                _row_value(row, landmarks_col, "landmarks", required=False)
                if landmarks_col
                else ""
            )

            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 6),
                    "rgb_path": _copy_if_requested(rgb_source, output, "frames", should_copy_files),
                    "semantic_path": (
                        _copy_if_requested(semantic_path, output, "masks", should_copy_files)
                        if semantic_path
                        else ""
                    ),
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "yaw": round(yaw, 8),
                    "terrain": terrain,
                    "landmarks": landmarks,
                }
            )

    info = {
        "source_root": str(root),
        "pose_file": str(pose_path),
        "scene_id": dataset.get("scene_id", "scene_001"),
        "sequence_id": dataset.get("sequence_id", "seq_001"),
        "dataset_name": dataset.get("name", "generic_outdoor_dataset"),
        "num_frames": len(rows),
        "copy_files": should_copy_files,
    }
    with (output / "manifest_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)
    return {"output_dir": str(output), "trajectory_csv": str(output_csv), **info}
