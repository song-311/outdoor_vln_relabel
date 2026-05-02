"""Semantic mask loading and label-stat extraction utilities.

The parser is intentionally dataset-agnostic. A lightweight YAML label map
describes whether masks are indexed or color-coded, and maps each dataset label
to the five Outdoor-VLN terrain groups plus a navigation role.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _parse_scalar(value: str) -> Any:
    """Parse the small YAML scalar subset used by semantic label maps."""
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid inline list value in semantic label map: {value}") from exc
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load a minimal indentation-based YAML mapping without third-party deps."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                raise ValueError(
                    "Simple semantic label map loader only supports mappings and "
                    f"inline lists; found block list on line {line_number} of {path}"
                )
            if ":" not in stripped:
                raise ValueError(
                    f"Invalid semantic label map line {line_number} in {path}: {raw_line}"
                )
            indent = len(line) - len(line.lstrip(" "))
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if raw_value.strip() == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_scalar(raw_value)
    return root


def _normalize_label_key(key: Any) -> Any:
    """Return an integer label id when a YAML key is numeric."""
    if isinstance(key, int):
        return key
    text = str(key).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _normalize_label_map(label_map: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a semantic label map dictionary."""
    if not isinstance(label_map, dict):
        raise ValueError("Semantic label map must be a dictionary")
    mask_type = str(label_map.get("mask_type", "indexed")).strip().lower()
    if mask_type not in {"indexed", "color"}:
        raise ValueError(f"Unsupported semantic mask_type: {mask_type}")
    labels = label_map.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Semantic label map must define a labels mapping")

    normalized_labels: Dict[Any, Dict[str, Any]] = {}
    for key, value in labels.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid semantic label entry for {key}")
        label_key = _normalize_label_key(key)
        entry = dict(value)
        if "name" not in entry:
            entry["name"] = f"label_{label_key}"
        entry["outdoor_group"] = str(entry.get("outdoor_group", "unknown"))
        entry["role"] = str(entry.get("role", "ignore"))
        if "color" in entry and entry["color"] is not None:
            entry["color"] = [int(channel) for channel in entry["color"]]
        normalized_labels[label_key] = entry

    normalized = dict(label_map)
    normalized["mask_type"] = mask_type
    normalized["labels"] = normalized_labels
    return normalized


def load_label_map(path: str) -> Dict[str, Any]:
    """Load a semantic label map YAML file.

    PyYAML is used when available. A small fallback parser handles the bundled
    simple mapping files so this module remains usable without heavy dependencies.
    """
    label_map_path = Path(path)
    if not label_map_path.is_file():
        raise FileNotFoundError(f"Semantic label map does not exist: {label_map_path}")
    try:
        import yaml
    except ImportError:
        label_map = _load_simple_yaml_mapping(label_map_path)
    else:
        with label_map_path.open("r", encoding="utf-8") as file:
            label_map = yaml.safe_load(file) or {}
    return _normalize_label_map(label_map)


def read_semantic_mask(mask_path: str) -> np.ndarray:
    """Read a semantic mask image as a numpy array.

    Indexed masks are usually single-channel PNGs. Color masks are RGB PNGs.
    The caller decides how to interpret the returned array using the label map.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Semantic mask file does not exist: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("PIL/Pillow is required to read semantic mask images") from exc
    with Image.open(path) as image:
        return np.asarray(image)


def _indexed_mask_array(mask: np.ndarray) -> np.ndarray:
    """Return a 2D label-id array for an indexed semantic mask."""
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return mask[..., 0]
    raise ValueError(f"Unsupported indexed semantic mask shape: {mask.shape}")


def _label_entry_for_id(label_id: Any, label_map: Dict[str, Any]) -> Dict[str, Any]:
    """Return configured label metadata or an unknown-label placeholder."""
    labels = label_map.get("labels", {})
    entry = labels.get(label_id)
    if entry is None:
        entry = {
            "name": f"unknown_{label_id}",
            "outdoor_group": "unknown",
            "role": "ignore",
        }
    return dict(entry)


def _stats_from_bool_mask(
    label_mask: np.ndarray,
    label_id: Any,
    entry: Dict[str, Any],
    total_pixels: int,
) -> Dict[str, Any]:
    """Create one label-stat dictionary from a boolean mask."""
    ys, xs = np.nonzero(label_mask)
    pixel_count = int(label_mask.sum())
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return {
        "label_id": label_id,
        "name": str(entry.get("name", f"label_{label_id}")),
        "outdoor_group": str(entry.get("outdoor_group", "unknown")),
        "role": str(entry.get("role", "ignore")),
        "pixel_count": pixel_count,
        "area_ratio": float(pixel_count / total_pixels) if total_pixels else 0.0,
        "bbox": [x1, y1, x2, y2],
        "center": [float(xs.mean()), float(ys.mean())],
    }


def _parse_indexed_labels(mask: np.ndarray, label_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse label statistics from an indexed mask."""
    indexed = _indexed_mask_array(mask)
    total_pixels = int(indexed.size)
    labels: List[Dict[str, Any]] = []
    for raw_label_id in np.unique(indexed):
        label_id = int(raw_label_id)
        label_mask = indexed == raw_label_id
        if not np.any(label_mask):
            continue
        entry = _label_entry_for_id(label_id, label_map)
        labels.append(_stats_from_bool_mask(label_mask, label_id, entry, total_pixels))
    return labels


def _parse_color_labels(mask: np.ndarray, label_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse label statistics from an RGB color mask."""
    if mask.ndim != 3 or mask.shape[2] < 3:
        raise ValueError(f"Color semantic masks must be HxWx3 arrays, got {mask.shape}")
    rgb = mask[..., :3]
    total_pixels = int(rgb.shape[0] * rgb.shape[1])
    labels: List[Dict[str, Any]] = []
    seen_colors: set[Tuple[int, int, int]] = set()
    for label_id, entry in label_map.get("labels", {}).items():
        color = entry.get("color")
        if color is None:
            continue
        color_tuple = tuple(int(channel) for channel in color[:3])
        seen_colors.add(color_tuple)
        label_mask = np.all(rgb == np.asarray(color_tuple, dtype=rgb.dtype), axis=-1)
        if not np.any(label_mask):
            continue
        labels.append(_stats_from_bool_mask(label_mask, label_id, entry, total_pixels))

    flattened = rgb.reshape(-1, 3)
    unique_colors = {tuple(int(v) for v in color) for color in np.unique(flattened, axis=0)}
    for color_tuple in sorted(unique_colors.difference(seen_colors)):
        label_mask = np.all(rgb == np.asarray(color_tuple, dtype=rgb.dtype), axis=-1)
        if not np.any(label_mask):
            continue
        entry = {
            "name": f"unknown_color_{color_tuple[0]}_{color_tuple[1]}_{color_tuple[2]}",
            "outdoor_group": "unknown",
            "role": "ignore",
        }
        labels.append(_stats_from_bool_mask(label_mask, list(color_tuple), entry, total_pixels))
    return labels


def parse_mask_labels(mask: np.ndarray, label_map: Dict[str, Any]) -> Dict[str, Any]:
    """Parse semantic mask labels into pixel counts, bboxes, centers, and ratios."""
    normalized_label_map = _normalize_label_map(label_map)
    if mask.size == 0:
        raise ValueError("Semantic mask is empty")
    if normalized_label_map["mask_type"] == "color":
        labels = _parse_color_labels(mask, normalized_label_map)
        height, width = int(mask.shape[0]), int(mask.shape[1])
    else:
        labels = _parse_indexed_labels(mask, normalized_label_map)
        indexed = _indexed_mask_array(mask)
        height, width = int(indexed.shape[0]), int(indexed.shape[1])
    labels = sorted(labels, key=lambda item: (-int(item["pixel_count"]), str(item["name"])))
    return {"labels": labels, "height": height, "width": width}
