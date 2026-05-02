"""Terrain classification interfaces for Stage-2 relabeling.

This module intentionally avoids model dependencies. It supports taxonomy-backed
normalization from metadata and stable placeholder behavior for future
segmentation adapters such as RELLIS-3D, RUGD, and WildScenes.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

VALID_TERRAINS = {
    "dirt_trail",
    "grass",
    "vegetation",
    "mud_water",
    "rough_terrain",
}


def _default_taxonomy_path() -> Path:
    """Return the package-default terrain taxonomy path."""
    return Path(__file__).resolve().parents[1] / "configs" / "terrain_taxonomy.yaml"


def _parse_scalar(value: str) -> Any:
    """Parse the small YAML scalar subset used by taxonomy files."""
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
            raise ValueError(f"Invalid inline list value in terrain taxonomy: {value}") from exc
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load a minimal indentation-based YAML mapping without third-party deps."""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                raise ValueError(
                    "Simple terrain taxonomy loader only supports mappings and "
                    f"inline lists; found block list on line {line_number} of {path}"
                )
            if ":" not in stripped:
                raise ValueError(f"Invalid taxonomy line {line_number} in {path}: {raw_line}")
            indent = len(line) - len(line.lstrip(" "))
            key, raw_value = stripped.split(":", 1)
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if raw_value.strip() == "":
                child: Dict[str, Any] = {}
                parent[key.strip()] = child
                stack.append((indent, child))
            else:
                parent[key.strip()] = _parse_scalar(raw_value)
    return root


def load_terrain_taxonomy(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the terrain taxonomy from YAML and validate its five classes."""
    taxonomy_path = Path(path) if path else _default_taxonomy_path()
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Terrain taxonomy file does not exist: {taxonomy_path}")
    try:
        import yaml
    except ImportError:
        taxonomy = _load_simple_yaml_mapping(taxonomy_path)
    else:
        with taxonomy_path.open("r", encoding="utf-8") as file:
            taxonomy = yaml.safe_load(file) or {}
    terrain_classes = taxonomy.get("terrain_classes")
    if not isinstance(terrain_classes, dict):
        raise ValueError(f"Terrain taxonomy must define terrain_classes: {taxonomy_path}")
    missing = sorted(VALID_TERRAINS.difference(terrain_classes))
    if missing:
        raise ValueError(
            f"Terrain taxonomy missing required classes: {', '.join(missing)}"
        )
    return taxonomy


def _canonical_key(name: str) -> str:
    """Normalize a terrain or alias string for matching."""
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _safe_default(default: str, taxonomy: Dict[str, Any]) -> str:
    """Normalize a default terrain and fall back to dirt_trail if invalid."""
    normalized = normalize_terrain_name(default, taxonomy)
    if normalized in VALID_TERRAINS:
        return normalized
    return "dirt_trail"


def normalize_terrain_name(name: str, taxonomy: Dict[str, Any]) -> str:
    """Normalize a terrain class name or alias to one of the five class names.

    Unknown values are returned unchanged so callers can decide whether to fall
    back to a default terrain or surface an error.
    """
    if name is None:
        return ""
    terrain_classes = taxonomy.get("terrain_classes", {})
    direct = str(name).strip()
    if direct in terrain_classes and direct in VALID_TERRAINS:
        return direct
    target = _canonical_key(direct)
    for terrain, properties in terrain_classes.items():
        if terrain in VALID_TERRAINS and target == _canonical_key(terrain):
            return terrain
        aliases = properties.get("aliases", []) if isinstance(properties, dict) else []
        for alias in aliases:
            if target == _canonical_key(str(alias)):
                return terrain
    return direct


def classify_terrain_from_metadata(
    metadata: Mapping[str, Any] | None, default: str = "dirt_trail"
) -> str:
    """Return a normalized terrain label from metadata or a safe default.

    Unknown metadata values do not crash the pipeline; they return the requested
    default when it is valid, otherwise dirt_trail.
    """
    taxonomy = load_terrain_taxonomy()
    fallback = _safe_default(default, taxonomy)
    if not metadata:
        return fallback
    terrain = (
        metadata.get("terrain")
        or metadata.get("terrain_label")
        or metadata.get("terrain_class")
    )
    if terrain is None:
        return fallback
    normalized = normalize_terrain_name(str(terrain), taxonomy)
    if normalized not in VALID_TERRAINS:
        return fallback
    return normalized


def classify_terrain_dummy(segment: Any, default: str = "dirt_trail") -> str:
    """Return a normalized placeholder terrain label for a segment."""
    taxonomy = load_terrain_taxonomy()
    return _safe_default(default, taxonomy)


def classify_terrain_from_frames(
    frames: List[Any],
    start_idx: int,
    end_idx: int,
    default: str = "dirt_trail",
    taxonomy: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify segment terrain by majority vote over frame-level metadata."""
    taxonomy = taxonomy or load_terrain_taxonomy()
    fallback = _safe_default(default, taxonomy)
    labels = []
    for frame in frames[start_idx : end_idx + 1]:
        terrain = getattr(frame, "terrain", None)
        if not terrain:
            metadata = getattr(frame, "metadata", None) or {}
            terrain = (
                metadata.get("terrain")
                or metadata.get("terrain_label")
                or metadata.get("terrain_class")
            )
        if terrain:
            normalized = normalize_terrain_name(str(terrain), taxonomy)
            if normalized in VALID_TERRAINS:
                labels.append(normalized)
    if not labels:
        return fallback
    return Counter(labels).most_common(1)[0][0]


def classify_terrain_from_semantic_mask_dummy(
    mask_path: str, taxonomy: Dict[str, Any]
) -> Optional[str]:
    """Infer terrain from a semantic mask filename using lightweight heuristics."""
    name = Path(str(mask_path)).name.lower()
    if any(token in name for token in ("grass", "meadow")):
        return "grass"
    if any(token in name for token in ("mud", "puddle", "water", "wet")):
        return "mud_water"
    if any(token in name for token in ("vegetation", "bush", "shrub")):
        return "vegetation"
    if any(token in name for token in ("rock", "gravel", "rough", "slope")):
        return "rough_terrain"
    if any(token in name for token in ("trail", "dirt", "path", "road")):
        return "dirt_trail"
    return None


def get_terrain_properties(terrain: str, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """Return taxonomy properties for a normalized terrain class."""
    normalized = normalize_terrain_name(terrain, taxonomy)
    if normalized not in VALID_TERRAINS:
        raise ValueError(f"Unknown terrain '{terrain}'")
    terrain_classes = taxonomy.get("terrain_classes", {})
    properties = terrain_classes.get(normalized, {})
    if not isinstance(properties, dict):
        raise ValueError(f"Invalid terrain properties for '{normalized}'")
    return dict(properties)
