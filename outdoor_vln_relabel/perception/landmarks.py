"""Rule-based landmark interfaces for Stage-3 Outdoor-VLN relabeling."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from outdoor_vln_relabel.schemas import Landmark, PathSegment


def _default_vocab_path() -> Path:
    """Return the package-default landmark vocabulary path."""
    return Path(__file__).resolve().parents[1] / "configs" / "landmark_vocab.yaml"


def _clean_name(name: str) -> str:
    """Normalize landmark text for lookup."""
    cleaned = " ".join(str(name).strip().lower().replace("_", " ").split())
    for article in ("the ", "a ", "an "):
        if cleaned.startswith(article):
            cleaned = cleaned[len(article) :]
    return cleaned


def _singularish(name: str) -> str:
    """Return a simple singular form for common generated landmark plurals."""
    name = _clean_name(name)
    replacements = {
        "bushes": "bush",
        "rocks": "rock",
        "puddles": "puddle",
        "trees": "large tree",
    }
    return replacements.get(name, name)


def _load_landmark_vocab_fallback(path: Path) -> Dict[str, Any]:
    """Load the Stage-3 landmark vocabulary without a YAML dependency."""
    vocab: Dict[str, Any] = {"landmark_categories": {}, "relations": [], "roles": []}
    section: Optional[str] = None
    category: Optional[str] = None
    in_names = False

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            if indent == 0 and stripped.endswith(":"):
                section = stripped[:-1]
                category = None
                in_names = False
                if section == "landmark_categories":
                    vocab["landmark_categories"] = {}
                elif section in {"relations", "roles"}:
                    vocab[section] = []
                else:
                    raise ValueError(
                        f"Unknown top-level landmark vocab section '{section}' "
                        f"on line {line_number}"
                    )
                continue

            if section == "landmark_categories":
                if indent == 2 and stripped.endswith(":"):
                    category = stripped[:-1]
                    vocab["landmark_categories"][category] = {
                        "default_role": "go_toward",
                        "names": [],
                    }
                    in_names = False
                    continue
                if category is None:
                    raise ValueError(
                        f"Landmark vocab entry outside category on line {line_number}"
                    )
                if indent == 4 and stripped.startswith("default_role:"):
                    vocab["landmark_categories"][category]["default_role"] = (
                        stripped.split(":", 1)[1].strip()
                    )
                    continue
                if indent == 4 and stripped == "names:":
                    in_names = True
                    continue
                if indent == 6 and in_names and stripped.startswith("- "):
                    vocab["landmark_categories"][category]["names"].append(
                        stripped[2:].strip()
                    )
                    continue

            if (
                section in {"relations", "roles"}
                and indent == 2
                and stripped.startswith("- ")
            ):
                vocab[section].append(stripped[2:].strip())
                continue

            raise ValueError(f"Invalid landmark vocab line {line_number}: {raw_line}")

    return vocab


def _validate_vocab(vocab: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the landmark vocabulary structure."""
    categories = vocab.get("landmark_categories")
    if not isinstance(categories, dict):
        raise ValueError("Landmark vocab must define landmark_categories")
    required = {
        "path_like",
        "obstacle_like",
        "object_like",
        "region_like",
        "geometry_like",
    }
    missing = sorted(required.difference(categories))
    if missing:
        raise ValueError(f"Landmark vocab missing categories: {', '.join(missing)}")
    for category, data in categories.items():
        if not isinstance(data, dict):
            raise ValueError(f"Invalid landmark category entry: {category}")
        if "default_role" not in data:
            raise ValueError(f"Landmark category missing default_role: {category}")
        if not isinstance(data.get("names"), list):
            raise ValueError(f"Landmark category names must be a list: {category}")
    if not isinstance(vocab.get("relations"), list):
        raise ValueError("Landmark vocab must define relations")
    if not isinstance(vocab.get("roles"), list):
        raise ValueError("Landmark vocab must define roles")
    return vocab


def load_landmark_vocab(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the landmark vocabulary from YAML."""
    vocab_path = Path(path) if path else _default_vocab_path()
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Landmark vocab file does not exist: {vocab_path}")
    try:
        import yaml
    except ImportError:
        vocab = _load_landmark_vocab_fallback(vocab_path)
    else:
        with vocab_path.open("r", encoding="utf-8") as file:
            vocab = yaml.safe_load(file) or {}
    return _validate_vocab(vocab)


def normalize_landmark_name(name: str, vocab: Dict[str, Any]) -> str:
    """Return the canonical vocab name for a landmark when one exists."""
    target = _singularish(name)
    for category_data in vocab.get("landmark_categories", {}).values():
        for candidate in category_data.get("names", []):
            if target == _singularish(candidate):
                return candidate
    return str(name)


def infer_landmark_category(name: str, vocab: Dict[str, Any]) -> str:
    """Infer a landmark category from the vocabulary or return unknown."""
    target = _singularish(name)
    for category, category_data in vocab.get("landmark_categories", {}).items():
        for candidate in category_data.get("names", []):
            if target == _singularish(candidate):
                return category
    return "unknown"


def get_default_role_for_category(category: str, vocab: Dict[str, Any]) -> str:
    """Return the default role for a landmark category."""
    category_data = vocab.get("landmark_categories", {}).get(category)
    if not isinstance(category_data, dict):
        return "go_toward"
    return str(category_data.get("default_role", "go_toward"))


def _landmark(
    name: str,
    category: str,
    role: str,
    relation: str,
    score: float,
    vocab: Dict[str, Any],
) -> Landmark:
    """Create a normalized Landmark record."""
    normalized_name = normalize_landmark_name(name, vocab)
    inferred = infer_landmark_category(normalized_name, vocab)
    if inferred != "unknown":
        category = inferred
    return Landmark(
        name=normalized_name,
        category=category,
        role=role,
        relation=relation,
        bbox=None,
        score=score,
    )


def _opposite_side(motion: str, default: str = "right") -> str:
    """Return a stable obstacle side opposite the turn direction."""
    if motion in {"forward_left", "turn_left"}:
        return "right"
    if motion in {"forward_right", "turn_right"}:
        return "left"
    return default


def detect_landmarks_dummy(
    segment: PathSegment,
    terrain: str = "dirt_trail",
    vocab: Optional[Dict[str, Any]] = None,
) -> List[Landmark]:
    """Generate plausible dummy landmarks from terrain and coarse motion."""
    vocab = vocab or load_landmark_vocab()
    motion = segment.motion
    landmarks: List[Landmark] = []

    if terrain == "dirt_trail":
        landmarks.append(
            _landmark("dirt trail", "path_like", "follow", "ahead", 0.9, vocab)
        )
        if motion in {"forward_left", "turn_left", "forward_right", "turn_right"}:
            landmarks.append(
                _landmark(
                    "bushes",
                    "obstacle_like",
                    "avoid",
                    _opposite_side(motion),
                    0.72,
                    vocab,
                )
            )
        return landmarks

    if terrain == "grass":
        landmarks.append(
            _landmark("grass field", "region_like", "go_toward", "ahead", 0.84, vocab)
        )
        if motion in {"forward_left", "turn_left"}:
            landmarks.append(
                _landmark("open area", "region_like", "go_toward", "left", 0.68, vocab)
            )
        elif motion in {"forward_right", "turn_right"}:
            landmarks.append(
                _landmark(
                    "large tree", "object_like", "go_toward", "right", 0.66, vocab
                )
            )
        return landmarks

    if terrain == "vegetation":
        landmarks.append(
            _landmark(
                "narrow passage", "geometry_like", "pass_between", "ahead", 0.82, vocab
            )
        )
        landmarks.append(
            _landmark(
                "bushes",
                "obstacle_like",
                "avoid",
                _opposite_side(motion, default="left"),
                0.76,
                vocab,
            )
        )
        return landmarks

    if terrain == "mud_water":
        avoid_relation = (
            "front_left" if motion in {"forward_right", "turn_right"} else "front_right"
        )
        if motion == "forward":
            avoid_relation = "ahead"
        landmarks.append(
            _landmark("open path", "path_like", "follow", "ahead", 0.78, vocab)
        )
        landmarks.append(
            _landmark("puddle", "obstacle_like", "avoid", avoid_relation, 0.86, vocab)
        )
        return landmarks

    if terrain == "rough_terrain":
        landmarks.append(
            _landmark("rocky path", "path_like", "follow", "ahead", 0.78, vocab)
        )
        landmarks.append(
            _landmark(
                "rocks",
                "obstacle_like",
                "avoid",
                _opposite_side(motion, default="left"),
                0.7,
                vocab,
            )
        )
        return landmarks

    landmarks.append(
        _landmark("open path", "path_like", "follow", "ahead", 0.6, vocab)
    )
    return landmarks


def _metadata_landmark_items(frame_landmarks: Any) -> Iterable[Dict[str, Any]]:
    """Yield landmark dictionaries from a frame metadata value."""
    if not frame_landmarks:
        return []
    if isinstance(frame_landmarks, list):
        return [item for item in frame_landmarks if isinstance(item, dict)]
    return []


def _default_relation(relation: Any) -> str:
    """Return a valid default relation."""
    if relation:
        return str(relation)
    return "ahead"


def _default_role(name: str, category: str, role: Any, vocab: Dict[str, Any]) -> str:
    """Return metadata role or infer a safe default role."""
    if role:
        return str(role)
    if category == "obstacle_like":
        return "avoid"
    if category == "path_like":
        return "follow"
    lowered = _clean_name(name)
    if category == "geometry_like" and ("narrow" in lowered or "gap" in lowered):
        return "pass_between"
    return get_default_role_for_category(category, vocab)


def detect_landmarks_from_metadata(
    frames: List[Any],
    start_idx: int,
    end_idx: int,
    vocab: Dict[str, Any],
) -> List[Landmark]:
    """Merge frame-level landmark metadata into segment-level landmarks."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for frame in frames[start_idx : end_idx + 1]:
        for item in _metadata_landmark_items(getattr(frame, "landmarks", None)):
            name = item.get("name")
            if not name:
                continue
            normalized_name = normalize_landmark_name(str(name), vocab)
            grouped[_clean_name(normalized_name)].append({**item, "name": normalized_name})

    landmarks: List[Landmark] = []
    for items in grouped.values():
        best = max(items, key=lambda item: float(item.get("score", 1.0)))
        name = str(best["name"])
        category = str(best.get("category") or infer_landmark_category(name, vocab))
        if category == "unknown":
            category = infer_landmark_category(name, vocab)
        role = _default_role(name, category, best.get("role"), vocab)
        relation = _default_relation(best.get("relation"))
        score = sum(float(item.get("score", 1.0)) for item in items) / len(items)
        landmarks.append(
            Landmark(
                name=name,
                category=category,
                role=role,
                relation=relation,
                bbox=best.get("bbox"),
                score=round(score, 6),
            )
        )
    return sorted(landmarks, key=lambda landmark: -landmark.score)


def _append_unique_landmark(
    landmarks: List[Landmark],
    name: str,
    category: str,
    role: str,
    relation: str,
    score: float,
    vocab: Dict[str, Any],
) -> None:
    """Append a landmark unless a same-name landmark already exists."""
    normalized = normalize_landmark_name(name, vocab)
    if any(_clean_name(landmark.name) == _clean_name(normalized) for landmark in landmarks):
        return
    landmarks.append(_landmark(normalized, category, role, relation, score, vocab))


def detect_landmarks_from_semantic_mask_dummy(
    frames: List[Any],
    start_idx: int,
    end_idx: int,
    terrain: str,
    vocab: Dict[str, Any],
) -> List[Landmark]:
    """Infer landmarks from semantic mask filenames using lightweight heuristics."""
    landmarks: List[Landmark] = []
    for frame in frames[start_idx : end_idx + 1]:
        semantic_path = getattr(frame, "semantic_path", None)
        if not semantic_path:
            continue
        name = Path(str(semantic_path)).name.lower()
        if any(token in name for token in ("bush", "vegetation", "shrub")):
            _append_unique_landmark(
                landmarks, "bushes", "obstacle_like", "avoid", "left", 0.68, vocab
            )
        if any(token in name for token in ("puddle", "mud", "water", "wet")):
            _append_unique_landmark(
                landmarks, "puddle", "obstacle_like", "avoid", "ahead", 0.74, vocab
            )
        if any(token in name for token in ("rock", "gravel")):
            _append_unique_landmark(
                landmarks, "rocks", "obstacle_like", "avoid", "right", 0.66, vocab
            )
            if terrain == "rough_terrain":
                _append_unique_landmark(
                    landmarks, "rocky path", "path_like", "follow", "ahead", 0.62, vocab
                )
        if any(token in name for token in ("trail", "path", "dirt")):
            _append_unique_landmark(
                landmarks, "dirt trail", "path_like", "follow", "ahead", 0.7, vocab
            )
    return landmarks
