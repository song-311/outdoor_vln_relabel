"""I/O helpers for JSONL-based Outdoor-VLN dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _record_to_dict(record: Any) -> Any:
    """Convert dataclass-like records to dictionaries before JSON encoding."""
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return record


def save_jsonl(records: Iterable[Any], path: str | Path) -> None:
    """Save records to a UTF-8 JSONL file, creating the parent directory."""
    output_path = Path(path)
    if output_path.parent and str(output_path.parent) != ".":
        ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(_record_to_dict(record), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def load_jsonl(path: str | Path) -> List[Any]:
    """Load a JSONL file into a list of dictionaries."""
    input_path = Path(path)
    records: List[Any] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}: {exc}"
                ) from exc
    return records

