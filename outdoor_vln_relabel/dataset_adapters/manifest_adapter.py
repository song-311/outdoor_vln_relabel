"""Manifest adapter for frame-level outdoor robot datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from outdoor_vln_relabel.schemas import FrameRecord

from .base_adapter import BaseDatasetAdapter

REQUIRED_TRAJECTORY_COLUMNS = {"frame_id", "timestamp", "rgb_path", "x", "y", "yaw"}


def _resolve_dataset_path(dataset_root: Path, value: str | None) -> Optional[str]:
    """Resolve a manifest path relative to dataset_root when needed."""
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    path = Path(stripped)
    if not path.is_absolute():
        path = dataset_root / path
    return str(path)


def _required_path(dataset_root: Path, value: str | None, field_name: str, row_number: int) -> str:
    """Resolve a required path field or raise a row-specific error."""
    resolved = _resolve_dataset_path(dataset_root, value)
    if resolved is None:
        raise ValueError(
            f"trajectory.csv row {row_number} missing required path field '{field_name}'"
        )
    return resolved


def _parse_json_field(value: str | None, field_name: str, row_number: int) -> Any:
    """Parse an optional JSON field with row-specific error context."""
    if value is None or not str(value).strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in trajectory.csv row {row_number} field '{field_name}': {exc}"
        ) from exc


def _load_global_metadata(dataset_root: Path) -> Dict[str, Any]:
    """Load optional dataset-level metadata.json."""
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"metadata.json must contain an object: {metadata_path}")
    return data


def _frame_metadata(
    global_metadata: Dict[str, Any], frame_id: int, row_metadata: Any
) -> Optional[Dict[str, Any]]:
    """Merge optional global frame metadata with row-level metadata."""
    merged: Dict[str, Any] = {}
    frame_metadata = global_metadata.get("frames", {})
    if isinstance(frame_metadata, dict):
        item = frame_metadata.get(str(frame_id)) or frame_metadata.get(frame_id)
        if isinstance(item, dict):
            merged.update(item)
    if isinstance(row_metadata, dict):
        merged.update(row_metadata)
    return merged or None


class ManifestDatasetAdapter(BaseDatasetAdapter):
    """Adapter for dataset_root/trajectory.csv manifest datasets."""

    def __init__(
        self, input_path: str | Path, scene_id: str, sequence_id: str = "default_sequence"
    ) -> None:
        """Store the dataset root for a manifest dataset."""
        super().__init__(input_path, scene_id, sequence_id)
        self.dataset_root = self.input_path
        self._frames: Optional[List[FrameRecord]] = None

    @property
    def trajectory_csv(self) -> Path:
        """Return the expected trajectory.csv path."""
        return self.dataset_root / "trajectory.csv"

    def load_frames(self) -> List[FrameRecord]:
        """Load FrameRecord objects from trajectory.csv."""
        if self._frames is not None:
            return self._frames
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Manifest dataset_root does not exist: {self.dataset_root}")
        if not self.trajectory_csv.is_file():
            raise FileNotFoundError(f"Missing manifest trajectory file: {self.trajectory_csv}")

        global_metadata = _load_global_metadata(self.dataset_root)
        frames: List[FrameRecord] = []
        with self.trajectory_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"trajectory.csv has no header: {self.trajectory_csv}")
            missing = REQUIRED_TRAJECTORY_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    "trajectory.csv missing required columns: "
                    + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    frame_id = int(row["frame_id"])
                    timestamp = float(row["timestamp"])
                    x = float(row["x"])
                    y = float(row["y"])
                    yaw = float(row["yaw"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value in trajectory.csv row {row_number}: {exc}"
                    ) from exc

                landmarks = _parse_json_field(row.get("landmarks"), "landmarks", row_number)
                if landmarks is not None and not isinstance(landmarks, list):
                    raise ValueError(
                        f"trajectory.csv row {row_number} landmarks must be a JSON list"
                    )
                row_metadata = _parse_json_field(row.get("metadata"), "metadata", row_number)

                frames.append(
                    FrameRecord(
                        scene_id=self.scene_id,
                        sequence_id=self.sequence_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        rgb_path=_required_path(
                            self.dataset_root, row.get("rgb_path"), "rgb_path", row_number
                        ),
                        depth_path=_resolve_dataset_path(
                            self.dataset_root, row.get("depth_path")
                        ),
                        lidar_path=_resolve_dataset_path(
                            self.dataset_root, row.get("lidar_path")
                        ),
                        semantic_path=_resolve_dataset_path(
                            self.dataset_root, row.get("semantic_path")
                        ),
                        position=[x, y],
                        yaw=yaw,
                        cmd_vel=None,
                        terrain=(row.get("terrain") or None),
                        landmarks=landmarks,
                        metadata=_frame_metadata(global_metadata, frame_id, row_metadata),
                    )
                )
        if not frames:
            raise ValueError(f"trajectory.csv contains no frames: {self.trajectory_csv}")
        self._frames = frames
        return frames

    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Return positions, yaws, and timestamps from manifest frames."""
        frames = self.load_frames()
        positions = [[frame.position[0], frame.position[1]] for frame in frames]
        yaws = [frame.yaw for frame in frames]
        timestamps = [frame.timestamp for frame in frames]
        return positions, yaws, timestamps

    def load_terrain_labels(self) -> Optional[object]:
        """Return per-frame terrain labels when present in trajectory.csv."""
        frames = self.load_frames()
        labels = [frame.terrain for frame in frames]
        if not any(label for label in labels):
            return None
        return labels
