"""Base adapter interfaces for Outdoor-VLN source datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from outdoor_vln_relabel.schemas import FrameRecord


class BaseDatasetAdapter(ABC):
    """Common interface for source datasets that can produce frame records."""

    def __init__(
        self, input_path: str | Path, scene_id: str, sequence_id: str = "default_sequence"
    ) -> None:
        """Store adapter metadata shared by concrete dataset readers."""
        self.input_path = Path(input_path)
        self.scene_id = scene_id
        self.sequence_id = sequence_id

    def load_frames(self) -> List[FrameRecord]:
        """Load synchronized frame records when a dataset provides image assets."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.load_frames is not implemented yet"
        )

    @abstractmethod
    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Return positions, yaws, and timestamps arrays for trajectory slicing."""

    def load_terrain_labels(self) -> Optional[object]:
        """Return optional terrain labels aligned to frames or segments."""
        return None


def _resolve_npz_path(input_path: Path) -> Path:
    """Resolve either a direct npz path or a directory containing one npz file."""
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    candidates = [
        input_path / "demo_traj.npz",
        input_path / "trajectory.npz",
        input_path / "traj.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    npz_files = sorted(input_path.glob("*.npz"))
    if len(npz_files) == 1:
        return npz_files[0]
    if not npz_files:
        raise FileNotFoundError(f"No .npz trajectory file found in {input_path}")
    raise ValueError(
        f"Multiple .npz files found in {input_path}; pass one file path explicitly"
    )


class NpzTrajectoryAdapter(BaseDatasetAdapter):
    """Adapter for demo npz files with positions, yaws, and timestamps arrays."""

    REQUIRED_KEYS = ("positions", "yaws", "timestamps")

    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Load positions, yaws, and timestamps from an npz trajectory file."""
        import numpy as np

        npz_path = _resolve_npz_path(self.input_path)
        with np.load(npz_path) as data:
            missing = [key for key in self.REQUIRED_KEYS if key not in data]
            if missing:
                raise ValueError(
                    f"{npz_path} missing required arrays: {', '.join(missing)}"
                )
            return data["positions"], data["yaws"], data["timestamps"]

    def load_terrain_labels(self) -> Optional[object]:
        """Load optional terrain_labels from the npz trajectory file."""
        import numpy as np

        npz_path = _resolve_npz_path(self.input_path)
        with np.load(npz_path, allow_pickle=False) as data:
            if "terrain_labels" not in data:
                return None
            return data["terrain_labels"].copy()
