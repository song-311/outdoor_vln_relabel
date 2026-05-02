"""RELLIS-3D dataset adapter placeholder."""

from __future__ import annotations

from typing import Tuple

from .base_adapter import BaseDatasetAdapter


class RellisAdapter(BaseDatasetAdapter):
    """Future adapter for RELLIS-3D off-road robot data."""

    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Raise a clear error until RELLIS-3D parsing is implemented."""
        raise NotImplementedError(
            "RellisAdapter is reserved for future RELLIS-3D support and is not "
            "implemented in Stage 1"
        )

