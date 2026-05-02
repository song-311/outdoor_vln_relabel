"""TartanGround dataset adapter placeholder."""

from __future__ import annotations

from typing import Tuple

from .base_adapter import BaseDatasetAdapter


class TartanGroundAdapter(BaseDatasetAdapter):
    """Future adapter for TartanGround simulation data."""

    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Raise a clear error until TartanGround parsing is implemented."""
        raise NotImplementedError(
            "TartanGroundAdapter is reserved for future TartanGround support and "
            "is not implemented in Stage 1"
        )

