"""BotanicGarden dataset adapter placeholder."""

from __future__ import annotations

from typing import Tuple

from .base_adapter import BaseDatasetAdapter


class BotanicGardenAdapter(BaseDatasetAdapter):
    """Future adapter for the BotanicGarden outdoor robot dataset."""

    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Raise a clear error until BotanicGarden parsing is implemented."""
        raise NotImplementedError(
            "BotanicGardenAdapter is reserved for future BotanicGarden support "
            "and is not implemented in Stage 1"
        )

