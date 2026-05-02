"""ROS bag adapter placeholder."""

from __future__ import annotations

from typing import Tuple

from .base_adapter import BaseDatasetAdapter


class RosbagAdapter(BaseDatasetAdapter):
    """Future adapter for ROS bag recordings with RGB, depth, odom, and LiDAR."""

    def load_trajectory_arrays(self) -> Tuple[object, object, object]:
        """Raise a clear error until ROS bag parsing is implemented."""
        raise NotImplementedError(
            "RosbagAdapter is reserved for future ROS bag support and is not "
            "implemented in Stage 1"
        )

