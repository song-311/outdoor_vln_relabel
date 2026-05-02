"""Dataset balancing placeholders."""

from __future__ import annotations

from typing import Any, Iterable, List


def balance_by_scene_and_terrain(records: Iterable[Any], target_total: int) -> List[Any]:
    """Return balanced records by scene and terrain.

    Stage 1 does not downsample or upsample yet. The target_total argument is
    reserved for the future 10-scene, 5-terrain, 100k-sample balancing stage.
    """
    return list(records)

