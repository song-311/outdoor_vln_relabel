"""Prepare a real Outdoor-VLN pilot manifest from a configurable raw dataset."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from outdoor_vln_relabel.dataset_adapters.raw_dataset_inspect import inspect_raw_dataset
from outdoor_vln_relabel.dataset_adapters.real_pilot_converter import (
    convert_real_pilot_to_manifest,
)


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a readable summary dictionary."""
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect or convert a real RELLIS-style pilot dataset."
    )
    parser.add_argument("--config", required=True, help="Real pilot YAML config")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output manifest directory. Defaults to config dataset.output_dir.",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect paths, pose columns, and frame matching",
    )
    parser.add_argument(
        "--allow_missing_masks",
        action="store_true",
        help="Allow conversion when semantic masks are missing",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional pilot frame cap overriding config pilot.max_frames",
    )
    return parser.parse_args()


def main() -> None:
    """Run raw inspection or manifest conversion."""
    args = parse_args()
    summary = inspect_raw_dataset(args.config)
    if args.inspect_only:
        _print_summary(summary)
        return

    result = convert_real_pilot_to_manifest(
        args.config,
        output_dir=args.output_dir,
        allow_missing_masks=args.allow_missing_masks,
        max_frames=args.max_frames,
    )
    print("Raw dataset inspect summary:")
    _print_summary(summary)
    print("Manifest conversion result:")
    _print_summary(result)


if __name__ == "__main__":
    main()
