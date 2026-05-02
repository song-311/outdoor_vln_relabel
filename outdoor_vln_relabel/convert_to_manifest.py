"""CLI for converting raw image+pose datasets into unified manifests."""

from __future__ import annotations

import argparse

from outdoor_vln_relabel.dataset_adapters.configurable_manifest_converter import (
    convert_dataset_to_manifest,
)
from outdoor_vln_relabel.dataset_adapters.dataset_inspect import inspect_manifest
from outdoor_vln_relabel.dataset_adapters.folder_pose_adapter import (
    convert_folder_pose_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert datasets to Outdoor-VLN manifest format.")
    parser.add_argument(
        "--mode",
        choices=["config", "folder_pose"],
        default="config",
        help="Conversion mode",
    )
    parser.add_argument("--config", default=None, help="Dataset conversion YAML config")
    parser.add_argument("--input_root", default=None, help="Input root for folder_pose mode")
    parser.add_argument("--output_dir", required=True, help="Output manifest directory")
    parser.add_argument("--scene_id", default="scene_001", help="Scene id for folder_pose mode")
    parser.add_argument("--sequence_id", default="seq_001", help="Sequence id for folder_pose mode")
    parser.add_argument(
        "--default_terrain",
        default="dirt_trail",
        help="Default terrain for folder_pose mode",
    )
    parser.add_argument(
        "--copy_files",
        action="store_true",
        help="Copy images/masks into the manifest directory when supported",
    )
    return parser.parse_args()


def _print_summary(summary: dict) -> None:
    """Print a compact inspect summary."""
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    """Run conversion and print manifest inspection summary."""
    args = parse_args()
    if args.mode == "config":
        if not args.config:
            raise ValueError("--mode config requires --config")
        result = convert_dataset_to_manifest(
            args.config, output_dir=args.output_dir, copy_files=args.copy_files
        )
    else:
        if not args.input_root:
            raise ValueError("--mode folder_pose requires --input_root")
        result = convert_folder_pose_dataset(
            args.input_root,
            args.output_dir,
            scene_id=args.scene_id,
            sequence_id=args.sequence_id,
            default_terrain=args.default_terrain,
        )
    print(f"Wrote manifest to {result['output_dir']}")
    print("Inspect summary:")
    _print_summary(inspect_manifest(str(result["output_dir"])))


if __name__ == "__main__":
    main()

