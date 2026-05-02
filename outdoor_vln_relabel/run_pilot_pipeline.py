"""Run the end-to-end Outdoor-VLN pilot pipeline on a manifest dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from outdoor_vln_relabel.analysis.report import write_markdown_report
from outdoor_vln_relabel.analysis.stats import compute_basic_stats, find_potential_issues
from outdoor_vln_relabel.analysis.visualize import make_sample_card
from outdoor_vln_relabel.build_dataset import _build_manifest_pairs
from outdoor_vln_relabel.dataset_adapters.manifest_adapter import ManifestDatasetAdapter
from outdoor_vln_relabel.io_utils import ensure_dir, save_jsonl
from outdoor_vln_relabel.rewrite.rewrite_dataset import rewrite_records


def _load_manifest_info(manifest_root: str) -> dict:
    """Load manifest_info.json when present."""
    info_path = Path(manifest_root) / "manifest_info.json"
    if not info_path.is_file():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_sequence_id(manifest_root: str) -> str:
    """Load sequence_id from manifest_info.json when present."""
    info = _load_manifest_info(manifest_root)
    if not info:
        return "default_sequence"
    return str(info.get("sequence_id") or "default_sequence")


def _resolve_semantic_label_map(manifest_root: str, cli_value: str | None) -> str | None:
    """Resolve CLI or manifest-provided semantic label-map path."""
    if cli_value:
        return cli_value
    info = _load_manifest_info(manifest_root)
    value = str(info.get("semantic_label_map") or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_file() or path.is_absolute():
        return str(path)
    manifest_relative = Path(manifest_root) / path
    if manifest_relative.is_file():
        return str(manifest_relative)
    return value


def _sample_records(records: List[dict], num_samples: int, seed: int) -> List[Tuple[int, dict]]:
    """Return deterministic random sample records."""
    indexed = list(enumerate(records))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    return indexed[: max(0, min(num_samples, len(indexed)))]


def _landmarks_text(record: dict) -> str:
    """Format landmarks for Markdown."""
    items = []
    for landmark in record.get("landmarks") or []:
        if isinstance(landmark, dict):
            items.append(
                f"{landmark.get('name')} ({landmark.get('role')}, {landmark.get('relation')})"
            )
    return ", ".join(items) if items else "none"


def _write_visualizations(records: List[dict], output_dir: Path, num_samples: int, seed: int) -> Path:
    """Write sample cards and a Markdown index."""
    ensure_dir(output_dir)
    samples = _sample_records(records, num_samples, seed)
    lines = ["# Outdoor-VLN Pilot Sample Visualizations", ""]
    for sample_number, (index, record) in enumerate(samples, start=1):
        image_name = f"sample_{sample_number:03d}_idx_{index:06d}.png"
        image_path = output_dir / image_name
        frame_paths = [
            path for path in [record.get("start_frame"), record.get("goal_frame")] if path
        ]
        make_sample_card(record, image_paths=frame_paths, output_path=str(image_path))
        lines.extend(
            [
                f"## Sample {index}",
                "",
                f"![sample]({image_name})",
                "",
                f"- instruction: {record.get('instruction', '')}",
                f"- terrain: {record.get('terrain')}",
                f"- motion: {record.get('motion')}",
                f"- confidence: {record.get('confidence')}",
                f"- landmarks: {_landmarks_text(record)}",
                "",
            ]
        )
    samples_md = output_dir / "samples.md"
    samples_md.write_text("\n".join(lines), encoding="utf-8")
    return samples_md


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Outdoor-VLN pilot pipeline.")
    parser.add_argument("--manifest_root", required=True, help="Input manifest root")
    parser.add_argument("--scene_id", required=True, help="Scene id for generated pairs")
    parser.add_argument("--output_dir", required=True, help="Pilot output directory")
    parser.add_argument("--default_terrain", default="dirt_trail")
    parser.add_argument(
        "--terrain_mode",
        choices=["dummy", "metadata", "mask", "metadata_or_mask"],
        default="metadata_or_mask",
    )
    parser.add_argument(
        "--landmark_mode",
        choices=["dummy", "metadata", "mask", "metadata_or_mask", "metadata_or_dummy"],
        default="metadata_or_mask",
    )
    parser.add_argument(
        "--rewrite_backend",
        choices=["rule_based", "llm_stub"],
        default="rule_based",
    )
    parser.add_argument("--num_rewrites", type=int, default=3)
    parser.add_argument("--num_vis_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--semantic_label_map",
        default=None,
        help="Optional semantic mask label-map YAML for evidence extraction",
    )
    return parser.parse_args()


def run_pilot_pipeline(
    manifest_root: str,
    scene_id: str,
    output_dir: str,
    default_terrain: str = "dirt_trail",
    terrain_mode: str = "metadata_or_mask",
    landmark_mode: str = "metadata_or_mask",
    rewrite_backend: str = "rule_based",
    num_rewrites: int = 3,
    num_vis_samples: int = 10,
    seed: int = 42,
    semantic_label_map: str | None = None,
) -> dict:
    """Run manifest ingestion, v3 generation, rewrite, QA, and visualization."""
    output_path = ensure_dir(output_dir)
    sequence_id = _load_sequence_id(manifest_root)
    resolved_label_map = _resolve_semantic_label_map(manifest_root, semantic_label_map)
    adapter = ManifestDatasetAdapter(manifest_root, scene_id, sequence_id)
    frames = adapter.load_frames()

    pairs_v3 = _build_manifest_pairs(
        frames=frames,
        scene_id=scene_id,
        sequence_id=sequence_id,
        config={},
        num_variants=3,
        default_terrain=default_terrain,
        terrain_mode=terrain_mode,
        terrain_taxonomy=None,
        use_landmarks=True,
        landmark_mode=landmark_mode,
        landmark_vocab_path=None,
        semantic_label_map_path=resolved_label_map,
    )
    pairs_v3_records = [
        pair.to_dict() if hasattr(pair, "to_dict") else dict(pair) for pair in pairs_v3
    ]
    pairs_v3_path = output_path / "pairs_v3.jsonl"
    save_jsonl(pairs_v3_records, pairs_v3_path)

    pairs_v4 = rewrite_records(
        pairs_v3_records,
        backend=rewrite_backend,
        num_variants=num_rewrites,
        keep_original=True,
    )
    pairs_v4_path = output_path / "pairs_v4_rewritten.jsonl"
    save_jsonl(pairs_v4, pairs_v4_path)

    stats = compute_basic_stats(pairs_v4)
    issues = find_potential_issues(pairs_v4)
    report_path = output_path / "report.md"
    issues_path = output_path / "issues.jsonl"
    write_markdown_report(stats, issues, str(report_path))
    save_jsonl(issues, issues_path)

    sample_vis_dir = output_path / "sample_vis"
    samples_md = _write_visualizations(pairs_v4, sample_vis_dir, num_vis_samples, seed)

    return {
        "manifest_root": manifest_root,
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "output_dir": str(output_path),
        "pairs_v3": str(pairs_v3_path),
        "pairs_v4_rewritten": str(pairs_v4_path),
        "report": str(report_path),
        "issues": str(issues_path),
        "sample_visualizations": str(samples_md),
        "semantic_label_map": resolved_label_map,
        "num_pairs_v3": len(pairs_v3_records),
        "num_pairs_v4": len(pairs_v4),
        "num_issues": len(issues),
        "stats": stats,
    }


def main() -> None:
    """Run manifest ingestion, v3 generation, rewrite, QA, and visualization."""
    args = parse_args()
    result = run_pilot_pipeline(
        manifest_root=args.manifest_root,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
        default_terrain=args.default_terrain,
        terrain_mode=args.terrain_mode,
        landmark_mode=args.landmark_mode,
        rewrite_backend=args.rewrite_backend,
        num_rewrites=args.num_rewrites,
        num_vis_samples=args.num_vis_samples,
        seed=args.seed,
        semantic_label_map=args.semantic_label_map,
    )

    print(f"pairs_v3: {result['pairs_v3']}")
    print(f"pairs_v4_rewritten: {result['pairs_v4_rewritten']}")
    print(f"report: {result['report']}")
    print(f"issues: {result['issues']}")
    print(f"sample_visualizations: {result['sample_visualizations']}")
    if result.get("semantic_label_map"):
        print(f"semantic_label_map: {result['semantic_label_map']}")


if __name__ == "__main__":
    main()
