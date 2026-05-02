"""Run a batch of real-pilot Outdoor-VLN experiments."""

from __future__ import annotations

import argparse
import ast
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

from outdoor_vln_relabel.dataset_adapters.real_pilot_converter import (
    convert_real_pilot_to_manifest,
)
from outdoor_vln_relabel.export_manual_audit import export_manual_audit
from outdoor_vln_relabel.io_utils import ensure_dir
from outdoor_vln_relabel.run_pilot_pipeline import run_pilot_pipeline


def _parse_scalar(value: str) -> Any:
    """Parse a small YAML scalar subset for experiment configs."""
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_experiment_fallback(path: Path) -> Dict[str, Any]:
    """Load the bundled experiment config shape without requiring PyYAML."""
    config: Dict[str, Any] = {"experiment": {}, "pilots": []}
    section = ""
    current_pilot: Dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if indent == 0 and stripped.endswith(":"):
                section = stripped[:-1]
                if section == "experiment":
                    config["experiment"] = {}
                elif section == "pilots":
                    config["pilots"] = []
                else:
                    raise ValueError(
                        f"Unsupported experiment config section '{section}' on line {line_number}"
                    )
                current_pilot = None
                continue
            if section == "experiment" and indent == 2 and ":" in stripped:
                key, raw_value = stripped.split(":", 1)
                config["experiment"][key.strip()] = _parse_scalar(raw_value)
                continue
            if section == "pilots":
                if indent == 2 and stripped.startswith("- "):
                    current_pilot = {}
                    config["pilots"].append(current_pilot)
                    remainder = stripped[2:].strip()
                    if remainder:
                        key, raw_value = remainder.split(":", 1)
                        current_pilot[key.strip()] = _parse_scalar(raw_value)
                    continue
                if indent == 4 and current_pilot is not None and ":" in stripped:
                    key, raw_value = stripped.split(":", 1)
                    current_pilot[key.strip()] = _parse_scalar(raw_value)
                    continue
            raise ValueError(f"Invalid experiment config line {line_number}: {raw_line}")
    return config


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load a real-pilot experiment YAML config."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {path}")
    try:
        import yaml
    except ImportError:
        config = _load_experiment_fallback(path)
    else:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must contain a mapping: {path}")
    if not isinstance(config.get("pilots"), list):
        raise ValueError("Experiment config must define pilots as a list")
    return config


def _pilot_int(pilot: Dict[str, Any], key: str, default: int) -> int:
    """Return an integer pilot config value."""
    value = pilot.get(key, default)
    return default if value in (None, "") else int(value)


def _write_summary_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    """Write a compact Markdown experiment summary."""
    lines = [
        "# Outdoor-VLN Real Pilot Experiment Summary",
        "",
        f"- experiment: {summary.get('experiment_name')}",
        f"- total pilots: {summary.get('total_pilots')}",
        f"- succeeded: {summary.get('succeeded')}",
        f"- failed: {summary.get('failed')}",
        f"- total pairs v4: {summary.get('total_pairs_v4')}",
        f"- total issues: {summary.get('total_issues')}",
        "",
        "| pilot | status | pairs_v3 | pairs_v4 | issues | output |",
        "|---|---|---:|---:|---:|---|",
    ]
    for pilot in summary.get("pilots", []):
        lines.append(
            "| "
            f"{pilot.get('name')} | {pilot.get('status')} | "
            f"{pilot.get('num_pairs_v3', 0)} | {pilot.get('num_pairs_v4', 0)} | "
            f"{pilot.get('num_issues', 0)} | {pilot.get('output_dir', '')} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config_path: str) -> Dict[str, Any]:
    """Run all pilots in one experiment config and write aggregate summaries."""
    config = load_experiment_config(config_path)
    experiment = config.get("experiment", {}) or {}
    experiment_name = str(experiment.get("name", "real_pilot_experiment"))
    output_root = ensure_dir(experiment.get("output_root", f"outputs/experiments/{experiment_name}"))

    pilot_summaries: List[Dict[str, Any]] = []
    failed_pilots: List[Dict[str, Any]] = []
    for pilot in config.get("pilots", []):
        if not isinstance(pilot, dict):
            continue
        name = str(pilot.get("name") or f"pilot_{len(pilot_summaries):03d}")
        pilot_dir = ensure_dir(output_root / name)
        manifest_dir = pilot_dir / "manifest"
        try:
            manifest_result = convert_real_pilot_to_manifest(
                str(pilot.get("config")),
                output_dir=str(manifest_dir),
                allow_missing_masks=bool(pilot.get("allow_missing_masks", False)),
                max_frames=(
                    int(pilot["max_frames"])
                    if pilot.get("max_frames") not in (None, "")
                    else None
                ),
            )
            pipeline_result = run_pilot_pipeline(
                manifest_root=str(manifest_dir),
                scene_id=str(pilot.get("scene_id") or name),
                output_dir=str(pilot_dir),
                default_terrain=str(pilot.get("default_terrain", "dirt_trail")),
                terrain_mode=str(pilot.get("terrain_mode", "metadata_or_mask")),
                landmark_mode=str(pilot.get("landmark_mode", "metadata_or_mask")),
                rewrite_backend=str(pilot.get("rewrite_backend", "rule_based")),
                num_rewrites=_pilot_int(pilot, "num_rewrites", 3),
                num_vis_samples=_pilot_int(pilot, "num_vis_samples", 20),
                seed=_pilot_int(pilot, "seed", 42),
                semantic_label_map=pilot.get("semantic_label_map"),
            )
            audit_result = export_manual_audit(
                pipeline_result["pairs_v4_rewritten"],
                str(pilot_dir / "manual_audit.csv"),
                num_samples=_pilot_int(pilot, "manual_audit_samples", 100),
                seed=_pilot_int(pilot, "seed", 42),
            )
            pilot_summaries.append(
                {
                    "name": name,
                    "status": "ok",
                    "output_dir": str(pilot_dir),
                    "manifest": manifest_result,
                    "manual_audit": audit_result,
                    "num_pairs_v3": pipeline_result["num_pairs_v3"],
                    "num_pairs_v4": pipeline_result["num_pairs_v4"],
                    "num_issues": pipeline_result["num_issues"],
                    "terrain_counts": pipeline_result["stats"].get("terrain_counts", {}),
                    "motion_counts": pipeline_result["stats"].get("motion_counts", {}),
                    "version_counts": pipeline_result["stats"].get("version_counts", {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch experiments moving.
            failure = {
                "name": name,
                "status": "failed",
                "output_dir": str(pilot_dir),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failed_pilots.append(failure)
            pilot_summaries.append(failure)

    summary = {
        "experiment_name": experiment_name,
        "config_path": config_path,
        "output_root": str(output_root),
        "total_pilots": len(config.get("pilots", [])),
        "succeeded": sum(1 for pilot in pilot_summaries if pilot.get("status") == "ok"),
        "failed": len(failed_pilots),
        "total_pairs_v4": sum(int(pilot.get("num_pairs_v4", 0)) for pilot in pilot_summaries),
        "total_issues": sum(int(pilot.get("num_issues", 0)) for pilot in pilot_summaries),
        "pilots": pilot_summaries,
    }
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary_markdown(summary, output_root / "experiment_summary.md")
    (output_root / "failed_pilots.json").write_text(
        json.dumps(failed_pilots, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a batch real-pilot experiment.")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    return parser.parse_args()


def main() -> None:
    """Run the experiment and print its summary path."""
    args = parse_args()
    summary = run_experiment(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
