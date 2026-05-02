# Outdoor-VLN-Relabel

Stage 2 builds terrain-aware instruction-path pairs from trajectory arrays while
keeping the Stage 1 motion-only API available.
Stage 3 optionally adds rule-based dummy landmarks with `--use_landmarks`.
Stage 4A adds manifest, metadata, and semantic-mask filename evidence grounding.
Stage 5B adds indexed/RGB semantic mask parsing. Stage 5C adds configurable
real-dataset pilot ingestion for RELLIS-3D-style image/mask/pose data.

## Demo Commands

Create or reuse a demo trajectory npz with `positions`, `yaws`, and
`timestamps`, then run:

```bash
python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_terrain_pairs.jsonl \
  --scene_id demo_scene \
  --default_terrain dirt_trail
```

Other terrain defaults:

```bash
python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_terrain_grass_pairs.jsonl \
  --scene_id demo_scene \
  --default_terrain grass

python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_terrain_mud_water_pairs.jsonl \
  --scene_id demo_scene \
  --default_terrain mud_water

python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_terrain_rough_pairs.jsonl \
  --scene_id demo_scene \
  --default_terrain rough_terrain
```

The npz adapter also supports an optional `terrain_labels` array. It may be a
single label, one label per frame, or one label per generated segment.

## Landmark-Aware Demo

```bash
python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_landmark_dirt.jsonl \
  --scene_id demo_scene \
  --default_terrain dirt_trail \
  --use_landmarks

python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_landmark_mud.jsonl \
  --scene_id demo_scene \
  --default_terrain mud_water \
  --use_landmarks

python -m outdoor_vln_relabel.build_motion_pairs_demo \
  --input_npz outputs/demo_traj.npz \
  --output_jsonl outputs/demo_landmark_vegetation.jsonl \
  --scene_id demo_scene \
  --default_terrain vegetation \
  --use_landmarks
```

## Manifest Evidence Demo

```bash
python -m outdoor_vln_relabel.make_demo_manifest \
  --output_dir outputs/demo_manifest

python -m outdoor_vln_relabel.build_dataset \
  --adapter manifest \
  --dataset_root outputs/demo_manifest \
  --output_jsonl outputs/demo_manifest_pairs.jsonl \
  --scene_id demo_manifest_scene \
  --terrain_mode metadata_or_mask \
  --use_landmarks \
  --landmark_mode metadata_or_mask
```

The manifest adapter expects `trajectory.csv` with `frame_id`, `timestamp`,
`rgb_path`, `x`, `y`, and `yaw`. Optional `semantic_path`, `terrain`, and
`landmarks` columns provide evidence for Stage 4A grounding.

## Dataset QA And Visualization

```bash
python -m outdoor_vln_relabel.analyze_dataset \
  --input_jsonl outputs/demo_manifest_pairs.jsonl \
  --output_report outputs/demo_manifest_report.md \
  --output_issues outputs/demo_manifest_issues.jsonl

python -m outdoor_vln_relabel.visualize_samples \
  --input_jsonl outputs/demo_manifest_pairs.jsonl \
  --output_dir outputs/demo_sample_vis \
  --num_samples 5 \
  --seed 42
```

## Constrained Rewrite Demo

```bash
python -m outdoor_vln_relabel.rewrite_instructions \
  --input_jsonl outputs/demo_manifest_pairs.jsonl \
  --output_jsonl outputs/demo_manifest_rewritten.jsonl \
  --backend rule_based \
  --num_variants 3 \
  --keep_original

python -m outdoor_vln_relabel.rewrite_instructions \
  --input_jsonl outputs/demo_manifest_pairs.jsonl \
  --output_jsonl outputs/unused.jsonl \
  --backend llm_stub \
  --prompt_output outputs/demo_rewrite_prompts.jsonl \
  --num_variants 3 \
  --max_records 3
```

## Real Dataset Pilot

Convert a simple `frames/` + `poses.csv` dataset:

```bash
python -m outdoor_vln_relabel.convert_to_manifest \
  --mode folder_pose \
  --input_root outputs/demo_folder_pose \
  --output_dir outputs/demo_folder_manifest \
  --scene_id folder_scene \
  --sequence_id folder_seq \
  --default_terrain grass
```

Run the full pilot pipeline on any unified manifest:

```bash
python -m outdoor_vln_relabel.run_pilot_pipeline \
  --manifest_root outputs/demo_manifest \
  --scene_id demo_pilot_scene \
  --output_dir outputs/demo_pilot_run \
  --default_terrain dirt_trail \
  --terrain_mode metadata_or_mask \
  --landmark_mode metadata_or_mask \
  --rewrite_backend rule_based \
  --num_rewrites 3 \
  --num_vis_samples 5
```

## Real Dataset Pilot Usage

For RELLIS-3D-style data, use a config file rather than hard-coded paths.

1. Copy `outdoor_vln_relabel/configs/real_dataset_pilots/rellis3d_pilot_template.yaml`
   to a sequence-specific config.
2. Edit `dataset.root`, `paths.image_dir`, `paths.mask_dir`, `paths.pose_file`,
   `pose_file.columns`, and `matching.*_pattern` to match your local dataset.
3. Inspect the raw dataset before conversion:

```bash
python -m outdoor_vln_relabel.prepare_real_pilot \
  --config path/to/rellis_seq_config.yaml \
  --inspect_only
```

4. Fix any reported pose-column or path errors. Missing pose data is a hard
   error; the converter will not fabricate official VLN trajectories.
5. Convert to a unified manifest:

```bash
python -m outdoor_vln_relabel.prepare_real_pilot \
  --config path/to/rellis_seq_config.yaml \
  --output_dir outputs/rellis_seq_manifest
```

6. Run the full pilot pipeline:

```bash
python -m outdoor_vln_relabel.run_pilot_pipeline \
  --manifest_root outputs/rellis_seq_manifest \
  --scene_id rellis_scene_000 \
  --output_dir outputs/rellis_pilot_run \
  --terrain_mode metadata_or_mask \
  --landmark_mode metadata_or_mask \
  --rewrite_backend rule_based \
  --num_rewrites 3 \
  --num_vis_samples 10
```

If `manifest_info.json` contains `semantic_label_map`, `run_pilot_pipeline`
uses it automatically unless `--semantic_label_map` is passed explicitly.

7. Review `report.md` and `sample_vis/samples.md`.
8. Export a manual audit sheet:

```bash
python -m outdoor_vln_relabel.export_manual_audit \
  --input_jsonl outputs/rellis_pilot_run/pairs_v4_rewritten.jsonl \
  --output_csv outputs/rellis_pilot_run/manual_audit.csv \
  --num_samples 100 \
  --seed 42
```

## Experiment And Audit Summaries

Batch one or more real-pilot configs:

```bash
python -m outdoor_vln_relabel.run_experiment \
  --config outdoor_vln_relabel/configs/experiment_templates/demo_real_pilot_experiment.yaml
```

Summarize a filled manual audit CSV:

```bash
python -m outdoor_vln_relabel.audit_summary \
  --audit_csv outputs/demo_real_pilot_run/manual_audit.csv \
  --output_json outputs/demo_real_pilot_run/audit_summary.json \
  --output_md outputs/demo_real_pilot_run/audit_summary.md
```

Export paper-ready distribution tables and failure cases:

```bash
python -m outdoor_vln_relabel.export_paper_tables \
  --input_jsonl outputs/demo_real_pilot_run/pairs_v4_rewritten.jsonl \
  --output_dir outputs/demo_real_pilot_run/paper_tables

python -m outdoor_vln_relabel.export_failure_cases \
  --input_jsonl outputs/demo_real_pilot_run/pairs_v4_rewritten.jsonl \
  --issues_jsonl outputs/demo_real_pilot_run/issues.jsonl \
  --output_dir outputs/demo_real_pilot_run/failure_cases \
  --max_cases 50
```
