# Audit Package Validation Report

## Summary
- hard_errors: 0
- warnings: 0

## 1. File existence

- JSONL: OK  `outputs/rellis_00000_run_official_map/pairs_v4_filtered.jsonl`
- CSV: OK  `outputs/rellis_00000_run_official_map/manual_audit_filtered.csv`
- sample_vis dir: OK  `outputs/rellis_00000_run_official_map_ego_line_only/sample_vis`
- samples.md: OK  `outputs/rellis_00000_run_official_map_ego_line_only/sample_vis/samples.md`

## 2. Count consistency

- JSONL records: 35
- CSV rows: 35
- PNG files: 35
- Counts consistent: OK

## 3. CSV index resolution

- All 35 CSV indices resolve to JSONL: OK

## 4. CSV index → PNG matching

- All 35 CSV indices have matching PNG: OK

## 5. Bad phrase scan (JSONL instructions)

- Bad phrases in instructions: **0**

## 6. Raw relation tokens in samples.md

- Raw relation tokens in samples.md: **0**

## 7. All landmark scores == 0.95

- Samples with all scores == 0.95: 0

## 8. Person/pedestrian missing from instruction

- Person in landmarks but missing from instruction: 0