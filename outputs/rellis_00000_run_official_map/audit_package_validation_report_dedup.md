# Audit Package Validation Report

## 1. Input Paths
- JSONL: `outputs/rellis_00000_run_official_map/pairs_v4_filtered_audit.jsonl`
- CSV: `outputs/rellis_00000_run_official_map/manual_audit_filtered_audit.csv`
- sample_vis dir: `outputs/rellis_00000_run_official_map_ego_line_only/sample_vis_audit_dedup`
- samples.md: `outputs/rellis_00000_run_official_map_ego_line_only/sample_vis_audit_dedup/samples.md`

## 2. Summary
| metric | value |
|--------|-------|
| num_jsonl_records | 8 |
| num_csv_rows | 8 |
| num_png_files | 8 |
| unique_segment_id | 8 |
| repeated_segment_id_count | 0 |
| unique_instruction_count | 8 |
| duplicate_instruction_count | 0 |
| bad_phrase_count | 0 |
| missing_png_count | 0 |
| csv_jsonl_mismatch_count | 0 |
| samples_md_mismatch_count | 0 |
| total_errors | 0 |
| total_warnings | 0 |
| **final status** | **PASS** |
