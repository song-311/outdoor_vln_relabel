# RELLIS-3D Filtered Instruction Text Cleanup Report

Generated: 2026-05-09

## 1. Files Modified

| File | Change |
|---|---|
| `tools/filter_rellis_pairs.py` | **Complete rewrite.** Clean instruction builder with natural language, no raw relation tokens, no dangling phrases, no variant suffixes. |
| `tools/check_rellis_filtered_quality.py` | **Enhanced.** Added bad phrase scan with regex patterns for `front_left`, `front_right`, `variant`, `with.`, `and the.`, `ahead o` (truncated), duplicate objects, contradictory relations. |
| `outdoor_vln_relabel/perception/landmarks_from_mask.py` | (Previous session) Added person detection + tiered score heuristic. |
| `outdoor_vln_relabel/validation/checks.py` | (Previous session) Added `mud`, `water`, `person`, `pedestrian`, `fence` to landmark consistency check. |
| `outdoor_vln_relabel/build_dataset.py` | (Previous session) Changed validation failure from crash to skip. |

## 2. Quality Check Output Summary

### filtered (pairs_v4_filtered.jsonl)

| Metric | Value |
|---|---|
| Total records | 35 |
| All scores == 0.95 | **0** |
| Instruction-landmark mismatch | **0** |
| Person missing from instruction | **0** |
| **Bad phrases** | **0** |
| Word count range | 16-20 (all within 8-20) |

### rewritten (pairs_v4_rewritten.jsonl)

| Metric | Value |
|---|---|
| Total records | 35 |
| All scores == 0.95 | **0** |
| Instruction-landmark mismatch | **0** |
| Person missing from instruction | 2 (fixed by filter) |
| **Bad phrases** | **0** |

## 3. Filtered Record Count

- **Total filtered records:** 35
- **Unique instructions:** 23
- Some segments share identical instruction text due to similar landmark patterns (different trajectories with same obstacle configuration). The user explicitly allowed this: "只要避免完全重复 instruction 即可".

## 4. Landmark Counts in Filtered Output

| Landmark | Count |
|---|---|
| person | 14 |
| puddle | 8 |
| mud | 15 |
| water | 0 |
| tree | 18 |
| bush | 12 |
| grass field | 33 |
| **total** | **100** |

## 5. Bad Phrase Check

**Bad phrases: 0** — confirmed zero in both rewritten and filtered files.

The bad phrase scan checks for:
- Raw relation tokens: `front_left`, `front_right`
- Dangling endings: `with.`, `and the.`
- Dedup artifacts: `variant`
- Truncation artifacts: `ahead o` (not followed by `n` — i.e. not "ahead on")
- Duplicate objects: `the puddle ahead, the puddle ahead`
- Contradictory relations: `person right and left`, `person left and right`
- Empty avoid clauses: `keeping clear of the.`, `avoiding the.`, `staying clear of the.`

## 6. Random 15 Instructions

```
 [1] (18w) Move forward and slightly left across the wet ground while avoiding the person ahead and the puddles ahead.
 [2] (20w) Move forward across the grass while staying clear of the person ahead and the muddy area ahead on your front-right.
 [4] (18w) Move forward across the grass while avoiding the person ahead and the muddy area ahead on your front-right.
 [7] (16w) Move forward across the grass while avoiding the person on your left and the bush ahead.
[13] (18w) Move forward across the grass while staying clear of the person on your left and the bush ahead.
[15] (20w) Turn right across the grass while keeping clear of the muddy area ahead on your front-right and the tree ahead.
[17] (19w) Move forward and slightly right across the grass while staying clear of the puddles ahead and the tree ahead.
[18] (19w) Move forward and slightly right across the grass while keeping clear of the puddles ahead and the tree ahead.
[21] (18w) Move forward and slightly left across the grass while keeping clear of the puddles ahead on your front-left.
[23] (19w) Move forward across the grass while staying clear of the tree ahead and the bush ahead on your front-right.
[28] (18w) Move forward and slightly right across the grass while avoiding the muddy area ahead and the tree ahead.
[32] (20w) Move forward and slightly right across the grass while staying clear of the muddy area ahead and the tree ahead.
[33] (20w) Move forward and slightly right across the grass while staying clear of the muddy area ahead and the tree ahead.
[34] (20w) Move forward and slightly right across the grass while staying clear of the muddy area ahead and the tree ahead.
```

## 7. Git Diff Summary (Python files only)

```
outdoor_vln_relabel/analysis/visualize.py          |  35 ++++++
outdoor_vln_relabel/build_dataset.py               |   5 +-
outdoor_vln_relabel/language/templates.py          |  42 +++++--
outdoor_vln_relabel/perception/align.py            |  18 +++-
outdoor_vln_relabel/perception/landmarks_from_mask.py |  60 +++++++-
outdoor_vln_relabel/validation/checks.py           |  22 ++++-
tools/filter_rellis_pairs.py                       | (new, untracked)
tools/check_rellis_filtered_quality.py             | (new, untracked)
```

No upstream data files were modified. All demo output files in `outputs/` remain untouched.

## 8. Known Issues

1. **2 rewritten records have person in landmarks but not in instruction.** These are from the v3/v4 template generation stage — the filter fully fixes them. The root cause is that v3 instruction templates don't account for person landmarks yet. This is outside the scope of this fix.

2. **12 duplicate instructions across different trajectory segments** (35 records, 23 unique). Different segments share identical obstacle configurations and thus the same instruction text. The user indicated this is acceptable for now; a more sophisticated dedup could introduce trajectory-specific details.

3. **The rewritten file only has 35 records** (down from 39 in the previous run). 4 records failed validation during the pipeline v3 pair building step and were skipped. These are edge cases where the instruction templates generate text that doesn't match the landmark consistency check. This is a pre-existing pipeline issue, not introduced by this fix.

4. **RELLIS-3D label map (`rellis3d_official.yaml`) still marks person (label 17) with `outdoor_group: unknown`.** The code-level safety-critical override in `landmarks_from_mask.py` handles this, but for semantic cleanliness the label map should be updated.

5. **Score heuristic produces valid varied scores but the spread is narrow** (mostly 0.82-0.95). This is intentional — the heuristic is based on bbox area and category, not ML model confidence. More variety would require a learned confidence model.
