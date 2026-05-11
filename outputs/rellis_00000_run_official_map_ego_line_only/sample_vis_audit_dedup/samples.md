# Outdoor-VLN Manual Audit Sample Visualizations

Source JSONL: `outputs/rellis_00000_run_official_map/pairs_v4_filtered_audit.jsonl`

Source CSV: `outputs/rellis_00000_run_official_map/manual_audit_filtered_audit.csv`

Total samples: 8


## Audit Row 001 / JSONL Index 000003

- image: audit_001_idx_000003.png
- segment_id: seq_00000_000005
- instruction_id: 13
- instruction: Turn right across the grass while staying clear of the muddy area slightly to your right and the tree ahead.
- terrain: grass
- motion: turn_right
- confidence: 0.862364
- landmarks: mud (avoid, front-right, score=0.84621), tree (avoid, ahead, score=0.869222), grass field (follow, ahead, score=0.9)

## Audit Row 002 / JSONL Index 000004

- image: audit_002_idx_000004.png
- segment_id: seq_00000_000006
- instruction_id: 16
- instruction: Move forward and slightly right across the grass while staying clear of the puddles ahead and the tree ahead.
- terrain: grass
- motion: forward_right
- confidence: 0.906085
- landmarks: puddle (avoid, ahead, score=0.897624), tree (avoid, ahead, score=0.93), grass field (follow, ahead, score=0.9)

## Audit Row 003 / JSONL Index 000006

- image: audit_003_idx_000006.png
- segment_id: seq_00000_000008
- instruction_id: 22
- instruction: Move forward across the grass while staying clear of the tree ahead and the bush slightly to your right.
- terrain: grass
- motion: forward
- confidence: 0.904653
- landmarks: tree (avoid, ahead, score=0.93), bush (avoid, front-right, score=0.878464), grass field (follow, ahead, score=0.9)

## Audit Row 004 / JSONL Index 000007

- image: audit_004_idx_000007.png
- segment_id: seq_00000_000009
- instruction_id: 25
- instruction: Move forward and slightly right across the grass while staying clear of the muddy area ahead and the tree ahead.
- terrain: grass
- motion: forward_right
- confidence: 0.891966
- landmarks: mud (avoid, ahead, score=0.85429), tree (avoid, ahead, score=0.93), grass field (follow, ahead, score=0.9)

## Audit Row 005 / JSONL Index 000002

- image: audit_005_idx_000002.png
- segment_id: seq_00000_000004
- instruction_id: 10
- instruction: Move forward across the grass while staying clear of the person on your left and the bush ahead.
- terrain: grass
- motion: forward
- confidence: 0.884002
- landmarks: person (avoid, left, score=0.923827), bush (avoid, ahead, score=0.866727), grass field (follow, ahead, score=0.9)

## Audit Row 006 / JSONL Index 000005

- image: audit_006_idx_000005.png
- segment_id: seq_00000_000007
- instruction_id: 19
- instruction: Move forward and slightly left across the grass while staying clear of the puddles slightly to your left.
- terrain: grass
- motion: forward_left
- confidence: 0.902492
- landmarks: puddle (avoid, front-left, score=0.876393), grass field (follow, ahead, score=0.9)

## Audit Row 007 / JSONL Index 000000

- image: audit_007_idx_000000.png
- segment_id: seq_00000_000001
- instruction_id: 1
- instruction: Move forward and slightly left across the wet ground while keeping clear of the person ahead and the puddles ahead.
- terrain: mud_water
- motion: forward_left
- confidence: 0.935142
- landmarks: person (avoid, ahead, score=0.924595), puddle (avoid, ahead, score=0.96)

## Audit Row 008 / JSONL Index 000001

- image: audit_008_idx_000001.png
- segment_id: seq_00000_000003
- instruction_id: 7
- instruction: Move forward across the grass while staying clear of the person ahead and the muddy area slightly to your right.
- terrain: grass
- motion: forward
- confidence: 0.894354
- landmarks: person (avoid, ahead, score=0.922454), mud (avoid, front-right, score=0.85926), grass field (follow, ahead, score=0.9)
