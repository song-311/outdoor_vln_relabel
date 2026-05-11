# Outdoor-VLN Dataset QA Report

## Basic Statistics
- total pairs: 35
- scenes: 1
- segments: 8

### Versions
| version | count |
|---|---:|
| evidence_landmark_terrain_motion_v3 | 23 |
| rewritten_evidence_landmark_terrain_motion_v4 | 12 |

### Rewrite Sources
| rewrite_source | count |
|---|---:|
| rule_based | 12 |

### Rewrite Counts
| kind | count |
|---|---:|
| original | 23 |
| rewritten | 12 |

## Terrain Distribution
| terrain | count |
|---|---:|
| grass | 33 |
| mud_water | 2 |

## Motion Distribution
| motion | count |
|---|---:|
| forward | 15 |
| forward_left | 5 |
| forward_right | 12 |
| turn_right | 3 |

## Landmark Role Distribution
| role | count |
|---|---:|
| avoid | 139 |
| follow | 36 |

## Landmark Relation Distribution
| relation | count |
|---|---:|
| ahead | 82 |
| front_left | 21 |
| front_right | 20 |
| left | 26 |
| right | 26 |

## Confidence Summary
- min: 0.862364
- max: 0.935142
- mean: 0.893253
- median: 0.891966

## Instruction Length Summary
- min: 9
- max: 29
- mean: 18.514286
- median: 16

## Landmark Summary
- total landmarks: 175
- avg landmarks per pair: 5.0

### Landmark Names
| landmark | count |
|---|---:|
| bush | 33 |
| grass field | 36 |
| mud | 27 |
| person | 21 |
| puddle | 13 |
| tree | 45 |

## Potential Issues
- total issues: 0

| issue_type | count |
|---|---:|
| _none_ | 0 |

