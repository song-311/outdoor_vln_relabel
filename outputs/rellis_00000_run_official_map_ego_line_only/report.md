# Outdoor-VLN Dataset QA Report

## Basic Statistics
- total pairs: 45
- scenes: 1
- segments: 9

### Versions
| version | count |
|---|---:|
| evidence_landmark_terrain_motion_v3 | 27 |
| rewritten_evidence_landmark_terrain_motion_v4 | 18 |

### Rewrite Sources
| rewrite_source | count |
|---|---:|
| rule_based | 18 |

### Rewrite Counts
| kind | count |
|---|---:|
| original | 27 |
| rewritten | 18 |

## Terrain Distribution
| terrain | count |
|---|---:|
| grass | 42 |
| mud_water | 3 |

## Motion Distribution
| motion | count |
|---|---:|
| forward | 15 |
| forward_left | 12 |
| forward_right | 12 |
| turn_right | 6 |

## Landmark Role Distribution
| role | count |
|---|---:|
| avoid | 180 |
| follow | 45 |

## Landmark Relation Distribution
| relation | count |
|---|---:|
| ahead | 114 |
| front_left | 27 |
| front_right | 12 |
| left | 12 |
| right | 60 |

## Confidence Summary
- min: 0.92615
- max: 0.95
- mean: 0.94523
- median: 0.95

## Instruction Length Summary
- min: 9
- max: 29
- mean: 19.755556
- median: 17

## Landmark Summary
- total landmarks: 225
- avg landmarks per pair: 5.0

### Landmark Names
| landmark | count |
|---|---:|
| bush | 78 |
| grass field | 45 |
| mud | 12 |
| puddle | 30 |
| tree | 60 |

## Potential Issues
- total issues: 0

| issue_type | count |
|---|---:|
| _none_ | 0 |

