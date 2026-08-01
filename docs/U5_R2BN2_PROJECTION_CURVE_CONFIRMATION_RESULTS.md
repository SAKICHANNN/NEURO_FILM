# U5.R2BN2 projection-curve confirmation results

BN2 passes all frozen gates on 64 official FiveK contents excluded from BN1.
Pair IDs and all manifest SHA-256 values have zero overlap with BN1's AY0,
AY3 and AY6 populations. Camera metadata is unavailable, so this is content
confirmation only, not camera OOD.

| measure | result |
|---|---:|
| adaptive mean gain over global | 5.35% |
| adaptive wins over global | 60.94% |
| adaptive P95 / worst error ratio | .8396 / .8718 |
| adaptive AO6 style ratio | 1.0226 |
| adaptive median / minimum safe dose | .8188 / .7130 |
| Oracle mean gain over identity | 21.61% |
| new boundary / out-of-cube | 0 / 0 |
| nonpositive sampled Jacobians | 0 |

The two independent reports are byte-identical at SHA-256
`04cb352f...26fde` (stable evidence `5f7d4479...540ab`). The unchanged
strict-interior operator still improves its same-family global baseline without
clipping or per-pixel post scaling.

This remains a FiveK digital-retouch control, not film or stock learning. BN2
opens one independent severe-artifact-first visual comparison of the adaptive
and global neutral bases under the unchanged AO6 look. It does not authorize
product integration or claim human preference.
