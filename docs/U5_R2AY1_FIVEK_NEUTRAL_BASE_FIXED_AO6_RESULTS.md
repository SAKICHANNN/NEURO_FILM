# U5.R2AY1 FiveK neutral base + fixed AO6 result

AY1 keeps every AO6 component fixed, including the retained
`density_then_anchor__density_s50` B0, its historical RGB8 boundary and the
AO6 `t15/c35` factorized residual. It compares identity, outer-train global,
camera-group-held-out ridge and per-pair oracle neutral bases against the same
fixed AO6 look applied to the existing neutral digital-retouch target.

Two 64-pair reports are byte-identical at
`bd4d327637e8bbac35259d0755216f8298fb5de1dcace7428f03caaff2aa9896`.

| frozen measurement | result |
|---|---:|
| ridge mean improvement over global | 12.62% |
| ridge wins over global | 75.00% |
| ridge/global P95 RMSE ratio | 0.8225 |
| bootstrap lower bound for absolute mean improvement | 0.00840 |
| target median style Delta E76 | 13.9682 |
| ridge/target median style-dose ratio | 1.0158 |
| maximum new boundary fraction | 0 |

All automatic gates pass. The result directly rejects the failure mode where
canonicalization merely averages away the strong look: the ridge path retains
slightly more median style dose than the same look on the paired neutral
target.

The fixed ten-row stress sheet deliberately includes the highest ridge errors,
the largest ridge regressions versus global and the largest gains. Autonomous
inspection finds zero confirmed severe corruption: the baby face, people,
architecture, stage gradients, textiles, foliage and fine edges remain
structurally intact. The look is conspicuous rather than saturation-only:
shadow hue, highlight warmth, density and palette relationships all move.
Some individual rows still prefer the global control, so this is not a
per-image or universal win.

Retain `bounded neutral base -> fixed AO6` as development architecture. Open
only an untouched-content confirmation on rows 65-128 of the already retained
FiveK high-precision pack. Those rows lack verified camera-group metadata, so
that child may establish content-population transfer only, never camera OOD,
film identity, stock response, calibration or product promotion.
