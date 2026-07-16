# RF1.4B1 Gold100 display-transform consistency results

## Decision

RF1.4B1 passes its frozen metric and visual gates, but it does **not** establish
a Kodak Gold emulsion response or a digital-to-film mapping. It establishes a
repeatable global mapping between BlueNeg's post-negation archive preview and
its display proxy across six held-out physical rolls.

The selected operator class is the simpler bounded ridge 3x3 affine. SepLUT17
passes the preregistered comparison against per-channel affine, yet does not
justify its extra nonlinear capacity because the 3x3 affine is slightly better
overall and on four of six roll point estimates.

## Frozen evidence

- evaluator commit: `81938b4d2b4c2a8c4d32fb7d228444b3f571332c`;
- two byte-identical reports: SHA-256 `f1771c6b6c509b2489b00fdb1478a2b3a4cf3e05915b068296851cf0ba1a488a`;
- 47 exact official pairs, six whole-roll LOO folds;
- 10,000 roll-cluster bootstrap resamples, seed 1411;
- visual renderer commit `6723f8d`; 47 full-resolution five-panel strips and
  six roll sheets; manifest SHA-256
  `7f270a153b59a2112b8d073fb72584bb7bc179e90147938cd5021fcbdad0e969`.

| Operator | roll-balanced mean Delta E76 | Relative to channel affine |
|---|---:|---:|
| bounded per-channel affine | 8.0522 | baseline |
| bounded ridge 3x3 affine | **5.3072** | **34.09% better** |
| monotone SepLUT17 + bounded 3x3 | 5.3800 | 33.19% better |

SepLUT improves all 6/6 rolls over channel affine and beats the median
single-wrong-roll SepLUT on 6/6. Its absolute improvement bootstrap interval is
`[+2.2431,+3.1441]` Delta E76. Median style displacement is 9.6652 Delta E76;
sampled raw clipping is zero and clipping change versus channel affine is
-0.02465.

## Visual gate

Autonomous Codex vision reviewed all six roll contact sheets plus the
full-resolution night-red, people/street and extreme-highlight risk strips.
No new severe banding, posterization, colour blocks, red speckles, seams,
geometry damage or texture rewrite was confirmed. Existing flare, clipping,
exposure and scan defects in the archive evidence were treated as source/target
properties, not algorithm output defects. Matrix and SepLUT outputs are
visually near-indistinguishable at review scale.

This is autonomous visual evidence, not owner preference or population
validation.

## Interpretation and next branch

The useful signal is cross-roll archive display-chain consistency, not
stock-specific authenticity. The result opens only `RF2.S0`: fit the selected
3x3 class once on all six rolls, then falsify its transplant as a bounded
`Look Approximation` on the existing digital gold/stress set. That test must
include identity, matched contrast/saturation/simple-affine controls, severe
artifact/style salience and OOD fallback. A transplant pass may create a
film-inspired product challenger; it still cannot reach `S2` or calibrated
claims without independent stock/process evidence.

Machine decision: `configs/real_film_gold_transform_consistency_decision_v1.json`.
