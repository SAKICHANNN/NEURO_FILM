# U5.R2AI0 Dual-Champion Global Composition Results

## Decision

`density_then_anchor__density_s50` is retained as a deterministic B0
film-inspired Look Approximation challenger. It passes the frozen automatic,
blind autonomous-vision and full-resolution severe-artifact gates. The
production default is unchanged.

This result is not a stock response, calibrated reference, identified
digital-to-film operator, owner preference result or population preference
result. It does not open current-pixel fitting, training, latent-mode
discovery or product integration.

## Frozen experiment

The bank contains only:

- the two sequential orders of the already frozen
  `anchor56_chroma_margin4_challenger` and
  `cyan_shadow_warm_highlight_like__s50` operators;
- density strengths `0.20`, `0.35` and `0.50`;
- one final four-code RGB8 margin;
- no fitting, routing, spatial logic, intermediate quantization or
  per-image selection.

All 41 frozen gold/stress images were rendered in each of two independent
processes. Both runs are exact:

| Artifact | SHA-256 |
|---|---|
| configuration | `136593e7765e0fee06cce8263c52b5326899577f149eebcde60c6b5f1136247f` |
| run A/B manifest | `8093823497a8ba69a5e2c15db003eed4d6e2b05e1e49cc10ed559478250d9242` |
| run A/B automatic report | `222b29eef53c5ea54b4a526cbcf36eade4a1c10ebbd754468107acc5f338d9dd` |
| pre-unblind review | `321a43874f4693c951742edd6087510ae33b89cb84eea1afc479e4edf79ccbc0` |
| post-reveal/full-resolution review | `4e377fe88f9eff13aa33f974aef5c640ad8876d49f17ad8f4318e69ae952a193` |

The rendering software commit is
`1f61119087cdb72d939b8db0c7b915e4adb7c5ce`.

## Automatic result

The frozen floors were Delta E76 `13.9944` style and `7.8946` non-basic
residual, each exactly `0.5` above the R2E1 parent result. Worst new hard
clipping on gold was capped at `0.5%`.

| Candidate | Gold style | Gold non-basic | Worst gold clip | Result |
|---|---:|---:|---:|---|
| anchor then density, s0.20 | 9.9582 | 6.4199 | 0% | fail |
| anchor then density, s0.35 | 12.0128 | 8.9470 | 0% | fail style |
| anchor then density, s0.50 | 14.3694 | 9.5564 | 0% | pass |
| density then anchor, s0.20 | 11.0019 | 6.1947 | 0% | fail |
| density then anchor, s0.35 | 13.1129 | 8.8908 | 0% | fail style |
| density then anchor, s0.50 | **15.0576** | **11.3299** | **0%** | **pass** |

Every candidate also has zero measured new stress-set hard clipping. Both
s0.50 orders therefore entered the preregistered visual stage.

## Blind autonomous visual evidence

The reviewer recorded all rankings before reading the private mapping. After
reveal:

- `density_then_anchor__density_s50` ranked `1, 2, 1`, ahead of both parents
  in all three rounds and first in two;
- `anchor_then_density__density_s50` ranked `3, 1, 2`, ahead of both parents
  in two rounds and first in one.

Both pass the minimum two-round rule. The first candidate is retained because
it has the stronger fixed comparison: three parent wins and two first-place
rounds. This is autonomous development evidence only, not a new owner vote or
independent-human validation.

## Full-resolution severe veto

All nine frozen gold images were inspected at full resolution for the retained
candidate: `01`, `05`, `08`, `09`, `11`, `18`, `21`, `29` and
`FS_FACE_01`.

- sample `11` does not reproduce the frozen red-speckle/posterization failure;
- face, hair, hand, text, bicycle and bridge geometry remain coherent;
- sample `09` retains a smooth wall/highlight gradient without banding;
- no seam, colour block, repeated texture or confirmed severe clipping failure
  is visible.

There are non-severe aesthetic risks: sample `01` amplifies fine source
chromatic edge fringing, sample `21` has strongly warm skin, and sample `29`
has aggressive highlights/dense shadows. They remain visible and recorded;
they are not relabelled as severe corruption to manipulate the result.

## Interpretation and branch

This experiment rejects the idea that the two previous deterministic
champions were already an exhaustive global style frontier. Operator order
matters, and composing the density-domain look before the bounded anchor
creates materially stronger non-basic colour than either parent while
retaining the current severe-artifact envelope.

The result is still development-set evidence from known project-owned
operators. It can overfit the visual population or autonomous reviewer despite
having no trained parameters. The next legal leaf is therefore an
independent-population confirmation/OOD contract for this one frozen
composition. It must not retune order, strength, margin or thresholds and must
not reuse the current 41-image population to decide promotion.

Authority:

- contract:
  `docs/planning/U5_R2AI0_DUAL_CHAMPION_GLOBAL_COMPOSITION_CONTRACT.md`;
- decision:
  `configs/u5_r2ai0_dual_champion_global_composition_decision_v1.json`;
- ignored full evidence:
  `outputs/u5_r2ai0_dual_champion_global_composition_v1/`.
