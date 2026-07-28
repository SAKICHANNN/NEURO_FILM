# U5.R2AI1 Dual-Champion Independent Confirmation Results

## Decision

The frozen `density_then_anchor__density_s50` composition does not confirm on
the independent 17-image, nine-camera-make population. Both formal processes
are byte-identical, but the candidate fails three preregistered automatic
gates. U5.R2AI1 therefore closes without threshold, row, resize, order,
strength or margin rescue.

The U5.R2AI0 development result remains valid as development-only B0 Look
Approximation evidence. It is not independently promoted, does not change the
production default and does not open any stock, calibration, training,
operator-fitting or latent-mode claim.

## Exact execution

The exact 17 eligible R2AI1S rows were rendered through only the two frozen
parents and the one retained composition. Each run contains 51 rendered PNGs,
one manifest and one automatic report.

| Artifact | SHA-256 |
|---|---|
| configuration | `28f7fcf0ba691a04cad1bfe530f626e9a868901fd7d5f481c7a4b9c67a2b64de` |
| run A/B manifest | `53933c4ab917a28db83184c30b7fe1ae42d4682f5b71ab1ba7c6e15f11ee3fd3` |
| run A/B automatic report | `5910361382178050c0d42f728d274ba20c3f233ba2f7ed410bb839df7feb2e60` |

The rendering software commit is
`4fac70db92f4f7eed4c2569d9951ab4aa6d736b3`. All 53 relative paths and
file hashes match between the two runs.

## Automatic result

The candidate retains substantial style and non-basic colour, and it creates
no measured new hard clipping. That is not enough to pass: the contract also
requires reliable advantage over the better parent and a bounded per-image
style envelope.

| Candidate | Median style | Median non-basic | P95 style | Maximum style | Worst new clip |
|---|---:|---:|---:|---:|---:|
| anchor56 parent | 7.9209 | 8.0781 | 20.0170 | 27.9554 | 0% |
| density s0.50 parent | **13.8732** | 9.3557 | 19.4666 | 20.0407 | 0% |
| density then anchor | 13.1756 | **12.6753** | 27.8119 | 32.2442 | 0% |

Eight of eleven automatic gates pass:

- style and non-basic retention pass;
- the P95 style envelope passes;
- median non-basic gain is `+0.5450` Delta E76 and passes;
- non-basic wins pass at `11/17` images and `7/9` makes;
- style wins pass at the grouped-make level with exactly `6/9`;
- every output has zero measured new hard clipping.

Three gates fail:

| Failed gate | Observed | Frozen requirement |
|---|---:|---:|
| median style gain over better parent | `-0.5261` | at least `+0.25` |
| per-image style wins | `7/17` | at least `10/17` |
| maximum per-image style | `32.2442` | at most `31.4746` |

The maximum comes from `sony_dslra560`. Its zero new clipping means this is
not evidence of a simple output-clamp failure; it is `1.28056x` the
development maximum against the frozen `1.25x` limit. Canon, Fujifilm,
Panasonic and Samsung win both grouped metrics; Nikon and Sony win only
style, while Olympus, Pentax and Ricoh win only non-basic residual. More
broadly, the composition adds residual colour structure on many images, but
it does not reliably add style beyond whichever single parent is already
stronger for that image.

## Visual stage

The frozen contract requires a complete automatic pass before blind sheets or
full-resolution candidate adjudication. Because the automatic gate fails:

- no blind sheet or private mapping was generated;
- no blind ranking was performed;
- no full-resolution severe-artifact decision was made;
- the absence of a visual-stage failure must not be described as visual
  safety.

## Interpretation and branch

The result rejects independent promotion of this fixed sequential
composition, not the existence of useful deterministic film-inspired colour
operators. Its failure is specifically one of cross-population incremental
value and upper-envelope stability, while non-basic structure and clipping
safety remain encouraging.

The exact 17 images have now been observed by the evaluator. They may support
diagnosis or a separately declared development study, but they cannot be
relabelled as an untouched confirmation population for a successor. A future
leaf must introduce a genuinely different information structure or operator
hypothesis, freeze its own development/confirmation split, and preserve the
same severe-artifact veto. Retuning this composition against the 17 rows is
forbidden.

Authority:

- contract:
  `docs/planning/U5_R2AI1_DUAL_CHAMPION_INDEPENDENT_CONFIRMATION_CONTRACT.md`;
- decision:
  `configs/u5_r2ai1_dual_champion_independent_confirmation_decision_v1.json`;
- ignored full evidence:
  `outputs/u5_r2ai1_dual_champion_independent_confirmation_v1/`.
