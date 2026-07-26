# NFCM-A4 Known-Operator Quality Baseline

## Question

Can the v1 single-reference safe-Lab recipe recover one known global
photographic look across different image content?

This is a product-quality falsification leaf, not a film-stock authenticity
test. The target images were produced by the same existing deterministic
Velvia-look renderer, so the hidden target look is held constant while image
content changes.

## Frozen inputs

- Reference: styled target image `01` from
  `outputs/eval/baseline_current/velvia_50/after`.
- Sources: neutral images `01`, `02`, `03`, `05`, `07` and `09` from
  `outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/inputs`.
- Known targets: the corresponding six images from the same
  `baseline_current/velvia_50/after` run.
- Recipe: one fit from reference `01`, reused unchanged for all six sources.
- Output evidence is intentionally ignored under
  `outputs/reference_color_match_quality/velvia_known_operator_v1`.

The same-content `01` row is a positive control. The other five rows are the
cross-content test.

## Results

Pixel diagnostics compare each neutral source and reference-matched output to
its same-content known target in D65 CIELAB. Positive improvement means the
reference match reduced median Delta E76.

| Image | Source to target median | Match to target median | Median improvement | Match p95 |
|---|---:|---:|---:|---:|
| 01 positive control | 9.78 | 3.93 | +59.8% | 8.41 |
| 02 | 9.47 | 9.83 | -3.8% | 21.96 |
| 03 | 10.30 | 16.91 | -64.1% | 23.39 |
| 05 | 7.30 | 9.58 | -31.4% | 16.19 |
| 07 | 7.03 | 11.24 | -59.8% | 17.42 |
| 09 | 8.11 | 22.35 | -175.4% | 25.02 |

The positive control passes, but all five cross-content rows regress. Gamut
compression touched 21.1% to 63.2% of pixels in four of those five rows and
50.4%/63.2% in the two strongest failures.

Autonomous visual inspection agrees with the metric direction:

- image `02` becomes too dark and warm;
- image `03` loses the target's cooler ground/foliage relation and raises the
  flower scene globally;
- image `07` desaturates the blue sky and darkens the architecture;
- image `09` raises shadows and introduces a cyan/grey cast instead of the
  target's deep blue-black response.

No geometry or texture rewrite was observed, as expected from the pointwise
safe-Lab renderer. The failure is colour-operator identification, not spatial
generation.

## Decision

The v1 algorithm is rejected as the final photographic matcher. It remains a
safe deterministic fallback and product-contract baseline only.

The evidence distinguishes two properties:

1. same-content fitting can move toward a target;
2. reference-image global moments do not identify a shared look across
   different content.

Therefore parameter tuning, stronger global moment matching, or declaring the
safe-Lab recipe a champion is forbidden. A promotable challenger must estimate
a content-independent grade through a canonical pivot or an equivalent
identified representation, then emit a bounded explicit operator.

## Promotion gate

A challenger may replace v1 only if it:

- uses exactly one reference and one frozen operator/grade identity across N
  sources;
- improves the held-out cross-content aggregate over both neutral source and
  v1 without relying on the known targets at inference;
- passes severe-artifact, clipping/gamut, neutral, skin, sky, highlight and
  batch-consistency gates;
- preserves deterministic recipe replay and the existing transactional file
  boundary;
- retains the `reference-look` claim ceiling;
- reports synthetic/known-operator evidence separately from real-world blind
  aesthetic preference.

The main neuro-film W1 task owns the current reference-identifiability research.
This branch will consume a committed passing descriptor/head through an
adapter; it will not duplicate the W1 experiment. The standalone D-PCT task
continues to own RAW/HDR/video and portable media execution.
