# Neural Film LUT V2 Results

> Created: 2026-06-03
>
> Branch: `research/neural-film-lut-v2`
>
> Scope: Scheme A/B/C distilled experiments over the fixed 20-image rawpixls
> eval set and six color film stocks.

## What Was Built

| Scheme | Implementation | Purpose | Status |
|--------|----------------|---------|--------|
| A: Style-Separated SepLUT | `src/models/neural_film_lut/distilled_seplut.py` plus torch scaffold in `seplut.py` | separate per-stock 1D tone curves and 3D residual color volume | complete distilled run |
| B: Conditional NILUT proxy | `src/models/neural_film_lut/distilled_variants.py` | continuous implicit RGB residual, sampled/evaluated like a LUT renderer | complete distilled run |
| C: Context-aware 4D LUT proxy | `src/models/neural_film_lut/distilled_variants.py` | RGB residual volume weighted by soft deterministic contexts | complete distilled run |

Additional tooling:

- `scripts/build_neural_film_targets.py`
  - creates identity/safe/soft-film/full-film pseudo-target images.
- `scripts/summarize_neural_lut_results.py`
  - creates cross-run JSON/CSV summaries from eval outputs.
- `scripts/fit_distilled_film_lut.py`
  - fits Scheme A distilled SepLUT.
- `scripts/evaluate_distilled_film_lut.py`
  - evaluates Scheme A distilled SepLUT.
- `scripts/fit_distilled_neural_variants.py`
  - fits Scheme B and C distilled variants.
- `scripts/evaluate_distilled_neural_variants.py`
  - evaluates Scheme B and C distilled variants.

## Generated Outputs

Pseudo-targets:

```text
outputs/neural_film_lut_v2/targets_v1/
outputs/neural_film_lut_v2/targets_v1/targets_manifest.csv
```

Models:

```text
outputs/neural_film_lut_v2/scheme_a_distilled_v1/model.npz
outputs/neural_film_lut_v2/scheme_b_nilut_distilled_v1/model.npz
outputs/neural_film_lut_v2/scheme_c_context4d_distilled_v1/model.npz
```

Candidate summary:

```text
outputs/eval/neural_film_lut_v2/candidate_summary_v1.json
outputs/eval/neural_film_lut_v2/candidate_summary_v1.csv
```

Main visual review directories:

```text
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_s0p5/
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_s1p0/
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_s1p5/
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_s2p0/

outputs/eval/neural_film_lut_v2/scheme_b_nilut_distilled_v1_s0p5/
outputs/eval/neural_film_lut_v2/scheme_b_nilut_distilled_v1_s1p0/
outputs/eval/neural_film_lut_v2/scheme_b_nilut_distilled_v1_s1p5/
outputs/eval/neural_film_lut_v2/scheme_b_nilut_distilled_v1_s2p0/

outputs/eval/neural_film_lut_v2/scheme_c_context4d_distilled_v1_s0p5/
outputs/eval/neural_film_lut_v2/scheme_c_context4d_distilled_v1_s1p0/
outputs/eval/neural_film_lut_v2/scheme_c_context4d_distilled_v1_s1p5/
outputs/eval/neural_film_lut_v2/scheme_c_context4d_distilled_v1_s2p0/
```

Useful contact sheets:

```text
outputs/eval/neural_film_lut_v2/scheme_b_nilut_distilled_v1_s1p0/velvia_50/contact_sheet.png
outputs/eval/neural_film_lut_v2/scheme_c_context4d_distilled_v1_s1p0/velvia_50/contact_sheet.png
outputs/eval/neural_film_lut_v2/scheme_c_context4d_distilled_v1_s1p0/portra_400/contact_sheet.png
```

## Quantitative Summary

All listed runs produced zero new clipped pixels and stayed inside `[4,251]`.

| Run | Strength | Mean L-SSIM Proxy | Stock Chroma Range | Max Neutral Contam. |
|-----|---------:|------------------:|-------------------:|--------------------:|
| A SepLUT | 0.50 | 0.9445 | 1.8797 | 3.66 |
| A SepLUT | 1.00 | 0.8898 | 3.8818 | 28.69 |
| A SepLUT | 1.50 | 0.8364 | 5.7300 | 42.82 |
| A SepLUT | 2.00 | 0.7860 | 7.1739 | 40.85 |
| B NILUT | 0.50 | 0.9501 | 1.8326 | 0.79 |
| B NILUT | 1.00 | 0.9034 | 3.5368 | 9.35 |
| B NILUT | 1.50 | 0.8597 | 5.0584 | 26.03 |
| B NILUT | 2.00 | 0.8186 | 6.4178 | 29.90 |
| C Context4D | 0.50 | 0.9471 | 1.8655 | 4.57 |
| C Context4D | 1.00 | 0.8958 | 3.9348 | 22.79 |
| C Context4D | 1.50 | 0.8457 | 6.0317 | 41.73 |
| C Context4D | 2.00 | 0.7975 | 7.9412 | 48.51 |

Subtle A outputs were also exported:

```text
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_subtle_s0p15/
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_subtle_s0p25/
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_subtle_s0p35/
outputs/eval/neural_film_lut_v2/scheme_a_distilled_v1_subtle_s0p5/
```

## Visual Read

Scheme A:

- Film-stock identity is visible and strength is smooth.
- It no longer looks like only a saturation slider.
- Normal and strong settings can become heavy; `s0p25` to `s0p5` is the more usable band.

Scheme B:

- Best safety/style balance in the first full comparison.
- At `s1p0`, it gives visible stock-specific color without as much neutral damage as A/C.
- It is still a global implicit RGB transform, so local subject behavior is weaker than C.

Scheme C:

- Strongest stock separation and best evidence that context conditioning matters.
- Velvia, foliage, sky, and warm regions diverge more clearly from Portra/Vision3.
- Current deterministic context fit is too aggressive in neutral regions at `s1p0+`; it needs context protection before promotion.

## Current Ranking

1. **Scheme B NILUT `s1p0`**: best immediate candidate for visual review.
2. **Scheme A SepLUT `s0p35` or `s0p5`**: safest backup with good controllability.
3. **Scheme C Context4D `s0p5`**: promising architecture, but not ready above subtle strength.

## Manual Review Needed

The remaining manual part is visual preference, not implementation:

1. Review B `s1p0` contact sheets across all six stocks.
2. Compare C `s0p5` and `s1p0` for whether context-specific style is worth the added neutral risk.
3. Decide whether to promote B now, or use C as the next optimization target with neutral/skin masks.

## Technical Caveats

- The torch Scheme A scaffold exists, but this Windows environment showed slow/hanging torch behavior during earlier attempts. The completed A/B/C results are numpy distilled variants.
- Metrics are fast proxy metrics, not the full skimage safety evaluator. The final promote candidate should receive a full safety pass on a smaller subset first, then the full 20-image set if stable.
- B and C are architecture probes distilled from `film_response_v1_s1p0`; they are not trained on real film scans yet.
