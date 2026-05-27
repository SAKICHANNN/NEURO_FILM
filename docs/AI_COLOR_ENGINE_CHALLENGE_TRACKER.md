# AI Color Engine Challenge Tracker

> Created: 2026-05-27
>
> Purpose: autonomously tune and evaluate the #2 and #3 color-module candidates
> against the current #1 `safe_lab + safe-rich` baseline.

## Mission

Try to make these two color engines beat the current production baseline:

1. Current champion: `safe_lab + safe-rich`
2. Challenger A: Image-Adaptive Neural LUT
3. Challenger B: Local/Semantic bounded color maps

The goal is not to make a louder color grade. A challenger only wins if it keeps
the content-preservation and no-clipping guarantees while improving color
richness, film-stat proximity, or local naturalness.

## Non-Negotiable Constraints

1. No final RGB diffusion/img2img output.
2. No committed `.env`, private images, datasets, weights, caches, or generated outputs.
3. All challenger outputs must preserve image geometry and luminance detail.
4. PNG validation outputs must stay inside `[4, 251]` unless explicitly testing a failure.
5. Challengers must not increase neutral/skin contamination beyond the safe_lab baseline.
6. Every experiment must write ignored metrics/contact sheets and tracked conclusions.

## Champion Baseline

Champion command:

```powershell
python scripts/evaluate_color_pipeline.py --output-dir outputs/eval/baseline_saferich --profile configs/color_rendering_profiles.yaml --profile-name safe_rich
```

Champion summary from `docs/COLOR_BASELINE_STABILITY_RESULTS.md`:

| Style | New Clip | Bounds | Min L-SSIM | Mean L-SSIM | Max Neutral Contam. | Mean Chroma |
|------|:---:|:---:|---:|---:|---:|---:|
| ektar_100 | 0/20 | 4..251 | 0.9952 | 0.9991 | 0.025% | 15.881 |
| portra_400 | 0/20 | 4..251 | 0.9956 | 0.9991 | 0.003% | 15.869 |
| portra_800 | 0/20 | 4..251 | 0.9952 | 0.9991 | 0.010% | 16.357 |
| velvia_50 | 0/20 | 4..251 | 0.9955 | 0.9991 | 1.648% | 17.077 |
| vision3_250d | 0/20 | 4..251 | 0.9951 | 0.9991 | 0.022% | 16.480 |
| vision3_500t | 0/20 | 4..251 | 0.9950 | 0.9991 | 0.008% | 15.646 |

B&W stocks are excluded from the color-only win gate until a separate B&W tone
metric exists.

## Win Criteria

A challenger can be marked `promote_candidate` only if all are true on the
20-image rawpixls seed set for the six color stocks:

| Gate | Required |
|------|----------|
| New clipping | `0` new clipped pixels |
| Output bounds | all outputs inside `[4, 251]` |
| L-SSIM | `>= safe_lab_min - 0.0005`, and never below `0.995` |
| Neutral contamination | `<= safe_lab` per style, or justified by visual review |
| Banding | no worse histogram empty-bin score than safe_lab by more than 2% |
| High-frequency delta | no worse than safe_lab by more than 5% |
| Richness | mean chroma or film-stat proximity improves by at least 3% |
| Visual | contact sheet judged at least as natural as safe_lab |

If a challenger improves richness but misses a safety gate, mark it
`experimental`, not `promote_candidate`.

## Dependency Order

| Order | Task | Depends On | Status | Completion Test | Commit Node |
|:---:|------|------------|:---:|-----------------|-------------|
| 0 | Create tracker and branch | clean integration branch | done | branch `research/color-engines-beat-safe-lab` exists | `docs: add ai color engine challenge tracker` |
| 1 | Add challenger comparison evaluator | Order 0 | pending | evaluator compares safe_lab vs challenger metrics | `eval: add color engine challenger comparison` |
| 2 | Neural LUT larger imitation run | Order 1 | pending | all color stocks train/eval smoke with metrics | `research: tune neural lut challenger` |
| 3 | Neural LUT safety comparison | Order 2 | pending | report decides if #2 beats #1 | `research: evaluate neural lut against safe lab` |
| 4 | Local bounded map engine scaffold | Order 1 | pending | local maps render and pass no-clip smoke | `research: add local bounded color maps` |
| 5 | Tune local maps against safe_lab | Order 4 | pending | local maps improve richness while passing gates | `research: tune local color map challenger` |
| 6 | Final challenge verdict | Orders 3 and 5 | pending | result doc ranks #1/#2/#3 with evidence | `research: record color engine challenge verdict` |

## Challenger A: Image-Adaptive Neural LUT

Starting point:

- `src/models/color_lut/`
- `scripts/train_neural_lut.py`
- `docs/NEURAL_LUT_RESULTS.md`

Planned tuning:

1. Train on all six color stocks, not only Portra/Velvia.
2. Increase LUT/basis capacity cautiously.
3. Export predictions for seed eval images.
4. Evaluate with the same safety metrics as safe_lab.
5. Only add film-stat/style pressure after imitation is stable.

Failure modes:

- only imitates safe_lab without improving anything,
- introduces clipping after LUT interpolation,
- decreases L-SSIM through RGB interpolation artifacts,
- overfits the tiny seed set.

## Challenger B: Local/Semantic Bounded Maps

Starting point: not implemented as a distinct engine yet.

Planned scaffold:

```text
safe_lab output
  + bounded local maps:
      sky chroma/cyan control
      foliage green/yellow separation
      warm highlight bias
      neutral/skin protection masks
  -> gamut-safe compositor
```

Default map source:

- heuristic masks first, because they are inspectable;
- learned map predictor only after deterministic maps prove useful.

Failure modes:

- halos around object boundaries,
- dirty skin/neutral contamination,
- excessive landscape-only tuning that hurts general photos,
- no measurable improvement over safe_lab.

## Output Locations

Ignored outputs:

```text
outputs/eval/color_engine_challenge/
outputs/neural_lut/challenge_*/
outputs/local_color_maps/challenge_*/
```

Tracked result docs:

```text
docs/AI_COLOR_ENGINE_CHALLENGE_RESULTS.md
docs/AI_COLOR_ENGINE_CHALLENGE_TRACKER.md
```

## Manual Items

- User visual approval of side-by-side contact sheets.
- Any decision that a richer but less neutral-safe output is aesthetically
  preferable.

