# FilmFX Layer Results

> Created: 2026-05-27 on the Windows RTX machine.

## Layer Foundation

Implemented:

- `src/filmfx/layers.py`
  - `FilmLayer` schema,
  - layer validation,
  - alpha/residual safety metrics.
- `src/filmfx/compositor.py`
  - alpha,
  - screen,
  - additive,
  - residual,
  - soft-light compositing.
- `scripts/pipeline_filmfx_layers.py`
- `configs/filmfx_profiles.yaml`
- `scripts/smoke_filmfx_layers.py`

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_filmfx_layers.py
```

Expected:

```text
bounds=[13, 243]
warm_screen alpha_max=0.08, affected_percent=44.79
blue_residual residual_abs_max=0.015
```

Decision:

- The layer compositor foundation is ready for deterministic grain, halation,
  dust, scratches, and later AI-generated RGBA/residual layers.
- No film-effect layer is enabled by default yet.

## Deterministic Effect Layers

Implemented:

- `grain_residual_layer`
  - zero-mean high-frequency residual,
  - seed-controlled,
  - luminance-modulated.
- `halation_layer`
  - highlight and edge supported only,
  - red/orange screen layer,
  - bounded alpha,
  - continuous exposure-dependent scatter radius.
- `dust_scratch_layer`
  - sparse alpha overlay,
  - seed-controlled.

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_filmfx_effects.py
```

Expected:

```text
bounds=[4, 251]
grain residual_abs_max=0.0566, residual_abs_mean=0.0078
halation alpha_max=0.0088, affected_percent=29.85
dust_scratch alpha_max=0.0360, affected_percent=0.46
continuous halation smoke bounds=[4, 251], alpha_max=0.0137
```

Halation model update:

The first halation implementation used fixed small/broad blur radii. It now
computes a continuous per-pixel radius from highlight strength:

```text
radius = min_radius + (max_radius - min_radius) * highlight^radius_gamma
```

The implementation approximates spatially varying convolution by softly assigning
each source pixel into Gaussian scale-space, then summing the blurred support.
This avoids visible threshold bands while keeping the effect deterministic and
bounded.

Physical Halation V2 update:

`physical_halation_layer` adds a more explicit film-prior model:

- approximate scene-linear/log-exposure source map,
- separate `amplify` and `impact` controls,
- red dominant long-tail layer,
- green coupling only for stronger cores,
- blue leakage near zero,
- dark-background/high-contrast visibility weighting,
- `vision3_500t` and `cinestill_800t` style profiles.

Sweep outputs:

```text
outputs/eval/halation_v2/vision3_restrained/contact_sheet.png
outputs/eval/halation_v2/vision3_standard/contact_sheet.png
outputs/eval/halation_v2/cinestill_no_remjet/contact_sheet.png
outputs/eval/halation_v2/cinestill_aggressive/contact_sheet.png
outputs/eval/halation_v2/impact_low/contact_sheet.png
outputs/eval/halation_v2/impact_high/contact_sheet.png
```

Physical Halation V2.1 adds:

- absolute/no-normalization source mode for synthetic HDR diagnostics,
- optional `source_linear_rgb` input for tests where scene exposure is known,
- source-suppressed background estimation,
- stricter bright-background suppression,
- four-column contact sheets:

```text
original | halation on black | halation on white | combined
```

V2.1 outputs:

```text
outputs/eval/halation_v2p1/vision3_restrained/contact_sheet.png
outputs/eval/halation_v2p1/vision3_standard/contact_sheet.png
outputs/eval/halation_v2p1/cinestill_no_remjet/contact_sheet.png
outputs/eval/halation_v2p1/cinestill_aggressive/contact_sheet.png
outputs/eval/halation_v2p1/impact_low/contact_sheet.png
outputs/eval/halation_v2p1/impact_high/contact_sheet.png
outputs/eval/halation_v2p1_physics/exposure_radius_contact_sheet.png
```

V2.1 physics checks:

```text
radius_monotonic=true
radius gains px=9.85, 8.83, 8.55, 6.06, 5.60
dark_to_bright_radius_ratio=2.26
dark_to_bright_alpha_sum_ratio=1.72
blue_leakage_max=0.0
center_green_ratio_gt_outer_count=6/6
```

The integrated renderer now supports:

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --style vision3_500t `
  --halation 1.1 `
  --halation-model physical `
  --halation-profile cinestill_800t `
  --halation-impact 0.85 `
  --output output.png `
  --write-layers `
  --write-metrics
```

## AI Artifact Layer Prototype Decision

Decision: defer AI grain/halation/damage training until deterministic layers have
user visual approval and a richer patch dataset exists.

Allowed future AI output forms:

```text
grain: residual_rgb_or_luma + strength/mask
halation: RGBA layer with bounded alpha
dust/scratch: sparse RGBA alpha overlay
light leak/bloom: RGBA layer or bounded residual
```

Current rationale:

- Deterministic grain/halation/dust layers already satisfy the layer contract and
  are easier to inspect.
- There is no paired clean/film-effect layer dataset in the repo.
- Training AI layers against pseudo labels before visual approval would mostly
  imitate the deterministic simulator, so it adds complexity without a clear win.
- Any later AI layer generator must write independent layer views and pass the
  same alpha/residual bounds before compositing.

Status:

- AI grain generator: deferred.
- AI halation generator: deferred.
- AI transparent scratch/dust generator: deferred.
- Layer contract for all of the above: active and enforced by compositor design.
