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
halation alpha_max=0.0345, affected_percent=22.02
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
