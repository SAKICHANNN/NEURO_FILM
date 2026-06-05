# Halation System Specification

> Status: V2.3 rule-family implementation
>
> Primary code:
>
> - `src/filmfx/effects.py`
> - `src/filmfx/halation_controls.py`
> - `scripts/render_film.py`
> - `scripts/evaluate_physical_halation_v2.py`
>
> Generated review outputs:
>
> - `outputs/eval/halation_v2p3_families/`
> - `outputs/eval/halation_v2p3_family_physics/`

## Purpose

The halation system generates an inspectable film-effect layer that simulates
highlight-driven halation and then composites that layer back onto a
content-preserving color render.

The current system is deliberately not a generic red glow. It is a layered,
physically locked renderer with:

1. Discrete physical rule families.
2. Discrete backing/stock structure modes.
3. Discrete color-response laws.
4. A small set of linked user sliders.
5. An expert escape hatch for low-level experiments.
6. Four-column visual review outputs.

The guiding principle is:

```text
Do not let ordinary users freely combine low-level parameters in ways that
destroy the intended film-physics relationship.
```

The normal UI/API should expose the locked control surface. Expert controls are
for research/debugging only.

## Current Functional Scope

The system can currently:

- create a halation layer from an RGB image;
- estimate a source map from approximate scene-linear exposure;
- treat bright, specular, high-contrast regions as better halation sources;
- suppress halation on bright backgrounds;
- show more halation on dark/high-contrast regions;
- keep `amount` separate from radius controls;
- make visible radius grow because stronger exposure tails cross a visibility
  threshold, not because the amount slider directly widens blur kernels;
- emit either color-negative red/orange halation or B&W/density-domain halation;
- write layer previews and metrics through the integrated renderer;
- generate contact sheets with:

```text
original | halation on black | halation on white | combined
```

The current implementation is still a simplified post-color-render layer. The
longer-term ideal remains to insert halation in a fuller exposure/density/scan
pipeline.

## Truth Level / Evidence Level

This document is an engineering specification for the current renderer. It is
not a claim that the current numeric constants are fully measured film physics.

Use the following evidence levels when reading or extending this document:

| Level | Meaning | Examples In This Document |
|-------|---------|---------------------------|
| Code fact | Directly true of the checked-in implementation. | `PhysicalHalationControls` fields, CLI flags, function names, generated output paths, tested invariants. |
| Measured project result | Produced by local scripts and recorded in repo outputs/metrics. | V2.3 alpha/visible metrics, physics-suite radius monotonicity, contact-sheet paths. |
| External-source-supported claim | Supported by Kodak/CineStill/Dehancer references, but not necessarily measured inside this repo. | Anti-halation backing suppresses reflected light; no-remjet-like stocks can show stronger halation; Amplify and Impact should be separate controls. |
| Physically motivated model assumption | Reasonable model form derived from prompt/references, but simplified. | Red-layer-dominant color-negative backscatter; green-layer strong-core coupling; visible radius grows by threshold crossing. |
| Uncalibrated heuristic | Current numeric choice made for stable visual behavior, not fitted to real film patches. | Type-specific slider ranges, kernel weights, alpha caps, source threshold ranges, density tint values. |
| Future ideal | Desired architecture not yet implemented. | Full negative density / dye-density / scan transform; real-film patch calibration; RAW/HDR source exposure path in integrated renderer. |

When this document says a type is "Vision3-like", "CineStill-like",
"classic", or "B&W", it describes the intended rule-family behavior and current
heuristic parameter envelope. It does not mean the renderer has been calibrated
against measured samples of that exact stock.

## Source References And Model Constraints

This implementation is based on the project's `halationguide.md`, prior online
research, and current tracked results in `docs/PHYSICAL_HALATION_V2_TRACKER.md`.

Reference constraints used by the implementation:

| Source Type | Constraint |
|-------------|------------|
| Kodak motion-picture glossary | Halation is caused by light scattering/reflection through emulsion/base surfaces; anti-halation backing absorbs light that would otherwise reflect back into the emulsion. |
| Kodak Vision3 documentation | Standard Vision3-style film should be restrained by anti-halation/backing behavior. |
| CineStill remjet notes | No-remjet-like stocks can show much stronger red/orange highlight halation. |
| Dehancer docs/manual | Profiles are distinct from Source Limiter, Background Gain, Local/Global Diffusion, Amplify, Hue/Blue Comp, and Impact; Amplify and Impact have different roles. |
| User prompt / `halationguide.md` | Strength should mainly change scattered exposure coupling; the visible radius grows slowly because stronger tails cross a threshold. |

Key mathematical intuition:

```text
E_h(r) = A * alpha * E_s * K(r)
```

Where:

| Symbol | Meaning |
|--------|---------|
| `E_h(r)` | Halation exposure at distance `r`. |
| `A` | User physical amount/amplify. |
| `alpha` | Backscatter / anti-halation coupling. |
| `E_s` | Source exposure at the film plane. |
| `K(r)` | Mostly fixed scattering kernel for a given stock/format/rule family. |

Visible radius is thresholded:

```text
A * alpha * E_s * K(r) > T
```

For a Gaussian-like tail:

```text
r_vis ~= sigma * sqrt(2 * log(A * alpha * E_s / T))
```

For an exponential-like tail:

```text
r_vis ~= lambda * log(A * alpha * E_s / T)
```

This is why `amount` may visually expand the halo but must not directly edit
the blur radius. The tail becomes visible; the physical kernel is not being
linearly stretched by source brightness.

The equations above are model constraints, not a fitted film stock transfer
function. They protect the qualitative relationship between source exposure,
kernel tails, and visible radius, while leaving numeric calibration for future
real-film patch fitting.

## Architecture

High-level rendering path:

```text
input image
  -> content-preserving color render
  -> base RGB float array
  -> optional grain layer
  -> optional halation layer
  -> optional dust/scratch layer
  -> composite_layers(...)
  -> output image
```

Halation-specific locked path:

```text
PhysicalHalationControls
  -> resolve family/type/color-response
  -> resolve locked sliders to low-level physical parameters
  -> build_physical_halation_layer(...)
  -> physical_halation_layer(...) OR density_halation_layer(...)
  -> FilmLayer(mode="screen")
```

Relevant public functions:

| Function | File | Purpose |
|----------|------|---------|
| `PhysicalHalationControls` | `src/filmfx/halation_controls.py` | User-facing locked control dataclass. |
| `resolve_physical_halation_controls` | `src/filmfx/halation_controls.py` | Converts locked controls to low-level kwargs. |
| `describe_physical_halation_controls` | `src/filmfx/halation_controls.py` | Returns metadata for UI/metrics. |
| `build_physical_halation_layer` | `src/filmfx/halation_controls.py` | Selects the correct rule-family renderer. |
| `physical_halation_layer` | `src/filmfx/effects.py` | Color-negative red/green backscatter renderer. |
| `density_halation_layer` | `src/filmfx/effects.py` | B&W/density-domain renderer. |
| `halation_layer` | `src/filmfx/effects.py` | Legacy/simple glow layer. |

## Data Types

Halation layers use the repo's `FilmLayer` structure:

```text
name: string
mode: "screen"
rgb: float32 HxWx3, 0..1
alpha: float32 HxWx1, 0..1
```

The black-layer preview is:

```text
layer.rgb * layer.alpha
```

The white-layer preview is:

```text
white * (1 - alpha) + layer.rgb * alpha
```

The final composite uses `composite_layers`.

## Physical Rule Families

### `color_negative_backscatter`

This is the color negative family.

Conceptual equation:

```text
H_R = A * beta_R * V * conv(S, K_R)
H_G = A * beta_G * V * conv(S_high, K_G)
H_B = 0
```

Terms:

| Term | Meaning |
|------|---------|
| `A` | Physical amount/amplify. |
| `beta_R` | Red-layer backscatter coupling. |
| `beta_G` | Green-layer coupling for strong cores. |
| `V` | Visibility gate from dark background, local contrast, and skin protection. |
| `S` | Highlight source map. |
| `S_high` | Higher-threshold source map for green core coupling. |
| `K_R` | Multi-scale red scattering kernel. |
| `K_G` | Smaller green-core kernel. |

Current implementation details:

- source is derived from approximate scene-linear luminance;
- log exposure is measured relative to middle gray;
- source map uses a softplus threshold and gamma;
- source is weighted by specular/highlight confidence;
- edge confidence increases support around high-contrast highlights;
- background visibility suppresses halation on bright regions;
- red channel uses near/mid/tail/global diffusion terms;
- green channel is generated from a higher source threshold and smaller kernels;
- blue leakage is explicitly kept near zero.

This family is used by:

- `vision3_ahu`
- `cinestill_no_remjet`
- `classic_dense_base`

### `bw_density_halation`

This is a separate black-and-white / density-domain family.

Conceptual equation:

```text
H_density = A * V * conv(S, K_density)
```

It intentionally does not expose:

- `profile`
- `hue_green`
- `no_remjet`

Those are color-negative concepts and should not be forced onto B&W density
halation.

Current implementation details:

- source map still comes from approximate scene-linear luminance;
- the kernel is wider/softer and neutral-density oriented;
- alpha is computed from a density-like exposure term;
- layer color is controlled by `density_tint`;
- color response can be neutral or warm-neutral.

This family is used by:

- `bw_clear_base`

## Discrete Modes

### `halation_model`

CLI values:

```text
simple
physical
```

| Value | Meaning | Recommended UI Exposure |
|-------|---------|-------------------------|
| `simple` | Legacy display-space glow. | Hide or put under debug/legacy. |
| `physical` | Current physically locked system. | Default. |

### `halation_model_family`

CLI/API values:

```text
auto
color_negative_backscatter
bw_density_halation
```

Recommended GUI behavior:

- Use `auto` by default.
- Infer family from `halation_type`.
- Only expose explicit family override in a developer/debug panel.

### `halation_type`

CLI/API values:

```text
auto
vision3_ahu
cinestill_no_remjet
classic_dense_base
bw_clear_base
```

Detailed behavior:

| Type | Rule Family | Intended Meaning | Visual Behavior |
|------|-------------|------------------|-----------------|
| `auto` | inferred from profile | Compatibility helper. | `cinestill_800t` -> no-remjet; `vision3_500t` -> AHU; generic -> dense base. |
| `vision3_ahu` | `color_negative_backscatter` | Anti-halation-undercoat / standard Vision3-like restrained behavior. | High trigger threshold, lower backscatter, lower alpha cap, subtle red/orange edge. |
| `cinestill_no_remjet` | `color_negative_backscatter` | No-remjet / CineStill-like behavior. | Stronger red/orange halo, lower source threshold, higher backscatter, broader global diffusion. |
| `classic_dense_base` | `color_negative_backscatter` | Softer classic negative / dense-base behavior. | Softer, more diffuse, less aggressive than no-remjet. |
| `bw_clear_base` | `bw_density_halation` | B&W / clear-base density glow. | Neutral or warm-neutral glow, no red/green-layer law. |

These modes are current model categories, not measured stock definitions. Their
names indicate the intended physical behavior envelope used by the renderer.
The numeric ranges below are uncalibrated heuristics until real-film patch
fitting is added.

### `halation_color_response`

CLI/API values:

```text
red_orange_core
deep_red
amber_core
neutral_density
warm_neutral_density
```

Detailed behavior:

| Color Response | Valid Family | Meaning |
|----------------|--------------|---------|
| `red_orange_core` | `color_negative_backscatter` | Default color-negative behavior: red outer halo, orange-ish strong core. |
| `deep_red` | `color_negative_backscatter` | Lower green core participation; redder and less yellow. |
| `amber_core` | `color_negative_backscatter` | Higher green core participation; stronger orange/amber core while retaining red outer halo. |
| `neutral_density` | `bw_density_halation` | Pure neutral density glow. |
| `warm_neutral_density` | `bw_density_halation` | Slightly warm density glow. |

Forbidden/unsupported in normal GUI:

```text
blue halation
cyan halation
purple halation
arbitrary hue wheel
```

Reason: blue/cyan/purple glow is usually bloom, lens flare, sensor flare, or
creative optical scatter, not physically constrained film-base halation in this
model.

This is a product/GUI rule. The renderer is intentionally conservative, but the
GUI should still enforce valid combinations before calling the renderer so users
do not create settings that look physically meaningful while actually relying on
fallback behavior.

## Locked Slider Surface

The locked control surface is the default for `--halation-model physical`.

Dataclass fields:

```python
PhysicalHalationControls(
    model_family="auto",
    halation_type="auto",
    color_response="red_orange_core",
    profile="cinestill_800t",
    amount=1.0,
    impact=0.85,
    anti_halation=0.75,
    source_selectivity=0.45,
    diffusion=0.55,
    warm_core=0.45,
    background_visibility=0.75,
    source_normalization="percentile",
)
```

### `amount`

Recommended UI:

```text
slider 0.0..2.4
default 1.0
display label: Amount
```

Meaning:

- physical secondary-exposure coupling;
- maps only to `amplify`;
- should make halos brighter and make tails more visible;
- must not directly change kernel radius.

Invariant:

```text
Changing amount changes only amplify.
```

### `impact`

Recommended UI:

```text
slider 0.0..1.0
default 0.85
display label: Impact
```

Meaning:

- final display mix;
- similar to opacity/transparency;
- changes how much of the formed layer is visible;
- does not alter source detection, geometry, or color physics.

Invariant:

```text
Changing impact changes only impact.
```

### `anti_halation`

Recommended UI:

```text
slider 0.0..1.0
default 0.75
display label: Anti-Halation Loss / No-Remjet
show only for color-negative family
```

Meaning:

- controls backscatter coupling range inside the selected type;
- for `vision3_ahu`, the allowed range remains restrained;
- for `cinestill_no_remjet`, the allowed range is much stronger;
- hidden or disabled for `bw_density_halation`.

Invariant:

```text
Changing anti_halation changes only no_remjet/backscatter coupling.
```

### `source_selectivity`

Recommended UI:

```text
slider 0.0..1.0
default 0.45
display label: Source Selectivity
```

Meaning:

- controls how bright/selective a source must be before it creates halation;
- maps to `source_limiter_stops`;
- high value: fewer sources, only very bright/specular highlights;
- low value: more highlights can create halation.

Invariant:

```text
Changing source_selectivity changes only source_limiter_stops.
```

### `diffusion`

Recommended UI:

```text
slider 0.0..1.0
default 0.55
display label: Diffusion
```

Meaning:

- physical/geometric scatter scale;
- the only user slider that directly changes radius-related terms;
- maps to `local_diffusion` and `global_diffusion`;
- the exact resolved range depends on `halation_type`.

Invariant:

```text
Changing diffusion changes only local_diffusion and global_diffusion.
```

### `warm_core`

Recommended UI:

```text
slider 0.0..1.0
default 0.45
display label: Warm Core
show only for color-negative family
```

Meaning:

- controls green-layer coupling in strong highlight cores;
- makes the strongest core more orange/amber;
- not a global hue wheel;
- hidden or disabled for `bw_density_halation`.

Invariant:

```text
Changing warm_core changes only hue_green in color-negative family.
```

### `background_visibility`

Recommended UI:

```text
slider 0.0..1.0
default 0.75
display label: Background Visibility
```

Meaning:

- controls how strongly dark/high-contrast surroundings reveal halation;
- maps to `background_gain` and `background_luma_target`;
- high value: dark-side halation becomes more apparent;
- low value: halation is more suppressed.

Invariant:

```text
Changing background_visibility changes only background_gain and
background_luma_target.
```

### `source_normalization`

Recommended UI:

```text
advanced select: percentile | none
default percentile
```

Meaning:

- `percentile`: stable preview behavior on ordinary images;
- `none`: absolute/source-linear diagnostics, useful for synthetic HDR tests.

Normal GUI should keep this hidden and fixed at `percentile`.

## Low-Level Expert Controls

Expert mode is entered through:

```text
--halation-expert-controls
--halation-control-mode expert
```

Expert controls in `scripts/render_film.py`:

| CLI Flag | Meaning |
|----------|---------|
| `--halation-source-normalization` | Source normalization mode. |
| `--halation-source-limiter` | Low-level source threshold. |
| `--halation-local-diffusion` | Low-level local diffusion scale. |
| `--halation-global-diffusion` | Low-level global glare term. |
| `--halation-hue-green` | Low-level green-layer coupling. |
| `--halation-background-gain` | Low-level background gate steepness. |
| `--halation-background-luma-target` | Low-level background luma target. |
| `--halation-no-remjet` | Low-level backscatter coupling override. |

Expert mode currently always calls `physical_halation_layer`, so it is a
color-negative expert path. It does not expose `density_halation_layer` expert
kwargs directly.

Recommendation:

- Keep expert controls hidden behind a disclosure panel.
- Label them as research/debug controls.
- Do not save expert presets as ordinary user presets unless they also pass
  visual and metric review.

## Resolved Parameter Ranges By Type

These are internal ranges used by `resolve_physical_halation_controls`.

Evidence level: code fact for the current resolver values; uncalibrated
heuristic for any claim that the numbers match real film. These ranges were
chosen to preserve the intended behavior ordering and produce reviewable
outputs, not to serve as measured stock data.

### `vision3_ahu`

| Resolved Field | Range / Value |
|----------------|---------------|
| family | `color_negative_backscatter` |
| profile | `vision3_500t` |
| `no_remjet` | 0.22..0.54 |
| `local_diffusion` | 0.68..1.16 |
| `global_diffusion` | 0.04..0.15 |
| `hue_green` base range | 0.08..0.26 |
| `source_limiter_stops` | 2.35..3.45 |
| `source_softness` | 0.44 |
| `source_gamma` | 1.48 |
| `output_alpha_cap` | 0.20 |

Expected visual, not measured stock claim:

- very restrained;
- high trigger threshold;
- low visible mean;
- good for standard Vision3-style behavior.

### `cinestill_no_remjet`

| Resolved Field | Range / Value |
|----------------|---------------|
| family | `color_negative_backscatter` |
| profile | `cinestill_800t` |
| `no_remjet` | 0.78..1.26 |
| `local_diffusion` | 0.95..1.75 |
| `global_diffusion` | 0.12..0.38 |
| `hue_green` base range | 0.18..0.58 |
| `source_limiter_stops` | 1.35..2.75 |
| `source_softness` | 0.42 |
| `source_gamma` | 1.45 |
| `output_alpha_cap` | 0.32 |

Expected visual, not measured stock claim:

- strong red/orange halation;
- especially visible around bright light on dark backgrounds;
- lower source threshold and stronger backscatter than Vision3.

### `classic_dense_base`

| Resolved Field | Range / Value |
|----------------|---------------|
| family | `color_negative_backscatter` |
| profile | `generic` |
| `no_remjet` | 0.34..0.84 |
| `local_diffusion` | 1.05..1.82 |
| `global_diffusion` | 0.08..0.26 |
| `hue_green` base range | 0.10..0.36 |
| `source_limiter_stops` | 1.85..3.10 |
| `source_softness` | 0.50 |
| `source_gamma` | 1.36 |
| `output_alpha_cap` | 0.24 |

Expected visual, not measured stock claim:

- softer and more diffuse;
- less aggressive than no-remjet;
- less clean/restrained than Vision3 AHU.

### `bw_clear_base`

| Resolved Field | Range / Value |
|----------------|---------------|
| family | `bw_density_halation` |
| `local_diffusion` | 0.90..1.92 |
| `global_diffusion` | 0.10..0.36 |
| `source_limiter_stops` | 1.55..3.00 |
| `source_softness` | 0.46 |
| `source_gamma` | 1.35 |
| `density_tint` | from color response |
| `output_alpha_cap` | 0.28 |

Expected visual, not measured stock claim:

- neutral or warm density glow;
- no red/green hue-radius curve;
- not a desaturated version of color-negative halation.

## CLI Integration

### Basic Physical Halation

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --style vision3_500t `
  --halation 1.0 `
  --halation-model physical `
  --halation-physics-lock `
  --halation-type cinestill_no_remjet `
  --halation-color-response red_orange_core `
  --output output.png `
  --write-layers `
  --write-metrics
```

### Amber Core CineStill-Like

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --style vision3_500t `
  --halation 1.2 `
  --halation-model physical `
  --halation-physics-lock `
  --halation-type cinestill_no_remjet `
  --halation-color-response amber_core `
  --halation-anti-halation 0.84 `
  --halation-source-selectivity 0.46 `
  --halation-diffusion 0.56 `
  --halation-warm-core 0.58 `
  --halation-background-visibility 0.78 `
  --halation-impact 0.88 `
  --output outputs\integration\render_film_halation_v2p3_amber_core_smoke.png `
  --write-layers `
  --write-metrics
```

### B&W Density Halation

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --style vision3_500t `
  --halation 1.18 `
  --halation-model physical `
  --halation-physics-lock `
  --halation-type bw_clear_base `
  --halation-color-response neutral_density `
  --halation-source-selectivity 0.50 `
  --halation-diffusion 0.66 `
  --halation-background-visibility 0.76 `
  --halation-impact 0.84 `
  --output outputs\integration\render_film_halation_v2p3_bw_density_smoke.png `
  --write-layers `
  --write-metrics
```

### Expert Mode

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --style vision3_500t `
  --halation 1.1 `
  --halation-model physical `
  --halation-expert-controls `
  --halation-profile cinestill_800t `
  --halation-source-limiter 1.9 `
  --halation-local-diffusion 1.25 `
  --halation-global-diffusion 0.22 `
  --halation-hue-green 0.32 `
  --halation-background-gain 1.45 `
  --halation-background-luma-target 0.20 `
  --halation-no-remjet 1.0 `
  --output output.png `
  --write-layers `
  --write-metrics
```

## Python API Integration

Locked path:

```python
import numpy as np

from src.filmfx import (
    PhysicalHalationControls,
    build_physical_halation_layer,
    composite_layers,
)

base_rgb = np.asarray(image, dtype=np.float32) / 255.0

controls = PhysicalHalationControls(
    halation_type="cinestill_no_remjet",
    color_response="amber_core",
    amount=1.2,
    impact=0.88,
    anti_halation=0.84,
    source_selectivity=0.46,
    diffusion=0.56,
    warm_core=0.58,
    background_visibility=0.78,
)

layer = build_physical_halation_layer(base_rgb, controls)
combined = composite_layers(base_rgb, [layer], output_margin=4)
```

Metrics/metadata:

```python
from src.filmfx import (
    describe_physical_halation_controls,
    layer_metrics,
    resolve_physical_halation_controls,
)

metadata = describe_physical_halation_controls(controls)
resolved = resolve_physical_halation_controls(controls)
metrics = layer_metrics(layer)
```

## Output Artifacts

Integrated renderer can write:

| Artifact | Trigger | Meaning |
|----------|---------|---------|
| output image | `--output` | Final composite. |
| layer preview folder | `--write-layers` | Layer previews saved next to output. |
| metrics JSON | `--write-metrics` | Bounds, layer metrics, halation metadata, and resolved kwargs. |

Metrics contain:

```json
{
  "halation_control_mode": "locked",
  "halation_metadata": {
    "model_family": "color_negative_backscatter",
    "halation_type": "cinestill_no_remjet",
    "color_response": "amber_core",
    "profile": "cinestill_800t"
  },
  "halation_resolved": {
    "profile": "cinestill_800t",
    "source_normalization": "percentile",
    "amplify": 1.2,
    "impact": 0.88,
    "source_limiter_stops": 1.994,
    "local_diffusion": 1.398,
    "global_diffusion": 0.2656,
    "hue_green": 0.5562,
    "background_gain": 1.63,
    "background_luma_target": 0.218,
    "no_remjet": 1.1832,
    "output_alpha_cap": 0.32
  }
}
```

For B&W/density family, `halation_resolved` intentionally differs:

```json
{
  "source_normalization": "percentile",
  "amplify": 1.18,
  "impact": 0.84,
  "source_limiter_stops": 2.275,
  "local_diffusion": 1.5732,
  "global_diffusion": 0.2716,
  "background_gain": 1.61,
  "background_luma_target": 0.216,
  "density_tint": [1.0, 1.0, 1.0],
  "output_alpha_cap": 0.28
}
```

No `hue_green`, `no_remjet`, or `profile` appears in the resolved B&W surface.

## Evaluation And Contact Sheets

Primary evaluator:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_physical_halation_v2.py `
  --limit 20 `
  --max-side 768 `
  --include-diagnostics `
  --control-mode family `
  --output-root outputs\eval\halation_v2p3_families
```

Generated V2.3 contact sheets:

```text
outputs/eval/halation_v2p3_families/family_vision3_ahu_red_orange/contact_sheet.png
outputs/eval/halation_v2p3_families/family_cinestill_no_remjet_deep_red/contact_sheet.png
outputs/eval/halation_v2p3_families/family_cinestill_no_remjet_amber_core/contact_sheet.png
outputs/eval/halation_v2p3_families/family_classic_dense_base_soft_red/contact_sheet.png
outputs/eval/halation_v2p3_families/family_bw_clear_base_neutral_density/contact_sheet.png
outputs/eval/halation_v2p3_families/family_bw_clear_base_warm_density/contact_sheet.png
```

Each run includes:

```text
original/
layers_black/
layers_white/
combined/
metrics.json
contact_sheet.png
```

Physics suite:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_halation_physics_suite.py `
  --output-root outputs\eval\halation_v2p3_family_physics `
  --profile cinestill_800t `
  --amplify 1.15 `
  --impact 0.90 `
  --source-limiter 1.8 `
  --local-diffusion 1.25 `
  --global-diffusion 0.18 `
  --hue-green 0.34 `
  --background-gain 1.45 `
  --background-luma-target 0.20
```

Current V2.3 physics check:

```text
radius_monotonic=true
radius gains=9.85, 8.83, 8.55, 6.06, 5.60 px
dark_to_bright_radius_ratio=2.26
blue_leakage_max=0.0
```

## Current V2.3 Visual Review Runs

| Run | Family | Type | Color Response | Alpha Max | Alpha Mean | Visible Mean | Bounds |
|-----|--------|------|----------------|----------:|-----------:|-------------:|--------|
| `family_vision3_ahu_red_orange` | `color_negative_backscatter` | `vision3_ahu` | `red_orange_core` | 0.0478 | 0.00026 | 0.04% | 4..251 |
| `family_cinestill_no_remjet_deep_red` | `color_negative_backscatter` | `cinestill_no_remjet` | `deep_red` | 0.3200 | 0.00708 | 16.81% | 4..251 |
| `family_cinestill_no_remjet_amber_core` | `color_negative_backscatter` | `cinestill_no_remjet` | `amber_core` | 0.3200 | 0.00708 | 16.81% | 4..251 |
| `family_classic_dense_base_soft_red` | `color_negative_backscatter` | `classic_dense_base` | `red_orange_core` | 0.1418 | 0.00211 | 4.94% | 4..251 |
| `family_bw_clear_base_neutral_density` | `bw_density_halation` | `bw_clear_base` | `neutral_density` | 0.2132 | 0.00521 | 13.87% | 4..251 |
| `family_bw_clear_base_warm_density` | `bw_density_halation` | `bw_clear_base` | `warm_neutral_density` | 0.2132 | 0.00521 | 13.87% | 4..251 |

Interpreting identical metrics:

- `deep_red` and `amber_core` can share identical alpha/visible metrics because
  only the color-response law changes.
- `neutral_density` and `warm_neutral_density` can share identical alpha/visible
  metrics because only the density tint changes.

## Tests

Current targeted tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_color_baseline_safety.py tests\test_halation_controls.py
```

Current result:

```text
8 passed
```

Important invariants tested:

- `amount` changes only `amplify`;
- `impact` changes only `impact`;
- `diffusion` changes only `local_diffusion` and `global_diffusion`;
- `anti_halation` changes only `no_remjet`;
- `source_selectivity` changes only `source_limiter_stops`;
- `background_visibility` changes only background gate parameters;
- color response changes color law, not geometry;
- B&W/density family does not expose color-negative-only resolved parameters.

## Known Limitations

1. The current renderer uses approximate scene-linear reconstruction from RGB
   unless `source_linear_rgb` is provided.
2. The default integrated renderer applies halation after the safe Lab color
   render, not inside a full negative-density/scan transform.
3. Directional dark-side halation is approximated by visibility weighting after
   symmetric convolution, not by fully directional scatter transport.
4. Expert mode currently routes only to the color-negative renderer.
5. Real-film patch calibration is not done.
6. Video temporal consistency is not evaluated.
7. Layer output is screen-composited; a future density pipeline may need a
   different compositor or pre-display working space.
8. `bw_density_halation` is a separate rule family, but it is still an
   uncalibrated density-like screen layer, not a real B&W sensitometric model.
9. Some invalid GUI combinations are intentionally described as forbidden for
   product UX even though the current Python resolver may tolerate a fallback
   tint in the B&W family.

## Future Work

Recommended next steps:

1. Add GUI preview toggles for combined / layer on black / layer on white.
2. Add family-specific default presets for user-facing film stocks.
3. Add a real-film halo patch calibration dataset and fitter.
4. Add optional `source_linear_rgb` path to integrated renderer for RAW/HDR
   inputs.
5. Add full density-domain insertion before display color transform.
6. Add temporal metrics for video sequences.
7. Add stricter GUI validation so invalid type/color-response combinations are
   impossible to select.
8. Optionally make invalid API combinations raise errors instead of falling back
   silently, once GUI and CLI compatibility requirements are settled.

## Extreme GUI Integration Guide

This section is intentionally detailed. It is meant to be handed directly to a
GUI implementer.

### Product-Level UX Principle

The GUI must guide users through physical choices first and continuous tweaks
second.

Correct mental model:

```text
Choose what physical halation family this is.
Then choose the stock/backing behavior.
Then choose the bounded color response.
Then tweak amount/diffusion/visibility inside that physical envelope.
```

Incorrect mental model:

```text
Give user many independent sliders and a hue wheel.
```

### Recommended Panel Structure

Panel title:

```text
Halation
```

Top row:

```text
[Enable Halation] [Physical Lock]
```

Defaults:

```text
Enable Halation: off
Physical Lock: on
```

When `Enable Halation` is off:

- do not build a halation layer;
- keep controls visible but disabled if that helps the user understand saved
  settings;
- final CLI should omit `--halation` or set `--halation 0`.

When `Physical Lock` is on:

- use `PhysicalHalationControls`;
- call `build_physical_halation_layer`;
- expose only safe controls.

When `Physical Lock` is off:

- show an "Expert" warning;
- route through `--halation-expert-controls`;
- expose low-level controls.

### GUI Control Layout

Recommended order:

```text
Halation
  Enable Halation
  Physical Lock

  Type
    Vision3 AHU
    CineStill No-Remjet
    Classic Dense Base
    B&W Clear Base

  Color Response
    Red / Orange Core
    Deep Red
    Amber Core
    Neutral Density
    Warm Density

  Amount
  Impact
  Source Selectivity
  Diffusion
  Background Visibility
  Warm Core
  Anti-Halation / No-Remjet

  Preview
    Combined
    Layer on Black
    Layer on White

  Advanced
    Source Normalization
    Expert Controls
```

### Control Widgets

Use:

- checkbox/toggle for `Enable Halation`;
- checkbox/toggle for `Physical Lock`;
- segmented control or dropdown for `halation_type`;
- segmented control or dropdown for `halation_color_response`;
- sliders for continuous values;
- disclosure panel for advanced/expert controls;
- icon buttons or tabs for preview mode.

Do not use:

- arbitrary color wheel for halation hue;
- arbitrary RGB color picker;
- per-channel low-level sliders in normal mode;
- radius slider named "radius" in normal mode.

### User-Facing Labels

Internal -> GUI label:

| Internal | GUI Label |
|----------|-----------|
| `halation_type` | Type |
| `vision3_ahu` | Vision3 AHU |
| `cinestill_no_remjet` | CineStill No-Remjet |
| `classic_dense_base` | Classic Dense Base |
| `bw_clear_base` | B&W Clear Base |
| `halation_color_response` | Color Response |
| `red_orange_core` | Red / Orange Core |
| `deep_red` | Deep Red |
| `amber_core` | Amber Core |
| `neutral_density` | Neutral Density |
| `warm_neutral_density` | Warm Density |
| `amount` | Amount |
| `impact` | Impact |
| `source_selectivity` | Source Selectivity |
| `diffusion` | Diffusion |
| `warm_core` | Warm Core |
| `background_visibility` | Background Visibility |
| `anti_halation` | Anti-Halation Loss |

Tooltip copy:

| Control | Tooltip |
|---------|---------|
| Amount | Changes scattered exposure coupling. It does not directly widen the blur radius. |
| Impact | Changes final layer mix after the halation is formed. |
| Source Selectivity | Controls how bright a source must be before it produces halation. |
| Diffusion | Controls the physical scatter scale. This is the geometry slider. |
| Warm Core | Controls orange/amber core participation in strong highlights. |
| Background Visibility | Controls how strongly dark/high-contrast backgrounds reveal the halo. |
| Anti-Halation Loss | Controls how much anti-halation suppression is lost, similar to no-remjet behavior. |

### Valid UI Combinations

When user selects:

```text
Type = Vision3 AHU
```

Set:

```text
halation_type = "vision3_ahu"
model_family = "auto"
allowed color responses:
  - red_orange_core
  - deep_red
  - amber_core
show:
  - warm_core
  - anti_halation
hide:
  - neutral_density
  - warm_neutral_density
```

Recommended defaults:

```text
amount = 0.88
impact = 0.82
anti_halation = 0.22
source_selectivity = 0.72
diffusion = 0.40
warm_core = 0.30
background_visibility = 0.68
color_response = red_orange_core
```

When user selects:

```text
Type = CineStill No-Remjet
```

Set:

```text
halation_type = "cinestill_no_remjet"
model_family = "auto"
allowed color responses:
  - red_orange_core
  - deep_red
  - amber_core
show:
  - warm_core
  - anti_halation
```

Recommended defaults:

```text
amount = 1.20
impact = 0.88
anti_halation = 0.84
source_selectivity = 0.46
diffusion = 0.56
warm_core = 0.58 for amber_core, 0.34 for deep_red
background_visibility = 0.78
color_response = amber_core or red_orange_core
```

When user selects:

```text
Type = Classic Dense Base
```

Set:

```text
halation_type = "classic_dense_base"
model_family = "auto"
allowed color responses:
  - red_orange_core
  - deep_red
  - amber_core
show:
  - warm_core
  - anti_halation
```

Recommended defaults:

```text
amount = 1.05
impact = 0.82
anti_halation = 0.52
source_selectivity = 0.56
diffusion = 0.68
warm_core = 0.30
background_visibility = 0.70
color_response = red_orange_core
```

When user selects:

```text
Type = B&W Clear Base
```

Set:

```text
halation_type = "bw_clear_base"
model_family = "auto"
allowed color responses:
  - neutral_density
  - warm_neutral_density
hide:
  - warm_core
  - anti_halation
```

Recommended defaults:

```text
amount = 1.18
impact = 0.84
source_selectivity = 0.50
diffusion = 0.66
background_visibility = 0.76
color_response = neutral_density
```

### Color Response Switching Rules

If current `halation_type` is color-negative:

```text
vision3_ahu
cinestill_no_remjet
classic_dense_base
```

Then valid color responses are:

```text
red_orange_core
deep_red
amber_core
```

If current color response is invalid after switching type, auto-correct:

```text
neutral_density -> red_orange_core
warm_neutral_density -> red_orange_core
```

If current `halation_type` is:

```text
bw_clear_base
```

Then valid color responses are:

```text
neutral_density
warm_neutral_density
```

Auto-correct:

```text
red_orange_core -> neutral_density
deep_red -> neutral_density
amber_core -> warm_neutral_density
```

### Slider Ranges And Steps

Recommended GUI ranges:

| Control | Range | Step | Default Source |
|---------|-------|------|----------------|
| Amount | 0.0..2.4 | 0.01 | type preset |
| Impact | 0.0..1.0 | 0.01 | type preset |
| Source Selectivity | 0.0..1.0 | 0.01 | type preset |
| Diffusion | 0.0..1.0 | 0.01 | type preset |
| Background Visibility | 0.0..1.0 | 0.01 | type preset |
| Warm Core | 0.0..1.0 | 0.01 | type preset |
| Anti-Halation Loss | 0.0..1.0 | 0.01 | type preset |

Do not expose low-level resolved values as primary sliders. If the UI needs to
show them for debugging, put them in a read-only "Resolved Parameters" panel.

### Recommended Preset Buttons

Optional quick presets:

```text
Clean
Balanced
Strong
Extreme
```

These should change locked sliders but not switch the physical type unless the
user explicitly selects a different type.

Example for `cinestill_no_remjet`:

| Preset | Amount | Impact | Anti-Halation | Selectivity | Diffusion | Warm Core | Background |
|--------|-------:|-------:|---------------:|------------:|----------:|----------:|-----------:|
| Clean | 0.85 | 0.75 | 0.65 | 0.58 | 0.42 | 0.35 | 0.70 |
| Balanced | 1.20 | 0.88 | 0.84 | 0.46 | 0.56 | 0.50 | 0.78 |
| Strong | 1.55 | 0.95 | 1.00 | 0.36 | 0.70 | 0.62 | 0.86 |
| Extreme | 1.90 | 1.00 | 1.00 | 0.28 | 0.82 | 0.70 | 0.92 |

### Preview Modes

The preview mode should not change the rendered layer; it should only change
display.

Preview modes:

```text
Combined
Layer on Black
Layer on White
Original
```

Mapping:

| Preview | Display |
|---------|---------|
| `Combined` | `composite_layers(base, [halation_layer])` |
| `Layer on Black` | `layer.rgb * layer.alpha` |
| `Layer on White` | `1 * (1 - alpha) + layer.rgb * alpha` |
| `Original` | base image |

Why both black and white matter:

- black background reveals faint red/density halos;
- white background exposes alpha shape and tint without dark-scene bias;
- combined preview shows actual output but can hide weak layers.

### Debounced Rendering

Recommended behavior:

```text
slider drag:
  update lightweight preview at reduced resolution after 80-150 ms debounce

slider release:
  render full preview

export:
  render full resolution
```

The current implementation is CPU/NumPy/SciPy based. Large previews can be slow.
Use thumbnail preview for interactive UI.

### GUI State Object

Suggested frontend state:

```json
{
  "enabled": true,
  "physicalLock": true,
  "modelFamily": "auto",
  "type": "cinestill_no_remjet",
  "colorResponse": "amber_core",
  "amount": 1.2,
  "impact": 0.88,
  "antiHalation": 0.84,
  "sourceSelectivity": 0.46,
  "diffusion": 0.56,
  "warmCore": 0.58,
  "backgroundVisibility": 0.78,
  "sourceNormalization": "percentile",
  "previewMode": "combined"
}
```

Mapping to Python:

```python
controls = PhysicalHalationControls(
    model_family=state["modelFamily"],
    halation_type=state["type"],
    color_response=state["colorResponse"],
    amount=state["amount"],
    impact=state["impact"],
    anti_halation=state["antiHalation"],
    source_selectivity=state["sourceSelectivity"],
    diffusion=state["diffusion"],
    warm_core=state["warmCore"],
    background_visibility=state["backgroundVisibility"],
    source_normalization=state["sourceNormalization"],
)
```

### GUI To CLI Mapping

If the GUI shells out to `render_film.py`, map:

| GUI State | CLI |
|-----------|-----|
| enabled true | `--halation <amount>` |
| physicalLock true | `--halation-physics-lock` |
| modelFamily | `--halation-model-family` |
| type | `--halation-type` |
| colorResponse | `--halation-color-response` |
| impact | `--halation-impact` |
| antiHalation | `--halation-anti-halation` |
| sourceSelectivity | `--halation-source-selectivity` |
| diffusion | `--halation-diffusion` |
| warmCore | `--halation-warm-core` |
| backgroundVisibility | `--halation-background-visibility` |
| sourceNormalization | `--halation-source-normalization` |

Always include:

```text
--halation-model physical
```

When disabled:

```text
omit --halation
```

or:

```text
--halation 0
```

### GUI Validation Rules

Validation should happen before calling the renderer.

Rules:

```text
amount: clamp 0.0..2.4
impact: clamp 0.0..1.0
antiHalation: clamp 0.0..1.0
sourceSelectivity: clamp 0.0..1.0
diffusion: clamp 0.0..1.0
warmCore: clamp 0.0..1.0
backgroundVisibility: clamp 0.0..1.0
```

Combination rules:

```text
if type == bw_clear_base:
  colorResponse must be neutral_density or warm_neutral_density
  hide antiHalation
  hide warmCore

if type in vision3_ahu, cinestill_no_remjet, classic_dense_base:
  colorResponse must be red_orange_core, deep_red, or amber_core
  show antiHalation
  show warmCore
```

Do not permit in the GUI:

```text
type=bw_clear_base + colorResponse=amber_core
type=cinestill_no_remjet + colorResponse=neutral_density
arbitrary hue color picker
negative sliders
diffusion > 1 in normal mode
```

Implementation note: the current Python resolver is more permissive for some
B&W color-response inputs and may map them to a warm-neutral density fallback.
The GUI should not expose that fallback as an intentional user-facing mode.

### Resolved Parameter Debug Panel

Optional debug panel:

```text
Resolved Parameters
  model_family
  halation_type
  color_response
  profile
  amplify
  impact
  source_limiter_stops
  local_diffusion
  global_diffusion
  hue_green or density_tint
  background_gain
  background_luma_target
  no_remjet if applicable
  output_alpha_cap
```

Make this read-only in normal mode.

### Metrics Panel

Optional metrics display:

```text
alpha_max
alpha_mean
affected_percent
visible_affected_percent
output bounds
```

Warning thresholds:

```text
alpha_max near cap:
  label as "clipping at layer cap" rather than image clipping

visible_affected_percent > 30%:
  warn "large area affected; check Layer on White"
```

### Export Behavior

When user exports with layers enabled:

Recommended outputs:

```text
final.png
final.metrics.json
final_layers/physical_halation.png
```

For better GUI review, add optional future exports:

```text
final_layers/halation_on_black.png
final_layers/halation_on_white.png
```

The evaluator already writes black/white layer views; integrated renderer only
writes the default layer preview at the moment.

### Error Handling

Renderer errors to map to UI:

| Error | User-Facing Handling |
|-------|----------------------|
| unsupported `halation_type` | Reset type to default and show non-blocking warning. |
| invalid `color_response` for family | Auto-correct to valid response. |
| missing input file | Show file error before render. |
| renderer exception | Keep last preview and show render failed message. |
| metrics missing | Show output but hide metrics panel. |

### Recommended Defaults For First GUI Version

Global default:

```text
Enable Halation: off
Physical Lock: on
Type: CineStill No-Remjet
Color Response: Red / Orange Core
Amount: 1.0
Impact: 0.85
Anti-Halation Loss: 0.75
Source Selectivity: 0.45
Diffusion: 0.55
Warm Core: 0.45
Background Visibility: 0.75
Preview: Combined
```

If the selected film style is:

```text
vision3_500t
```

Use:

```text
Type: Vision3 AHU
Amount: 0.88
Impact: 0.82
Anti-Halation Loss: 0.22
Source Selectivity: 0.72
Diffusion: 0.40
Warm Core: 0.30
Background Visibility: 0.68
```

If the selected look is CineStill-like:

```text
Type: CineStill No-Remjet
Color Response: Amber Core
Amount: 1.20
Impact: 0.88
Anti-Halation Loss: 0.84
Source Selectivity: 0.46
Diffusion: 0.56
Warm Core: 0.58
Background Visibility: 0.78
```

If the selected look is B&W:

```text
Type: B&W Clear Base
Color Response: Neutral Density
Amount: 1.18
Impact: 0.84
Source Selectivity: 0.50
Diffusion: 0.66
Background Visibility: 0.76
hide Warm Core
hide Anti-Halation Loss
```

### Manual QA Checklist For GUI

Before shipping GUI halation controls:

1. Toggle halation on/off and confirm image returns exactly to no-halation
   render when off.
2. Switch `Vision3 AHU` to `CineStill No-Remjet`; verify stronger red/orange
   halation.
3. Switch `CineStill No-Remjet` color response from `Deep Red` to `Amber Core`;
   verify color changes while geometry remains stable.
4. Switch to `B&W Clear Base`; verify controls hide `Warm Core` and
   `Anti-Halation Loss`.
5. Verify `B&W Clear Base` produces neutral/warm density layer, not red halo.
6. Drag `Amount`; verify halo gets stronger without a radius slider changing.
7. Drag `Diffusion`; verify radius/softness changes visibly.
8. Drag `Source Selectivity`; verify weaker highlights drop out first.
9. Drag `Background Visibility`; verify dark-side visibility changes more than
   bright background visibility.
10. Check `Layer on Black` and `Layer on White` previews for every type.
11. Export with metrics and confirm `halation_metadata` matches UI choices.
12. Confirm invalid combinations cannot be selected.

### Implementation Checklist For GUI Developer

1. Add halation state object.
2. Add Enable and Physical Lock toggles.
3. Add type segmented control/dropdown.
4. Add color-response segmented control/dropdown.
5. Add validity filtering for color response.
6. Add sliders with clamping.
7. Hide/show controls based on family.
8. Map state to `PhysicalHalationControls` or CLI flags.
9. Add preview mode switcher.
10. Add metrics/resolved debug panel if useful.
11. Add export with `--write-metrics`.
12. Add smoke tests for at least:
    - Vision3 AHU red/orange;
    - CineStill no-remjet amber;
    - B&W clear-base neutral.

### Minimal GUI API Contract

The GUI can treat this as the stable contract:

```python
def render_halation_preview(base_rgb, state):
    if not state["enabled"]:
        return base_rgb

    controls = PhysicalHalationControls(
        model_family=state.get("modelFamily", "auto"),
        halation_type=state["type"],
        color_response=state["colorResponse"],
        amount=state["amount"],
        impact=state["impact"],
        anti_halation=state.get("antiHalation", 0.75),
        source_selectivity=state["sourceSelectivity"],
        diffusion=state["diffusion"],
        warm_core=state.get("warmCore", 0.45),
        background_visibility=state["backgroundVisibility"],
    )

    layer = build_physical_halation_layer(base_rgb, controls)
    return composite_layers(base_rgb, [layer], output_margin=4)
```

For preview modes:

```python
def preview_layer(layer, mode):
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    if mode == "layer_black":
        return layer.rgb * alpha
    if mode == "layer_white":
        return 1.0 * (1.0 - alpha) + layer.rgb * alpha
    raise ValueError(mode)
```

### Final GUI Guidance

The first GUI version should optimize for trustworthy defaults, not maximum
knobs.

Expose:

```text
Type
Color Response
Amount
Impact
Source Selectivity
Diffusion
Background Visibility
Warm Core
Anti-Halation Loss
```

Hide:

```text
source_softness
source_gamma
source_limiter_stops
local_diffusion
global_diffusion
hue_green
background_gain
background_luma_target
no_remjet
output_alpha_cap
source_normalization
```

The UI should help the user pick a physically meaningful mode, then tune inside
that mode. That is the difference between this system and a decorative glow
filter.
