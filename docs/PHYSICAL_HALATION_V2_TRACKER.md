# Physical Halation V2 Tracker

> Created: 2026-06-04
>
> Branch: `research/physical-halation-v2`
>
> Goal: replace the current simplified red screen glow with an inspectable,
> parameterized halation layer renderer guided by film physics and the user's
> supplied prompt.

## Prompt-Derived Requirements

The supplied prompt argues that halation should be:

1. **Physical-prior first**: no whole-image AI repainting.
2. **Layer-based**: output an independent halation layer and then a bounded
   composite.
3. **Exposure-driven**: source should be based on a scene-linear/log-exposure
   estimate, not only raw sRGB whiteness.
4. **Two-control strength model**:
   - `amplify`: physical scattered-light exposure coupling,
   - `impact`: final display mix/opacity.
5. **Slow visible-radius growth**: brighter sources mainly lift a mostly fixed
   kernel above the visibility threshold; radius growth should be log-like, not
   `radius = k * brightness`.
6. **Red/green layer separation**:
   - red layer is the dominant long-tail halo,
   - green layer appears near extreme highlights and creates orange/yellow cores,
   - blue leakage should stay near zero.
7. **Background-aware visibility**:
   - dark/high-contrast backgrounds show halation,
   - bright backgrounds suppress it,
   - skin-like areas should be protected.
8. **Profile presets**:
   - `vision3_500t`: remjet present, restrained,
   - `cinestill_800t`: no-remjet-like, stronger red/orange halation.
9. **Review contract**:
   - contact sheets must be `original | halation layer on black | combined`,
   - multiple parameter combinations must be exported.

## Online Research Notes

Checked on 2026-06-04:

| Source | Relevant Point | Design Decision |
|--------|----------------|-----------------|
| Kodak glossary of motion-picture terms | anti-halation backing absorbs light that would otherwise reflect back into the emulsion; remjet is an anti-halation backing | model halation as secondary exposure caused by backscatter, not generic bloom |
| Kodak Vision3 500T 5219/7219 technical data | 5219/7219 has acetate safety base with rem-jet backing; datasheet includes sensitometric and spectral dye-density curves | `vision3_500t` preset should be restrained; keep the renderer in an approximate linear/exposure domain |
| Dehancer halation docs/manual | separates Source Limiter, Amplify, Impact, Local Diffusion, Global Diffusion; Amplify is not simple opacity | expose matching controls and keep `amplify` separate from `impact` |

## V2 Algorithm

Input is the current rendered RGB in `[0,1]`. If a future pipeline supplies true
linear/HDR data, this renderer can consume it directly. For now, it reconstructs
an approximate linear working image from display RGB:

```text
linear_rgb = srgb_to_linear(base_rgb)
Y = luminance(linear_rgb)
logE = log2(Y / middle_gray + eps)
```

### Source Map

```text
S = softplus((logE - source_limiter_stops) / source_softness)^source_gamma
S *= specular_confidence
S *= edge_confidence
```

Rationale:

- high luma alone is not enough,
- saturated/white high-intensity highlights get higher confidence,
- smooth thresholds avoid hard bands.

### Background Visibility

```text
local_mean = gaussian(Y, background_radius)
V_dark = sigmoid((background_luma_target - local_mean) * background_gain)
V_contrast = normalized local contrast
V_skin = 1 - skin_mask * skin_protect
V = V_dark * V_contrast * V_skin
```

Rationale:

- halation is more visible on dark/high-contrast sides,
- bright skies/walls should not get broad red haze,
- skin-adjacent high values should not become red/orange contamination.

### Kernels

Red layer:

```text
K_R = gaussian(sigma1) + gaussian(sigma2) + exponential-like tail + low-frequency glare
```

Green layer:

```text
K_G = smaller gaussian kernels, triggered from a higher source threshold
```

Blue layer:

```text
K_B = 0
```

### Strength Controls

```text
H_R = amplify * red_backscatter * no_remjet * V * conv(S, K_R)
H_G = amplify * green_backscatter * hue_green * no_remjet * V * conv(S_high, K_G)
layer_alpha = impact * exposure_to_alpha(H_R + H_G)
```

`amplify` changes physical exposure coupling and therefore can make the visible
radius grow slowly. `impact` changes final display mix while preserving the
source/radius relationship.

## Parameter Sweep

Initial sweeps:

| Run | Profile | Amplify | Impact | Source Limiter | Local Diffusion | Global Diffusion | Hue Green | Background Gain |
|-----|---------|--------:|-------:|----------------:|----------------:|-----------------:|----------:|----------------:|
| restrained | vision3_500t | 0.55 | 0.70 | 2.8 | 0.85 | 0.08 | 0.16 | 1.00 |
| standard | vision3_500t | 0.85 | 0.85 | 2.4 | 1.00 | 0.12 | 0.22 | 1.15 |
| no_remjet | cinestill_800t | 1.10 | 0.85 | 1.9 | 1.25 | 0.22 | 0.32 | 1.45 |
| aggressive | cinestill_800t | 1.45 | 1.00 | 1.6 | 1.55 | 0.34 | 0.42 | 1.70 |
| impact_low | cinestill_800t | 1.35 | 0.45 | 1.8 | 1.45 | 0.28 | 0.36 | 1.55 |
| impact_high | cinestill_800t | 1.35 | 1.00 | 1.8 | 1.45 | 0.28 | 0.36 | 1.55 |

## Output Contract

Each run writes:

```text
outputs/eval/halation_v2/<run_name>/after/*.png
outputs/eval/halation_v2/<run_name>/layers/*.png
outputs/eval/halation_v2/<run_name>/contact_sheet.png
outputs/eval/halation_v2/<run_name>/metrics.json
```

The contact sheet layout is:

```text
original | halation layer on black | combined
```

## Current Status

| Order | Task | Status |
|:---:|------|:---:|
| 1 | Research and tracker | done |
| 2 | V2 layer implementation | done |
| 3 | Contact sheet evaluator | done |
| 4 | Multi-parameter exports | done |
| 5 | Result summary | done |

## Execution Results: 2026-06-04

Implemented:

- `physical_halation_layer` in `src/filmfx/effects.py`
- package export in `src/filmfx/__init__.py`
- physical mode in `scripts/render_film.py`
- sweep/contact-sheet evaluator in `scripts/evaluate_physical_halation_v2.py`

Full sweep command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_physical_halation_v2.py `
  --limit 20 `
  --max-side 768 `
  --include-diagnostics `
  --output-root outputs\eval\halation_v2
```

Generated outputs:

```text
outputs/eval/halation_v2/vision3_restrained/contact_sheet.png
outputs/eval/halation_v2/vision3_standard/contact_sheet.png
outputs/eval/halation_v2/cinestill_no_remjet/contact_sheet.png
outputs/eval/halation_v2/cinestill_aggressive/contact_sheet.png
outputs/eval/halation_v2/impact_low/contact_sheet.png
outputs/eval/halation_v2/impact_high/contact_sheet.png
outputs/eval/halation_v2/summary.json
```

Each run contains:

```text
original/*.png
layers/*.png
combined/*.png
metrics.json
contact_sheet.png
```

Main metrics:

| Run | Profile | Amplify | Impact | Alpha Max | Visible Mean | Bounds |
|-----|---------|--------:|-------:|----------:|-------------:|--------|
| `vision3_restrained` | `vision3_500t` | 0.55 | 0.70 | 0.0327 | 0.01% | 4..251 |
| `vision3_standard` | `vision3_500t` | 0.85 | 0.85 | 0.0736 | 0.44% | 4..251 |
| `cinestill_no_remjet` | `cinestill_800t` | 1.10 | 0.85 | 0.3200 | 12.68% | 4..251 |
| `cinestill_aggressive` | `cinestill_800t` | 1.45 | 1.00 | 0.3200 | 29.59% | 4..251 |
| `impact_low` | `cinestill_800t` | 1.35 | 0.45 | 0.2310 | 10.79% | 4..251 |
| `impact_high` | `cinestill_800t` | 1.35 | 1.00 | 0.3200 | 23.17% | 4..251 |

Integrated renderer smoke:

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py $input `
  --style vision3_500t `
  --halation 1.1 `
  --halation-model physical `
  --halation-profile cinestill_800t `
  --halation-impact 0.85 `
  --halation-source-limiter 1.9 `
  --halation-local-diffusion 1.25 `
  --halation-global-diffusion 0.22 `
  --halation-hue-green 0.32 `
  --halation-background-gain 1.45 `
  --output outputs\integration\render_film_physical_halation_smoke.png `
  --write-layers `
  --write-metrics
```

Smoke result:

```text
outputs/integration/render_film_physical_halation_smoke.png
outputs/integration/render_film_physical_halation_smoke.metrics.json
bounds=[4, 251]
layer=physical_halation
alpha_max=0.1201
```

Visual read:

- `vision3_restrained` is very subtle, matching remjet-present Vision3 behavior.
- `vision3_standard` is still restrained but visible on the strongest sources.
- `cinestill_no_remjet` is the best first visual review target.
- `cinestill_aggressive` is intentionally strong and can affect too much of some
  natural images.
- `impact_low` and `impact_high` confirm that Impact mostly changes display mix,
  while the source/radius structure remains similar.

Known limitations:

- Current input is approximate linear reconstructed from display RGB, not true
  RAW/HDR scene-linear exposure.
- No real-film patch calibration has been performed yet.
- The model can still trigger on bright flowers/color charts because there is no
  semantic light-source classifier.
- Blue leakage is intentionally zero in the layer model.
