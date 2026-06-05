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
   - contact sheets must be `original | halation layer on black | halation layer on white | combined`,
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
original | halation layer on black | halation layer on white | combined
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

## Deep Reflection: Remaining Autonomous Work

After re-reading the prompt and checking current online references, the V2 commit
is not the maximum independent implementation. It is a useful first renderer, but
several prompt requirements can still be implemented without user intervention.

Online corrections checked on 2026-06-04:

| Source | Correction / Reinforcement | Implementation Impact |
|--------|----------------------------|-----------------------|
| Kodak VISION3 500T 5219/7219 technical information | current 5219/7219 material says an anti-halation undercoat replaces the traditional remjet backing layer | describe `vision3_500t` as restrained anti-halation-present, not literally old remjet-present |
| CineStill help / film notes | CineStill films do not have remjet backing; remjet protects against halation of highlights | keep no-remjet-like profile as stronger red/orange halation |
| Dehancer desktop manual | `Amplify` affects emulsion sensitivity to scattered light, while `Impact` controls overall effect; `Local Diffusion` controls geometric radius | keep `amplify` separate from `impact`; do not let amplify directly change radius |

Critical prompt check:

> The scattering kernel should be mostly fixed. Strength slider A should mainly
> change scattered exposure, not directly change blur radius. Visible radius
> grows because stronger tails exceed a visibility threshold.

Current V2 mostly follows this because `amplify` multiplies exposure after fixed
red/green convolution kernels. However, it is still incomplete:

1. Source map uses per-image percentile normalization.
   - Good for preview.
   - Not physically absolute.
   - Fix: add `source_normalization=none|percentile`.
2. Source exposure is reconstructed from display RGB.
   - Better than sRGB thresholding.
   - Not true scene exposure.
   - Fix: allow an optional `source_linear_rgb` array for HDR/synthetic/RAW-like
     tests.
3. No quantitative radius test exists.
   - Fix: create synthetic point-light exposure sweeps and measure visible radius
     against source exposure.
4. No halation-specific metrics exist.
   - Fix: implement a local HalationEvalSuite with:
     - `visible_radius_vs_exposure`,
     - `radial_falloff`,
     - `hue_radius_curve`,
     - `dark_side_ratio`,
     - `background_suppression`,
     - `blue_leakage`,
     - `bounds`.
5. Contact sheets show black-background layer only.
   - User now asks for black and white layer views.
   - Fix: contact sheets become:
     `original | layer on black | layer on white | combined`.
6. Edge-side-aware behavior is only approximate.
   - Current implementation convolves symmetrically, then applies visibility.
   - Fix now: evaluate dark-side/bright-side ratio on synthetic diagnostic
     scenes; deeper directional convolution is future work if metrics fail.
7. Real film patch calibration is not done.
   - Fully robust calibration needs curated real film image licensing and patch
     extraction.
   - Autonomous now: add script structure and synthetic metrics; defer real patch
     fitting until a source list/license boundary is chosen.

## V2.1 Autonomous Task Plan

| Order | Task | Status | Completion Test |
|:---:|------|:---:|-----------------|
| 6 | Add absolute/relative source normalization | done | evaluator can run `source_normalization=none` for diagnostics |
| 7 | Allow synthetic HDR source exposure | done | point-light test can pass `source_linear_rgb` > 1.0 |
| 8 | Add HalationEvalSuite | done | writes `halation_eval.json` with radius, hue, background, blue leakage metrics |
| 9 | Add black+white layer contact sheets | done | sheets use 4 columns |
| 10 | Export V2.1 parameter sweeps | done | outputs under `outputs/eval/halation_v2p1/` |
| 11 | Update final ranking | done | tracker summarizes best parameters and remaining manual review |

## Execution Results: V2.1

Implemented:

- `source_normalization="none"` for absolute/HDR diagnostics.
- `source_linear_rgb` for synthetic RAW/HDR-like source exposure.
- Source-suppressed background estimation so bright source cores do not suppress
  their own halation.
- `background_luma_target`, defaulted to `0.20` after synthetic suppression
  testing.
- Four-column contact sheets:

```text
original | halation on black | halation on white | combined
```

- `scripts/evaluate_halation_physics_suite.py` for synthetic halation metrics.

V2.1 visual sweep:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_physical_halation_v2.py `
  --limit 20 `
  --max-side 768 `
  --include-diagnostics `
  --output-root outputs\eval\halation_v2p1
```

V2.1 outputs:

```text
outputs/eval/halation_v2p1/vision3_restrained/contact_sheet.png
outputs/eval/halation_v2p1/vision3_standard/contact_sheet.png
outputs/eval/halation_v2p1/cinestill_no_remjet/contact_sheet.png
outputs/eval/halation_v2p1/cinestill_aggressive/contact_sheet.png
outputs/eval/halation_v2p1/impact_low/contact_sheet.png
outputs/eval/halation_v2p1/impact_high/contact_sheet.png
outputs/eval/halation_v2p1/summary.json
```

V2.1 visual sweep metrics:

| Run | Alpha Max | Alpha Mean | Visible Mean | Bounds |
|-----|----------:|-----------:|-------------:|--------|
| `vision3_restrained` | 0.0274 | 0.00019 | 0.01% | 4..251 |
| `vision3_standard` | 0.0634 | 0.00050 | 0.51% | 4..251 |
| `cinestill_no_remjet` | 0.3200 | 0.00459 | 11.03% | 4..251 |
| `cinestill_aggressive` | 0.3200 | 0.01051 | 24.90% | 4..251 |
| `impact_low` | 0.2158 | 0.00371 | 9.00% | 4..251 |
| `impact_high` | 0.3200 | 0.00824 | 19.70% | 4..251 |

Physics suite:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_halation_physics_suite.py `
  --output-root outputs\eval\halation_v2p1_physics `
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

Physics outputs:

```text
outputs/eval/halation_v2p1_physics/halation_eval.json
outputs/eval/halation_v2p1_physics/exposure_radius_contact_sheet.png
```

Physics metrics:

- Visible radius by exposure stop:
  - stop 1: `0.0 px`
  - stop 2: `9.85 px`
  - stop 3: `18.68 px`
  - stop 4: `27.23 px`
  - stop 5: `33.30 px`
  - stop 6: `38.90 px`
- Radius monotonic: `true`
- Radius gains: `9.85, 8.83, 8.55, 6.06, 5.60 px`
- Radius gain decelerates after threshold: `true`
- Dark/bright visible-radius ratio: `2.26x`
- Dark/bright alpha-sum ratio: `1.72x`
- Blue leakage max: `0.0`
- Center green/red ratio >= outer green/red ratio: `6/6`

Interpretation:

- The core prompt requirement is now quantitatively checked: fixed kernels plus
  stronger exposure make the visible radius grow monotonically and with
  decelerating increments.
- `amplify` still does not directly change kernel radius.
- `impact` remains a display mix control.
- Bright background suppression is now measured, not only eyeballed.
- Remaining non-autonomous work is real-film patch calibration against a curated
  and license-safe image set.

## V2.2 Physical-Lock Control Surface

User feedback on 2026-06-05:

> Avoid exposing all low-level parameters in a way that lets users break physical
> rigor. Create packaged sliders, or a "physical rigor" checkbox that links
> controls like Photoshop's aspect-ratio lock.

Updated online reference check on 2026-06-05:

| Source | Constraint Used | Locked-Control Consequence |
|--------|-----------------|----------------------------|
| Kodak motion-picture glossary / essential reference | anti-halation backing absorbs light that would otherwise reflect back into the emulsion | model halation as secondary backscatter exposure; anti-halation controls coupling, not blur geometry |
| Kodak VISION3 500T 5219/7219 technical information | current 5219/7219 has anti-halation undercoat, so stock halation should be restrained | `vision3_500t` profile has lower backscatter and tighter alpha cap |
| CineStill film notes/help | 800T is no-remjet-like and has stronger red/orange highlight halation | `cinestill_800t` profile has higher backscatter coupling and broader diffusion range |
| Dehancer desktop manual/article | Source Limiter, Background Gain, Local Diffusion, Global Diffusion, Amplify, and Impact are distinct controls; Local Diffusion controls radius; Amplify is glare energy, not simple opacity | expose locked sliders that map to the same physical roles without letting one slider alter unrelated terms |
| User prompt | strength slider A should mainly change scattered exposure, not directly change blur radius; visible radius grows because stronger tails cross visibility threshold | `amount` maps only to `amplify`; only `diffusion` changes local/global radius terms |

### Locked Slider Contract

Implemented in `src/filmfx/halation_controls.py`.

Default physical mode in `scripts/render_film.py` is now locked. Expert mode is
still available with `--halation-control-mode expert` or
`--halation-expert-controls`, but normal UI/CLI controls should bind to the
locked surface.

| User Slider | Physical Meaning | Allowed Low-Level Effects |
|-------------|------------------|---------------------------|
| `amount` | effective secondary scattered exposure | `amplify` only |
| `impact` | display/output mix after the layer is formed | `impact` only |
| `anti_halation` | anti-halation/remjet suppression vs no-remjet-like backscatter | `no_remjet` only |
| `source_selectivity` | how bright a source must be before it contributes | `source_limiter_stops` only |
| `diffusion` | film/emulsion scatter geometry | `local_diffusion` and `global_diffusion` only |
| `warm_core` | green-layer coupling in the hottest core | `hue_green` only |
| `background_visibility` | dark/high-contrast background gating | `background_gain` and `background_luma_target` only |

Locked invariants are now tested in `tests/test_halation_controls.py`:

- changing `amount` cannot change radius, source gating, background gating, hue,
  or no-remjet coupling,
- changing `impact` cannot change physical layer formation,
- changing `diffusion` only changes local/global diffusion geometry,
- changing `anti_halation` only changes backscatter coupling,
- source and background gates stay separate.

### V2.2 Locked Output Sweep

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_physical_halation_v2.py `
  --limit 20 `
  --max-side 768 `
  --include-diagnostics `
  --control-mode locked `
  --output-root outputs\eval\halation_v2p2_locked
```

Contact sheets:

```text
outputs/eval/halation_v2p2_locked/locked_vision3_clean/contact_sheet.png
outputs/eval/halation_v2p2_locked/locked_vision3_push/contact_sheet.png
outputs/eval/halation_v2p2_locked/locked_cinestill_balanced/contact_sheet.png
outputs/eval/halation_v2p2_locked/locked_cinestill_strong/contact_sheet.png
outputs/eval/halation_v2p2_locked/locked_amount_low/contact_sheet.png
outputs/eval/halation_v2p2_locked/locked_amount_high/contact_sheet.png
outputs/eval/halation_v2p2_locked/summary.json
```

Each contact sheet keeps the requested format:

```text
original | halation on black | halation on white | combined
```

Locked sweep metrics:

| Run | Alpha Max | Alpha Mean | Visible Mean | Bounds |
|-----|----------:|-----------:|-------------:|--------|
| `locked_vision3_clean` | 0.0291 | 0.00016 | 0.01% | 4..251 |
| `locked_vision3_push` | 0.0627 | 0.00041 | 0.45% | 4..251 |
| `locked_cinestill_balanced` | 0.3200 | 0.00524 | 12.42% | 4..251 |
| `locked_cinestill_strong` | 0.3200 | 0.00978 | 22.26% | 4..251 |
| `locked_amount_low` | 0.2582 | 0.00333 | 7.91% | 4..251 |
| `locked_amount_high` | 0.3200 | 0.00815 | 18.60% | 4..251 |

The `locked_amount_low` and `locked_amount_high` runs share the same geometric
diffusion values. Their visible coverage difference is therefore caused by
stronger scattered exposure crossing the visibility threshold, not by changing
kernel radius.

### V2.2 Physics Check

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_halation_physics_suite.py `
  --output-root outputs\eval\halation_v2p2_locked_physics `
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

Outputs:

```text
outputs/eval/halation_v2p2_locked_physics/halation_eval.json
outputs/eval/halation_v2p2_locked_physics/exposure_radius_contact_sheet.png
```

Metrics:

- Radius monotonic: `true`
- Radius gains: `9.85, 8.83, 8.55, 6.06, 5.60 px`
- Dark/bright visible-radius ratio: `2.26x`
- Dark/bright alpha-sum ratio: `1.72x`
- Blue leakage max: `0.0`

### Rollback Note

V2.2 is intentionally isolated on `research/physical-halation-v2p2-calibration`.
The previous V2.1 baseline is commit `2001e1c` / branch
`research/physical-halation-v2`. If visual review rejects V2.2 locked controls,
revert the V2.2 commit or switch back to the V2.1 branch.

## V2.3 Physical Rule Families

User feedback on 2026-06-05:

> Discrete mode choices should not only select discrete parameters; their larger
> meaning is that they represent different physical rules.

Reflection after reading `halationguide.md` and current online sources:

- A mode such as `cinestill_no_remjet` is not merely "more amount"; it changes
  the backscatter path and anti-halation suppression.
- A mode such as `bw_clear_base` should not reuse the color-negative
  red/green-layer equation with saturation removed; it needs a separate
  density-domain rule family.
- Color response choices should be bounded laws: red outer / orange core, deep
  red, amber core, or neutral density. They should not open arbitrary blue,
  cyan, or purple halation, which is more like bloom, lens flare, or optical
  scatter.

Online reference check on 2026-06-05:

| Source | Constraint Used |
|--------|-----------------|
| Kodak motion-picture glossary | halation is caused by scattering/reflection through emulsion/base surfaces; anti-halation backing absorbs light that would reflect back into the emulsion |
| Dehancer halation docs/manual | Halation Profiles are separated from Source Limiter, Background Gain, Local/Global Diffusion, Amplify, Hue, Blue Comp., and Impact; profiles also separate standard emulsion and No Remjet |
| CineStill remjet help | remjet protects against highlight halation; CineStill films do not have remjet backing |
| Black-and-white film references | some B&W/reversal/clear-base stocks have their own halation/glow behavior, so neutral density glow is a separate rule family from color-negative red-layer backscatter |

### Rule-Family Design

Implemented:

- `src/filmfx/effects.py`
  - `physical_halation_layer`: color-negative backscatter family.
  - `density_halation_layer`: black-and-white / density-domain family.
- `src/filmfx/halation_controls.py`
  - `model_family`
  - `halation_type`
  - `color_response`
  - locked sliders
  - `build_physical_halation_layer`

Discrete control levels:

| Level | Field | Current Values | Meaning |
|-------|-------|----------------|---------|
| Physical rule family | `model_family` | `color_negative_backscatter`, `bw_density_halation` | Selects the equation family |
| Backing / stock structure | `halation_type` | `vision3_ahu`, `cinestill_no_remjet`, `classic_dense_base`, `bw_clear_base` | Selects anti-halation/backing and allowed parameter ranges |
| Color response law | `color_response` | `red_orange_core`, `deep_red`, `amber_core`, `neutral_density`, `warm_neutral_density` | Selects bounded hue/density behavior |
| Locked sliders | `amount`, `impact`, `anti_halation`, `source_selectivity`, `diffusion`, `warm_core`, `background_visibility` | Continuous control within the selected law | Fine adjustment without breaking the rule family |

Rule differences:

```text
color_negative_backscatter:
  H_R = A * beta_R * V * conv(S, K_R)
  H_G = A * beta_G * V * conv(S_high, K_G)
  H_B = 0

bw_density_halation:
  H_density = A * V * conv(S, K_density)
  no red/green hue-radius law
  no no_remjet/hue_green/profile controls in resolved parameter surface
```

### V2.3 Locked Family Output Sweep

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_physical_halation_v2.py `
  --limit 20 `
  --max-side 768 `
  --include-diagnostics `
  --control-mode family `
  --output-root outputs\eval\halation_v2p3_families
```

Contact sheets:

```text
outputs/eval/halation_v2p3_families/family_vision3_ahu_red_orange/contact_sheet.png
outputs/eval/halation_v2p3_families/family_cinestill_no_remjet_deep_red/contact_sheet.png
outputs/eval/halation_v2p3_families/family_cinestill_no_remjet_amber_core/contact_sheet.png
outputs/eval/halation_v2p3_families/family_classic_dense_base_soft_red/contact_sheet.png
outputs/eval/halation_v2p3_families/family_bw_clear_base_neutral_density/contact_sheet.png
outputs/eval/halation_v2p3_families/family_bw_clear_base_warm_density/contact_sheet.png
outputs/eval/halation_v2p3_families/summary.json
```

Each sheet keeps:

```text
original | halation on black | halation on white | combined
```

V2.3 family metrics:

| Run | Family | Type | Color Response | Alpha Max | Alpha Mean | Visible Mean | Bounds |
|-----|--------|------|----------------|----------:|-----------:|-------------:|--------|
| `family_vision3_ahu_red_orange` | `color_negative_backscatter` | `vision3_ahu` | `red_orange_core` | 0.0478 | 0.00026 | 0.04% | 4..251 |
| `family_cinestill_no_remjet_deep_red` | `color_negative_backscatter` | `cinestill_no_remjet` | `deep_red` | 0.3200 | 0.00708 | 16.81% | 4..251 |
| `family_cinestill_no_remjet_amber_core` | `color_negative_backscatter` | `cinestill_no_remjet` | `amber_core` | 0.3200 | 0.00708 | 16.81% | 4..251 |
| `family_classic_dense_base_soft_red` | `color_negative_backscatter` | `classic_dense_base` | `red_orange_core` | 0.1418 | 0.00211 | 4.94% | 4..251 |
| `family_bw_clear_base_neutral_density` | `bw_density_halation` | `bw_clear_base` | `neutral_density` | 0.2132 | 0.00521 | 13.87% | 4..251 |
| `family_bw_clear_base_warm_density` | `bw_density_halation` | `bw_clear_base` | `warm_neutral_density` | 0.2132 | 0.00521 | 13.87% | 4..251 |

The two CineStill color-response runs intentionally share alpha/visible metrics:
their geometry and scattered exposure are held fixed; only the red/green
response law changes. Likewise, the two B&W density runs share alpha/visible
metrics and differ only in neutral vs warm density tint.

### V2.3 Verification

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_color_baseline_safety.py tests\test_halation_controls.py
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

Results:

- Pytest: `8 passed`
- Radius monotonic: `true`
- Radius gains: `9.85, 8.83, 8.55, 6.06, 5.60 px`
- Dark/bright visible-radius ratio: `2.26x`
- Blue leakage max: `0.0`

Integrated renderer smoke outputs:

```text
outputs/integration/render_film_halation_v2p3_amber_core_smoke.png
outputs/integration/render_film_halation_v2p3_bw_density_smoke.png
```
