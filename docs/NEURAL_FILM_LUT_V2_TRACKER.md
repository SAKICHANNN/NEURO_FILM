# Neural Film LUT V2 Tracker

> Created: 2026-06-03
>
> Branch: `research/neural-film-lut-v2`
>
> Purpose: restart the machine-learning LUT path with architectures that can be
> both image-specific and film-stock-specific, while avoiding severe color
> distortion, clipping, and the style-collapse failure seen in Neural LUT V1.

## Mission

Build and evaluate a new Neural LUT family that can produce visible
film-stock-specific style on a per-image basis.

Required properties:

1. **Photo-specific**: the transform must react to each input image's tone,
   color distribution, and content masks.
2. **Film-specific**: Portra, Ektar, Velvia, and Vision3 outputs must be
   visibly and measurably different on the same input.
3. **Content-preserving**: no geometry changes, no generative hallucination, no
   texture replacement.
4. **Safe output**: no new hard clipping; PNG outputs must stay inside `[4,251]`
   unless explicitly marked as a failure test.
5. **No severe color distortion**: neutral/skin contamination, hue inversions,
   posterization, and gamut compression artifacts must be bounded.
6. **Strength-controllable**: each viable scheme must export multiple strengths
   for visual review.

This tracker does not replace the current deterministic renderers. It opens a
new research line whose best output may later be combined with `safe-rich`,
`local_maps_tuned_s1p2`, or `film_response_v1`.

## Prior Failure: Neural LUT V1

V1 components:

- `src/models/color_lut/`
- `scripts/train_neural_lut.py`
- `scripts/evaluate_neural_lut.py`

Observed failure:

```text
train: 120 examples, 6 color stocks, 300 steps
initial_l1 = 0.013045
final_l1   = 0.005296
ratio      = 2.46x
eval promote_count = 0/6
```

Second attempt:

```text
steps = 800
num_basis = 12
basis_init_std = 0.01
```

Still failed.

Root cause:

- the softmax basis mixture collapsed to one basis LUT,
- all six film stocks produced effectively the same output,
- the model learned an average enhancement rather than stock identity,
- stronger parameter sweeps did not solve the structural collapse.

V2 rule:

> Do not continue V1 by only increasing `steps`, `num_basis`, or learning rate.
> Any V2 model must contain explicit anti-collapse design.

## Online Research Notes

Researched sources on 2026-06-03:

| Source | Relevant Idea | Use In This Project |
|--------|---------------|---------------------|
| Zeng et al., "Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time" | image-adaptive LUT weights predicted from a downsampled image; efficient high-res application | keep image encoder + LUT runtime, but avoid shared softmax collapse |
| Yang et al., "AdaInt: Learning Adaptive Intervals for 3D Lookup Tables" | non-uniform LUT sampling increases expressiveness in color regions needing nonlinear response | later V2.3 if uniform 17/33 LUT cannot express film response cleanly |
| Yang et al., "SepLUT" | cascade 1D LUTs for tone/channel curves with 3D LUT for cross-channel color | primary candidate for film: tone response and color mixing should be separated |
| Liu et al., "4D LUT" | RGB + context map enables content-dependent transforms for same RGB values | primary candidate for local film behavior: sky/foliage/skin/highlight can differ |
| Conde et al., "NILUT" | implicit continuous LUT conditioned by style vector; memory-efficient multi-style LUT | candidate for style-conditioned stock identity without storing many grids |
| HDRNet / deep bilateral learning | image-specific local affine color transforms using low-res prediction, high-res application | fallback/hybrid for local tone and color where LUT-only is too global |
| SA-LUT / spatial-adaptive 4D LUT | recent photorealistic style-transfer route combining style-guided 4D LUT and context map | informs context/style separation, but avoid reference-image dependency in first V2 |

Initial conclusion:

- A single shared softmax basis bank is the wrong default.
- Film stock identity should be represented by either independent heads,
  style-specific residuals, or a conditioned implicit function with explicit
  style-separation losses.
- Image specificity should come from low-dimensional global context plus
  optional spatial/context maps, not from unconstrained pixel-wise RGB residuals.
- Tone response and color cross-talk should be separated; this matches SepLUT
  and our visual feedback that saturation alone is not enough.

## Data Strategy

### Available Local Signals

| Source | Role |
|--------|------|
| rawpixls seed images | fixed visual/eval set |
| `safe-rich` outputs | conservative no-clip baseline |
| `local_maps_tuned_s1p2` | automatic safety winner; not visually filmic enough |
| `film_response_v1_s1p0/s1p4` | stronger stock style teacher; too hard for old safety gates |
| `configs/film_color_stats.json` | per-stock Lab distribution anchors |
| ignored local film-domain data | style statistics and possible unpaired losses |

### Pseudo-Target Policy

V2 should not learn directly from a single teacher only. Use a blended target
curriculum:

1. `strength=0.0`: identity target.
2. `strength=0.35`: safe-rich target.
3. `strength=0.70`: softened film-response target.
4. `strength=1.00`: film-response style target.
5. optional: local-map safety target mixed into neutral/skin regions.

Rationale:

- identity anchoring gives a stable strength control,
- safe-rich anchoring prevents clipping/gamut blowups,
- film-response target adds stock-specific style,
- mixed targets reduce "one average LUT" collapse.

## Hard Safety Gates

Every scheme must export `summary.json` and contact sheets. A run is invalid if
any hard gate fails:

| Gate | Required |
|------|----------|
| New clipping | `0` new clipped pixels |
| Output bounds | all PNG outputs inside `[4, 251]` |
| NaN/Inf | none in LUTs or outputs |
| Geometry | exact input width/height |
| Determinism | same seed/checkpoint produces byte-stable PNGs or documented deterministic tolerance |
| Strength zero | `strength=0` is identity or within 1 LSB |
| Runtime | full-res eval does not OOM on RTX 5070 Ti 12GB |

## Soft Quality Gates

These gates decide whether a safe run is worth visual review:

| Gate | Target |
|------|--------|
| L-SSIM | conservative mode `>=0.995`; style mode `>=0.970` unless visually justified |
| HF delta | conservative mode within 5%; style mode reported but not fatal |
| Neutral contamination | `<= safe-rich + 0.10%` for low-neutral stocks; special-case Velvia with visual check |
| Skin contamination | no obvious orange/magenta/green skin drift on contact sheets |
| Chroma | not merely uniform saturation; must include hue/tone/style separability |
| Banding | no visible posterization; LUT smoothness metrics not worse than V1 |
| Stock separability | same input across six stocks must be visually and metrically distinct |

## Style Success Metrics

Old promote gates rewarded "not changing much." V2 needs style-specific metrics.

Compute these in addition to existing safety metrics:

1. **Same-image stock separation**
   - render each image with all six stocks,
   - compute pairwise Lab delta between outputs,
   - require nonzero separation after neutral/skin masking,
   - fail if all stocks cluster like V1.

2. **Film-stat proximity**
   - compare output Lab mean/std/quantiles to `film_color_stats.json`,
   - report per-stock distance to target stats,
   - do not optimize this alone because it can cause color distortion.

3. **Tone-response signature**
   - compare source/output luminance curves,
   - report toe/shoulder/midtone contrast changes,
   - require style-specific signatures for Ektar, Portra, Velvia, Vision3.

4. **Hue-sector response**
   - track sky/foliage/warm/skin/neutral sectors separately,
   - ensure "style" is not just global chroma gain.

5. **Strength monotonicity**
   - increasing strength should smoothly increase style distance,
   - no sudden clipping, hue flips, or banding at high strengths.

## Candidate Schemes

### Scheme A: Style-Separated SepLUT

Priority: P0

Design:

```text
image encoder + style id + strength
  -> per-stock 3x1D tone/channel LUT head
  -> per-stock 3D residual LUT head
  -> monotonic 1D curves
  -> bounded 3D residual
  -> gamut-safe output margin
```

Why it may work:

- separates tone response from cross-channel film color,
- prevents softmax collapse by using per-style heads or per-style residuals,
- allows film stock identity in the tone curve, not just saturation.

Anti-collapse requirements:

- no shared final softmax over all stocks,
- per-style residual parameters or low-rank style adapters,
- style contrastive loss on same-image multi-stock outputs,
- identity strength anchor,
- smoothness and monotonicity penalties.

Expected outputs:

```text
outputs/eval/neural_film_lut_v2/scheme_a_seplut_s0p5/
outputs/eval/neural_film_lut_v2/scheme_a_seplut_s1p0/
outputs/eval/neural_film_lut_v2/scheme_a_seplut_s1p5/
outputs/eval/neural_film_lut_v2/scheme_a_seplut_s2p0/
```

Stop conditions:

- any strength clips,
- all styles still cluster,
- 1D curves become non-monotonic after regularization,
- contact sheets still read as saturation-only.

### Scheme B: Conditional NILUT Distilled To Grid

Priority: P1

Design:

```text
MLP(rgb, style_embedding, image_context, strength) -> rgb'
sample MLP on a 33^3 grid per image/style
apply sampled LUT with existing trilinear operator
```

Why it may work:

- continuous color transform can express smooth style-conditioned mappings,
- style embedding can be supervised with explicit style separation,
- sampled output still becomes a deterministic LUT-like renderer.

Safety constraints:

- MLP output bounded with residual scale,
- grid smoothness penalty,
- identity strength anchor,
- sampled grid must be checked for out-of-range values before rendering.

Expected outputs:

```text
outputs/eval/neural_film_lut_v2/scheme_b_nilut_s0p5/
outputs/eval/neural_film_lut_v2/scheme_b_nilut_s1p0/
outputs/eval/neural_film_lut_v2/scheme_b_nilut_s1p5/
outputs/eval/neural_film_lut_v2/scheme_b_nilut_s2p0/
```

Stop conditions:

- MLP overfits teacher but produces banded sampled LUTs,
- strength interpolation is unstable,
- style embedding collapses or causes hue inversions.

### Scheme C: Context-Aware 4D LUT

Priority: P1

Design:

```text
image encoder -> global style coefficients
context encoder -> low-res context map
RGB + context -> RGB via 4D LUT
```

V2.0 simplification:

- start with deterministic context maps:
  - neutral,
  - skin,
  - sky/cyan,
  - foliage,
  - warm/highlight,
  - shadow.
- only learn LUT/color response first.
- learn context map later if deterministic context helps.

Why it may work:

- same RGB triplet can map differently in sky vs skin vs foliage,
- directly targets user complaint that global/local chroma maps lack real stock
  behavior,
- can preserve protected regions without making the whole transform weak.

Expected outputs:

```text
outputs/eval/neural_film_lut_v2/scheme_c_4dlut_s0p5/
outputs/eval/neural_film_lut_v2/scheme_c_4dlut_s1p0/
outputs/eval/neural_film_lut_v2/scheme_c_4dlut_s1p5/
outputs/eval/neural_film_lut_v2/scheme_c_4dlut_s2p0/
```

Stop conditions:

- context halos,
- skin/neutral contamination,
- context map becomes a hidden segmentation artifact generator,
- runtime or memory too high.

### Scheme D: Bilateral Grid / Local Affine Film LUT Hybrid

Priority: P2 fallback

Design:

```text
image encoder + style id
  -> low-res bilateral grid or tile grid of affine RGB/Lab transforms
  -> guided upsample / interpolation
  -> bounded output
```

Why it may work:

- strong photo-specific local tone/color behavior,
- proven route for real-time image enhancement,
- can complement a global LUT when 3D/4D LUT is too rigid.

Why it is fallback:

- less "LUT pure" than A/B/C,
- higher risk of local halos,
- must be carefully bounded to avoid content-like artifacts.

Expected outputs:

```text
outputs/eval/neural_film_lut_v2/scheme_d_bilateral_s0p5/
outputs/eval/neural_film_lut_v2/scheme_d_bilateral_s1p0/
outputs/eval/neural_film_lut_v2/scheme_d_bilateral_s1p5/
outputs/eval/neural_film_lut_v2/scheme_d_bilateral_s2p0/
```

## Training Loss Plan

Each trainable scheme should support a loss config with these terms:

| Loss | Purpose | Initial Weight |
|------|---------|---------------:|
| `target_l1_lab` | match pseudo-target Lab color/tone | 1.0 |
| `target_huber_rgb` | prevent harsh RGB outliers | 0.25 |
| `luma_ssim_loss` | preserve structure/luminance detail | 0.40 |
| `neutral_protect_loss` | keep near-neutrals stable | 0.60 |
| `skin_protect_loss` | keep skin-like colors stable | 0.40 |
| `clip_margin_loss` | penalize leaving `[4,251]` margin | 2.0 |
| `lut_smoothness_loss` | prevent banding/posterization | 0.25 |
| `monotonic_1d_loss` | keep tone/channel curves sane | 0.50 for Scheme A |
| `style_separation_loss` | prevent stock collapse | 0.30 |
| `strength_identity_loss` | force `strength=0` identity | 1.0 |
| `strength_monotonic_loss` | smooth strength control | 0.25 |

Do not add all losses blindly. Start with a minimal stable subset, then add
terms only when a failure mode appears.

## Evaluation Output Contract

Every scheme must write:

```text
outputs/eval/neural_film_lut_v2/<run_name>/summary.json
outputs/eval/neural_film_lut_v2/<run_name>/<stock>/metrics.json
outputs/eval/neural_film_lut_v2/<run_name>/<stock>/manifest.csv
outputs/eval/neural_film_lut_v2/<run_name>/<stock>/contact_sheet.png
outputs/eval/neural_film_lut_v2/<run_name>/<stock>/after/*.png
outputs/eval/neural_film_lut_v2/<run_name>/<stock>/diff_maps/*.png
```

Every viable scheme must export strengths:

```text
s0p5  subtle
s1p0  normal
s1p5  strong
s2p0  very strong
```

Contact sheets should show:

```text
before | safe-rich | teacher/reference | model output
```

If no teacher is used in that scheme, use:

```text
before | safe-rich | model s1p0 | model current strength
```

## Dependency Order

| Order | Task | Depends On | Status | Completion Test | Commit Node |
|:---:|------|------------|:---:|-----------------|-------------|
| 0 | Create branch and V2 tracker | clean previous branch | done | branch + tracker exist | `docs: add neural film lut v2 tracker` |
| 1 | Add style-separation metrics | Order 0 | done | `candidate_summary_v1.json/csv` reports style chroma range | included in V2 commit |
| 2 | Build pseudo-target generator | Order 1 | done | emits identity/safe/soft-film/full-film target set | included in V2 commit |
| 3 | Scheme A SepLUT scaffold | Order 2 | done | torch scaffold plus numpy distilled smoke/eval | included in V2 commit |
| 4 | Scheme A full six-stock run | Order 3 | done | exports s0p5/s1p0/s1p5/s2p0 plus subtle strengths | included in V2 commit |
| 5 | Scheme B NILUT scaffold | Order 2 | done | smoke fit/eval passed identity/bounds path | included in V2 commit |
| 6 | Scheme B full six-stock run | Order 5 | done | exports s0p5/s1p0/s1p5/s2p0 and compare metrics | included in V2 commit |
| 7 | Scheme C 4D LUT scaffold | Order 2 | done | deterministic context smoke passed bounds path | included in V2 commit |
| 8 | Scheme C full six-stock run | Order 7 | done | exports s0p5/s1p0/s1p5/s2p0 and context diagnostics | included in V2 commit |
| 9 | Scheme D fallback decision | Orders 4,6,8 | done | not run because A/B/C all produced safe candidates | included in V2 commit |
| 10 | Final visual candidate report | Orders 4,6,8,9 | done | ranks schemes and points to contact sheets | `docs/NEURAL_FILM_LUT_V2_RESULTS.md` |

## First Implementation Choice

Start with **Scheme A: Style-Separated SepLUT**.

Reasons:

- it directly fixes the old basis-collapse issue,
- it separates tone curve and color cross-talk,
- it is closest to real ISP/film rendering practice,
- it can export explicit strength controls,
- it can be bounded and evaluated with existing safety tooling.

Minimal first smoke:

```text
styles: portra_400, velvia_50
images: 3
image_size: 96 or 128
lut_1d_size: 33
lut_3d_size: 17
strengths: 0.0, 0.7, 1.0
teacher: identity + safe-rich + softened film_response_v1
```

Pass criteria:

- `strength=0` output within 1 LSB of input,
- no clipping at strengths 0.7 and 1.0,
- Portra and Velvia outputs differ on the same image,
- no single-head or single-basis collapse,
- contact sheet reads more filmic than local-map saturation sweep.

## Manual Review Items

The user must review:

1. stock-specific contact sheets for each strength,
2. whether the style is filmic or merely saturated,
3. whether stronger tone changes are acceptable despite old L-SSIM/HF failures,
4. final choice of scheme/strength to promote into a production-facing branch.

Everything else should be automated or documented here.

## Execution Results: 2026-06-03

Implemented and evaluated three distilled V2 candidate families:

| Scheme | Output Prefix | Result |
|--------|---------------|--------|
| A Style-Separated SepLUT | `scheme_a_distilled_v1_*` | safe, controllable, usable around `s0p35-s0p5`; heavy above `s1p0` |
| B Conditional NILUT proxy | `scheme_b_nilut_distilled_v1_*` | best first-pass balance of style, safety, and neutral stability |
| C Context-aware 4D LUT proxy | `scheme_c_context4d_distilled_v1_*` | strongest stock separation, but neutral contamination grows quickly |

All A/B/C main runs:

- rendered six stocks over the 20-image eval set,
- exported `s0p5`, `s1p0`, `s1p5`, and `s2p0`,
- produced per-stock `metrics.json`, `manifest.csv`, `contact_sheet.png`,
  `after/`, and `diff_maps/`,
- had zero new clipped pixels,
- stayed inside `[4,251]`.

Key generated artifacts:

```text
outputs/neural_film_lut_v2/targets_v1/targets_manifest.csv
outputs/eval/neural_film_lut_v2/candidate_summary_v1.json
outputs/eval/neural_film_lut_v2/candidate_summary_v1.csv
docs/NEURAL_FILM_LUT_V2_RESULTS.md
```

Current ranking:

1. **Scheme B NILUT `s1p0`** for first manual visual review.
2. **Scheme A SepLUT `s0p35` or `s0p5`** as safer backup.
3. **Scheme C Context4D `s0p5`** as next architecture to optimize, not promote
   yet above subtle strength.

Scheme D is deliberately not started because A/B/C did not fail the basic
visual-style and safety gates. The next useful engineering step is not another
fallback architecture; it is neutral/skin protection for C or full safety
promotion for B.
