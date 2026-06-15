# FiveK Auto-Base Model Tracker

> Created: 2026-06-15
>
> Split from: `docs/FIVEK_AUTO_OPTIMIZE_TRACKER.md`
>
> Goal: compress FiveK into small supervised assets and use those assets to
> build an optional pre-film auto-base layer.

## 1. Product Boundary

FiveK is not a film-stock dataset. It should not teach Velvia, Portra, Vision3,
grain, halation, or stock identity.

FiveK may teach a separate pre-film auto-base layer:

```text
input image or RAW-derived RGB
  -> auto optimize layer
       exposure / white balance / contrast / highlight-shadow balance
       clean neutral base image
  -> film stock color layer
  -> grain / halation / defects
```

The output of the auto-base layer must remain a neutral photographic base. It
should improve the input before film rendering, not replace the film stock
transform.

## 2. Film-Ready But Not Film-Styled

The auto-base layer may add:

```text
natural exposure correction
clean white balance
reasonable contrast
gentle highlight shoulder
mild shadow toe
subtle reduction of harsh digital rendering
```

The auto-base layer must not add:

```text
Velvia saturation or hue identity
Portra skin-color identity
Vision3 shadow color identity
grain
halation
bloom
obvious LUT style
content-changing generation
```

The product wording should be **film-ready photographic base**, not "film
style".

## 3. Local Source State

Current local FiveK assets:

```text
data/raw/fivek/fivek_dataset.tar
data/raw/fivek/fivek_dataset.tar.sha1
data/raw/fivek/fivek_index.html
data/raw/fivek/fivek_expert_a_paths.txt
data/raw/fivek/fivek_expert_c_paths.txt
data/raw/fivek/expert_tiff/c/
```

Important size facts measured before this split:

```text
data/raw/fivek/                      about 290 GB
data/raw/fivek/fivek_dataset.tar     about 50.8 GB
data/raw/fivek/expert_tiff/c/        about 260 GB
```

The source archive and TIFFs are ignored local data and must not be committed.

## 4. Cache V1 Scaffold

Create ignored output assets under:

```text
outputs/fivek_auto_optimize/cache_v1/
```

The first cache version contains:

```text
inputs_512/
targets_512/
pairs_512/
manifest.csv
summary.json
contact_sheet.png
```

Each row represents one paired supervised example:

```text
id
source_name
target_expert
target_tiff
input_proxy
target_512
pair_512
width
height
target_min
target_max
target_luma_mean
target_luma_std
target_chroma_mean
```

The first implementation uses available Expert C TIFFs as targets. Because RAW
decoding from the 50GB tar is a separate risk and dependency, Cache V1 uses a
conservative input proxy produced from the same expert TIFF:

```text
expert TIFF
  -> input proxy: compressed, lower-contrast, mildly desaturated display image
  -> target: Expert C downsampled display image
```

This does not claim to be true RAW-to-expert supervision. It is a compact first
asset for testing the auto-optimize model surface, manifest discipline, metrics,
and storage policy.

Cache V1 must be treated as:

```text
allowed:
  manifest discipline
  pair layout validation
  training/evaluation interface smoke tests
  contact sheet policy
  small-model plumbing

not allowed:
  final quality judgment
  FiveK deletion justification
  claims of RAW-to-expert learning
  stock-specific film style training
```

## 5. Cache V1 Result

Generated command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_auto_optimize_cache.py `
  --count 256 `
  --size 512 `
  --output-dir outputs\fivek_auto_optimize\cache_v1
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/cache_v1/inputs_512/
outputs/fivek_auto_optimize/cache_v1/targets_512/
outputs/fivek_auto_optimize/cache_v1/pairs_512/
outputs/fivek_auto_optimize/cache_v1/manifest.csv
outputs/fivek_auto_optimize/cache_v1/summary.json
outputs/fivek_auto_optimize/cache_v1/contact_sheet.png
```

Measured cache size:

```text
inputs_512:   256 files, about 12.17 MB
targets_512:  256 files, about 71.66 MB
pairs_512:    256 files, about 26.77 MB
manifest.csv: 256 rows
```

Summary means from `summary.json`:

```text
target_luma_mean:  0.2447
target_luma_std:   0.2494
target_chroma_mean: 0.0502
input_luma_mean:   0.2219
input_luma_std:    0.1872
input_chroma_mean: 0.0276
```

Verification:

- `py_compile` passed for the cache builder.
- Smoke cache generation with 4 examples passed.
- Full cache generation produced 256 manifest rows.
- Each of `inputs_512`, `targets_512`, and `pairs_512` contains 256 files.
- `contact_sheet.png` is visually nonblank and shows the expected
  `input_proxy | Expert C target` layout.

## 6. Future Cache Versions

### V2: RAW-Derived Inputs

Use `fivek_dataset.tar` to locate the matching original RAW/DNG files, decode
them with LibRaw/rawpy or a documented CLI, and pair:

```text
RAW neutral/default render -> Expert C TIFF render
```

This is the correct supervision for RAW input auto optimization.

V2 requirements:

- preserve source RAW path and Expert C TIFF path in the manifest;
- record LibRaw/rawpy parameters, white-balance mode, demosaic mode, output bit
  depth, and whether auto-brightness was disabled;
- generate at least 1024px training pairs and a smaller 512px smoke subset;
- produce contact sheets with `RAW/default render | Expert C target | delta`;
- include representative validation subsets for dark, high-key, foliage, skin,
  sky, indoor tungsten, mixed light, and high dynamic range scenes.

### V3: Multi-Expert Targets

Add Expert A/B/D/E where available. This can teach user preference variation or
allow target-style choice:

```text
neutral
bright
contrast
warm
natural
```

### V4: Compact Statistical Knowledge

Extract small global/local response assets:

```text
tone curve pairs
Lab histogram deltas
highlight/shadow correction distributions
white-balance shift distributions
scene brightness buckets
```

These assets can support a deterministic auto optimizer without a large neural
network.

## 7. Model Candidate Order

Prefer mature, high-resolution-friendly enhancement families:

```text
1. Bilateral-grid affine / HDRNet-style model
2. Image-adaptive 3D LUT
3. Local parametric filters
```

Avoid starting with:

```text
large full-resolution U-Net
diffusion enhancement
content-changing generative model
```

The candidate must be optional, controllable, and easy to disable.

## 8. Dependency Order

| Order | Task | Status | Completion Test |
|------:|------|:---:|-----------------|
| 1 | Inspect FiveK layout | done | local source paths and sizes recorded |
| 2 | Implement Cache V1 builder | done | `scripts/build_fivek_auto_optimize_cache.py` builds Expert C proxy cache |
| 3 | Generate Cache V1 | done | manifest, summary, and contact sheet exist |
| 4 | Build RAW-derived Cache V2 | partial | 8-image smoke and 64-image mini cache pair RAW/default render with Expert C target; user visual validation pending |
| 5 | Extract compact response assets | partial | mini64 tone/chroma/luma response assets written; broader representative cache pending |
| 6 | Fit deterministic response baseline | partial | mini64 response baseline split into tone-locked and optional color/WB residual modes; user visual validation pending |
| 7 | Train first lightweight candidate | pending | candidate improves base tone without stock-style drift |
| 8 | Compare to Expert C | partial | mini64 baseline metrics and contact sheet generated |
| 9 | Storage recommendation | pending | explicit keep/cold-store/delete recommendation for FiveK assets |

## 9. Non-Goals

- Do not train or promote a film stock style model from FiveK alone.
- Do not delete FiveK raw assets from this tracker.
- Do not modify the film color default.
- Do not replace `data/film_domain` with FiveK.
- Do not make the FiveK auto-base layer responsible for stock-specific film
  identity.

## 10. RAW-Derived Cache V2 Smoke Result

Generated command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_raw_cache.py `
  --count 8 `
  --size 512 `
  --output-dir outputs\fivek_auto_optimize\raw_cache_v2_smoke
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/raw_cache_v2_smoke/raw_renders_512/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/targets_512/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/pairs_512/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/manifest.csv
outputs/fivek_auto_optimize/raw_cache_v2_smoke/summary.json
outputs/fivek_auto_optimize/raw_cache_v2_smoke/contact_sheet.png
```

Measured smoke size:

```text
raw_renders_512: 8 files, about 0.37 MB
targets_512:     8 files, about 2.09 MB
pairs_512:       8 files, about 0.82 MB
manifest.csv:    8 rows
missing_count:   0
```

What this proves:

- FiveK Expert C TIFF names can be matched to RAW/DNG tar members by stem.
- The local `fivek_dataset.tar` can be read without full extraction.
- RAW members can be temporarily extracted, decoded through the shared
  `src.preprocess` path, and paired with Expert C targets.
- The smoke contact sheet is nonblank and correctly laid out as
  `RAW/default render | Expert C`.

What this does not prove yet:

- The generic LibRaw/rawpy RAW render is the best input representation for the
  final auto-base model.
- The Expert C target is the preferred product aesthetic.
- The 8-image smoke subset is representative enough to justify deleting or
  cold-storing FiveK.

Manual visual validation needed:

```text
outputs/fivek_auto_optimize/raw_cache_v2_smoke/contact_sheet.png
```

## 11. RAW-Derived Mini64 And Response Stats

Generated mini cache command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_raw_cache.py `
  --count 64 `
  --size 512 `
  --output-dir outputs\fivek_auto_optimize\raw_cache_v2_mini64
```

Generated compact response command:

```powershell
.\.venv\Scripts\python.exe scripts\extract_fivek_response_stats.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --output-dir outputs\fivek_auto_optimize\response_stats_v1_mini64 `
  --bins 32 `
  --sample-stride 2
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/raw_cache_v2_mini64/
outputs/fivek_auto_optimize/response_stats_v1_mini64/response_stats.json
outputs/fivek_auto_optimize/response_stats_v1_mini64/response_curves.npz
outputs/fivek_auto_optimize/response_stats_v1_mini64/per_image_stats.csv
outputs/fivek_auto_optimize/response_stats_v1_mini64/response_curves.png
```

Measured response stats:

```text
raw_cache_v2_mini64 rows: 64
missing_count: 0
response bins: 32
per_image_stats rows: 64
NPZ keys:
  bin_centers
  counts
  target_luma_by_raw_luma
  luma_delta_by_raw_luma
  target_chroma_by_raw_luma
  chroma_delta_by_raw_luma
  rgb_delta_by_raw_luma
```

What this proves:

- The response extractor can turn a RAW-derived cache into small tone/chroma
  response assets.
- The output is small enough to keep while raw TIFF/RAW data remains ignored.
- The assets are suitable for deterministic baseline experiments and for
  initializing lightweight auto-base candidates.

What this does not prove yet:

- The 64-image sample is representative enough for product behavior.
- The response assets are sufficient to delete or cold-store FiveK.
- The selected Expert C target is the final product aesthetic.

Storage decision remains unchanged:

```text
Do not delete data/raw/fivek/.
Mini64 response stats are useful compression artifacts, not deletion proof.
```

## 12. Deterministic Response Baseline V1 Mini64

Generated baseline command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v1_mini64 `
  --strength 1.0
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/response_baseline_v1_mini64/baseline/
outputs/fivek_auto_optimize/response_baseline_v1_mini64/manifest.csv
outputs/fivek_auto_optimize/response_baseline_v1_mini64/summary.json
outputs/fivek_auto_optimize/response_baseline_v1_mini64/contact_sheet.png
```

Mean metrics on the 64-image mini subset:

```text
raw_target_luma_mae:       0.061296
baseline_target_luma_mae:  0.055659
raw_target_rgb_mae:        0.064870
baseline_target_rgb_mae:   0.060201
raw_target_chroma_mae:     0.031508
baseline_target_chroma_mae:0.030872
baseline_luma_delta_mean:  0.017139
baseline_chroma_delta_mean:-0.001175
```

Interpretation:

- The deterministic baseline modestly moves RAW/default renders toward Expert C
  in luma, RGB, and chroma MAE.
- The transform is compact and explainable: it applies an interpolated RGB delta
  as a function of RAW/default luminance.
- The result is intentionally not stock-specific and should remain a neutral
  photographic base, not a Velvia/Portra/Vision3 look.

What this proves:

- The compact response assets can be applied back to images without training.
- The response baseline gives a measurable improvement over RAW/default render
  on the same Mini64 subset.
- The baseline provides a conservative first candidate for later lightweight
  model comparisons.

What this does not prove yet:

- It does not prove the response baseline is visually preferable.
- It does not prove generalization beyond the Mini64 subset.
- It does not replace a future train/validation split.
- It does not justify deleting `data/raw/fivek/`.
- User feedback on the 0.25/0.50/0.75/1.00 legacy RGB-delta sweep showed
  that even low strength can visibly over-correct color temperature and that
  the strength ladder is not perceptually well separated.

Manual visual validation needed:

```text
outputs/fivek_auto_optimize/response_baseline_v1_mini64/contact_sheet.png
```

## 13. Response Split Plan: Tone vs Color/WB

Reason for the split:

- The first deterministic baseline applied a single RGB-delta curve learned
  from RAW/default to Expert C.
- That approach improved numeric distance to Expert C, but it mixed exposure,
  tone curve, chroma, and white-balance decisions into one scalar strength.
- User visual review found that even `strength=0.25` changed color temperature
  too much, while `0.25` and `0.75` did not feel clearly separated in the
  contact sheets.
- Therefore the single `strength` control is not a valid product abstraction
  for a neutral auto-base layer.

New decomposition:

```text
RAW/default render
  -> tone_locked pass
       Uses luma_delta_by_raw_luma.
       Scales RGB channels together to target luminance.
       Intention: change exposure/tone while preserving source RGB ratios.
  -> optional color/WB residual pass
       Computes old legacy_rgb response minus tone_locked response.
       Controlled separately by color_strength.
       Default: 0.0.
```

CLI modes:

```text
--mode tone_locked
  Default. Preserves source RGB ratios during the tone pass.

--mode legacy_rgb
  Reproduces the original RGB-delta baseline for failure comparison.
```

Primary controls:

```text
--tone-strength
  Amount of luma response to apply.

--color-strength
  Amount of Expert C residual color/WB response to add after tone locking.
  Default must stay 0.0 until user visual validation supports raising it.

--strength
  Backward-compatible alias for tone strength.
```

Required measurements:

```text
baseline_target_luma_mae
baseline_target_rgb_mae
baseline_target_chroma_mae
baseline_luma_delta_mean
baseline_chroma_delta_mean
baseline_red_green_ratio_delta
baseline_blue_green_ratio_delta
```

Acceptance rule for this stage:

- Prefer a candidate that preserves small red/green and blue/green ratio deltas
  over one that wins Expert C RGB MAE by moving white balance aggressively.
- A neutral auto-base default should be judged first by visual non-annoyance
  and stable WB, second by Expert C metric improvement.
- `legacy_rgb` remains useful as a warning baseline, not a default.

Manual validation outputs to generate:

```text
outputs/fivek_auto_optimize/response_baseline_v2_tone_locked/
outputs/fivek_auto_optimize/response_baseline_v2_tone_color_s010/
outputs/fivek_auto_optimize/response_baseline_v2_legacy_rgb/
```

## 14. Response Split V2 Mini64 Result

Generated commands:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v2_tone_locked `
  --mode tone_locked `
  --tone-strength 1.0 `
  --color-strength 0.0

.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v2_tone_color_s010 `
  --mode tone_locked `
  --tone-strength 1.0 `
  --color-strength 0.10

.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v2_legacy_rgb `
  --mode legacy_rgb `
  --tone-strength 1.0
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/response_baseline_v2_tone_locked/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v2_tone_color_s010/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v2_legacy_rgb/contact_sheet.png
```

Mean metrics:

```text
candidate                  luma_mae  rgb_mae   chroma_mae  R/G delta   B/G delta
raw/default -> Expert C     0.061296  0.064870  0.031508    n/a         n/a
tone_locked c0.00           0.055659  0.060514  0.032865   -0.000994  -0.002551
tone_locked c0.10           0.055659  0.060446  0.032547   -0.006802  -0.000824
legacy_rgb                  0.055659  0.060201  0.030872   -0.058393  +0.014512
```

Interpretation:

- `tone_locked c0.00` keeps almost the same luma improvement as `legacy_rgb`
  while dramatically reducing red/green and blue/green ratio drift.
- `legacy_rgb` still wins Expert C RGB/chroma MAE, but it does so by moving
  white balance strongly, matching the user's visual complaint.
- `tone_locked c0.10` is a cautious diagnostic point; even 0.10 color residual
  increases R/G drift enough that it should not become the default without
  visual approval.

User visual validation result:

```text
legacy_rgb:
  rejected as a default candidate.
  Reason: strong white-balance/color-temperature movement and no trustworthy
  product improvement despite slightly better Expert C metrics.

tone_locked c0.00:
  rejected as a useful default candidate.
  Reason: it suppresses WB drift numerically, but the user saw no meaningful
  visual improvement.

tone_locked c0.10:
  rejected as a default candidate.
  Reason: it starts reintroducing color/WB drift without solving the usefulness
  problem.
```

Archived validation priority:

```text
1. outputs/fivek_auto_optimize/response_baseline_v2_tone_locked/contact_sheet.png
2. outputs/fivek_auto_optimize/response_baseline_v2_tone_color_s010/contact_sheet.png
3. outputs/fivek_auto_optimize/response_baseline_v2_legacy_rgb/contact_sheet.png
```

Product implication:

- Do not promote the Mini64 deterministic response baseline to a default
  product layer.
- Do not use Expert C closeness as the primary success criterion for product
  color or WB.
- FiveK may still be useful for input-format coverage, RAW/default render
  study, exposure statistics, and future supervised experiments, but the
  current compact response baseline should be treated as a rejected baseline.
- Any later auto-base layer needs a better objective than "move toward Expert C",
  and should probably avoid learning global WB/color-temperature corrections
  unless that control is explicit and separately validated.

## 15. WB-Anchored Color Residual Plan

Reason for this follow-up:

- Fully removing color response with `tone_locked` also removed visible
  usefulness.
- Applying the full RGB response with `legacy_rgb` produced unacceptable
  color-temperature/WB drift.
- The next hypothesis is that the useful part, if any, may live in local color,
  saturation, or scene-dependent residuals, while the harmful part is mostly a
  global illuminant shift.

Implementation rule:

```text
RAW/default render
  -> tone_locked luma response
  -> add a controllable fraction of the legacy RGB residual
  -> measure robust source and candidate channel ratios on midtones
  -> diagonally correct candidate so global R/G and B/G return toward source
  -> restore candidate luminance so the WB anchor does not undo tone response
```

CLI mode:

```text
--mode wb_anchored
```

Controls:

```text
--tone-strength
  Luma response strength.

--color-strength
  Amount of legacy RGB residual to test before WB anchoring.

--wb-anchor-strength
  Strength of global R/G and B/G restoration.
  Default: 1.0.
```

Why this differs from `tone_locked`:

- `tone_locked` blocks nearly all color residuals.
- `wb_anchored` allows color residuals, but removes their global
  color-temperature component.

Why this differs from `legacy_rgb`:

- `legacy_rgb` can freely move the whole image white balance toward Expert C.
- `wb_anchored` treats global WB as an input property unless the user exposes a
  separate WB control later.

Generated validation outputs:

```text
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c025/
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c050/
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c100/
```

## 16. WB-Anchored Residual V3 Mini64 Result

Generated commands:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v3_wb_anchor_c025 `
  --mode wb_anchored `
  --tone-strength 1.0 `
  --color-strength 0.25 `
  --wb-anchor-strength 1.0

.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v3_wb_anchor_c050 `
  --mode wb_anchored `
  --tone-strength 1.0 `
  --color-strength 0.50 `
  --wb-anchor-strength 1.0

.\.venv\Scripts\python.exe scripts\evaluate_fivek_response_baseline.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64\manifest.csv `
  --response outputs\fivek_auto_optimize\response_stats_v1_mini64\response_curves.npz `
  --output-dir outputs\fivek_auto_optimize\response_baseline_v3_wb_anchor_c100 `
  --mode wb_anchored `
  --tone-strength 1.0 `
  --color-strength 1.0 `
  --wb-anchor-strength 1.0
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c025/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c050/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c100/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_comparison/contact_sheet.png
```

Mean metrics:

```text
candidate                  luma_mae  rgb_mae   chroma_mae  R/G delta   B/G delta
tone_locked c0.00           0.055659  0.060514  0.032865   -0.000994  -0.002551
wb_anchor c0.25             0.055659  0.060552  0.032787   +0.000318  +0.001707
wb_anchor c0.50             0.055659  0.060551  0.032736   +0.000027  +0.002219
wb_anchor c1.00             0.055658  0.060632  0.032958   -0.000848  +0.003086
legacy_rgb                  0.055659  0.060201  0.030872   -0.058393  +0.014512
```

Interpretation:

- The WB anchor suppresses the strongest failure mode of `legacy_rgb`: global
  R/G drift drops from about `-0.058` to roughly `0.000` to `0.001`.
- The anchored variants keep the same luma improvement as the previous
  deterministic baselines.
- RGB/chroma MAE no longer beats `legacy_rgb`; this is expected because the
  metric rewards movement toward Expert C's WB choice.
- Whether the anchored residuals are visually useful remains unresolved and
  requires user review.

Manual visual validation priority:

```text
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_comparison/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c025/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c050/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v3_wb_anchor_c100/contact_sheet.png
```

## 17. Expert TIFF ICC Bug And V4 Rebuild

Bug:

- FiveK Expert C TIFF files are 16-bit RGB TIFFs with embedded ICC profiles.
- Local inspection found the embedded profile is `ProPhoto RGB` / ROMM.
- The original cache builders used `ImageOps.exif_transpose(image).convert("RGB")`.
- That path converts image mode, but does not perform embedded ICC -> sRGB
  color management.
- Therefore old cached Expert C PNG targets interpreted ProPhoto RGB values as
  if they were already sRGB.

Impact:

- Previous Expert C target PNGs under these outputs are color-managed
  incorrectly:

```text
outputs/fivek_auto_optimize/cache_v1/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/
outputs/fivek_auto_optimize/raw_cache_v2_mini64/
outputs/fivek_auto_optimize/response_stats_v1_mini64/
outputs/fivek_auto_optimize/response_baseline_v1_mini64*
outputs/fivek_auto_optimize/response_baseline_v2_*
outputs/fivek_auto_optimize/response_baseline_v3_*
```

- The old baseline images themselves may be internally reproducible, but their
  target reference and response statistics were learned against the wrong
  target color interpretation.
- All product judgment must use ICC-corrected V4 or later outputs.

Fix:

- `scripts/build_fivek_raw_cache.py` now converts Expert TIFF targets from the
  embedded ICC profile to sRGB before resizing/caching.
- `scripts/build_fivek_auto_optimize_cache.py` applies the same fix for the
  older proxy cache builder.

Measured smoke difference:

```text
old naive target vs ICC-corrected target mean MAE on 8-image smoke: 0.034654
```

V4 generated outputs:

```text
outputs/fivek_auto_optimize/raw_cache_v2_icc_smoke/
outputs/fivek_auto_optimize/raw_cache_v2_mini64_icc/
outputs/fivek_auto_optimize/response_stats_v2_mini64_icc/
outputs/fivek_auto_optimize/response_baseline_v4_icc_tone_locked/
outputs/fivek_auto_optimize/response_baseline_v4_icc_wb_anchor_c050/
outputs/fivek_auto_optimize/response_baseline_v4_icc_legacy_rgb/
outputs/fivek_auto_optimize/response_baseline_v4_icc_comparison/contact_sheet.png
```

V4 mean metrics:

```text
candidate              luma_mae  rgb_mae   chroma_mae  R/G delta   B/G delta
raw/default -> Expert   0.073732  0.080232  0.055720    n/a         n/a
tone_locked ICC         0.058086  0.068083  0.051858   -0.000924  -0.004172
wb_anchor ICC c0.50     0.058124  0.068616  0.053264   -0.000706  +0.000772
legacy_rgb ICC          0.058163  0.068684  0.053362   -0.070292  -0.015197
```

Manual visual validation priority after the fix:

```text
outputs/fivek_auto_optimize/response_baseline_v4_icc_comparison/contact_sheet.png
outputs/fivek_auto_optimize/raw_cache_v2_mini64_icc/contact_sheet.png
outputs/fivek_auto_optimize/raw_cache_v2_icc_smoke/contact_sheet.png
```

## 18. ICC V5 Chroma/WB Guard Plan

User validation after the ICC fix:

- The ICC-corrected Expert C column is now visually plausible and much better
  than the pre-ICC target.
- It is still a stronger human edit than the desired default auto-base layer.
- Main remaining complaints:
  - Slightly too saturated.
  - White balance/color temperature still feels over-calibrated.

New candidate rule:

```text
RAW/default render
  -> apply ICC-corrected response statistics
  -> use WB anchoring to suppress global R/G and B/G drift
  -> use chroma anchoring to limit saturation growth over the source
  -> keep the luma/tone response as intact as possible
```

Why this is different from earlier attempts:

- `legacy_rgb` learned Expert C color/WB too directly.
- `tone_locked` protected WB but had too little useful visual effect.
- `wb_anchored` protected global WB but could still inherit too much saturation
  or local chroma growth from Expert C.
- V5 adds an explicit chroma guard so saturation becomes a separate, bounded
  control rather than an accidental side effect.

New CLI controls:

```text
--chroma-anchor-strength
  0.0 disables chroma guarding.
  1.0 fully limits chroma growth to the configured source-relative headroom.

--chroma-headroom
  Allowed chroma increase over the source before guarding clamps the candidate.
  Example: 0.05 allows up to about 5 percent chroma growth.
```

Validation outputs to generate:

```text
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_soft/
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_medium/
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_strict/
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_neutral/
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_comparison/contact_sheet.png
```

## 19. ICC V5 Chroma/WB Guard Result

Generated candidates:

```text
soft:
  tone_strength=1.00
  color_strength=0.50
  wb_anchor_strength=1.00
  chroma_anchor_strength=0.35
  chroma_headroom=0.15

medium:
  tone_strength=1.00
  color_strength=0.35
  wb_anchor_strength=1.00
  chroma_anchor_strength=0.70
  chroma_headroom=0.08

strict:
  tone_strength=1.00
  color_strength=0.25
  wb_anchor_strength=1.00
  chroma_anchor_strength=1.00
  chroma_headroom=0.03

neutral:
  tone_strength=0.85
  color_strength=0.20
  wb_anchor_strength=1.00
  chroma_anchor_strength=1.00
  chroma_headroom=0.00
```

Mean metrics:

```text
candidate           luma_mae  rgb_mae   chroma_mae  chroma_delta  R/G delta   B/G delta
v4 wb c0.50          0.058124  0.068616  0.053264    +0.011377    -0.000706  +0.000772
v5 soft              0.058123  0.068781  0.053597    +0.011087    +0.000263  +0.000863
v5 medium            0.058111  0.069009  0.054197    +0.010099    -0.000076  +0.000451
v5 strict            0.058103  0.069484  0.055468    +0.008485    -0.000465  +0.000092
v5 neutral           0.058553  0.069826  0.055861    +0.006766    -0.000212  -0.000009
Expert C target      n/a       n/a       n/a          strongest    human edit  human edit
```

Interpretation:

- V5 keeps WB movement very small after the final WB-anchor pass.
- Chroma growth decreases from `v4 wb c0.50` to `v5 neutral`, but the tradeoff
  is reduced closeness to Expert C RGB/chroma metrics.
- `neutral` is the best technical answer to the user's "less saturated and less
  over-calibrated" feedback.
- `medium` is the more balanced candidate if `neutral` feels too weak.
- These are still Mini64 deterministic baselines, not product defaults.

Manual visual validation priority:

```text
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_comparison/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_neutral/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v5_icc_guard_medium/contact_sheet.png
```

## 20. ICC V6 Filtered Target Plan

User validation after V5:

- V5 candidates still look too similar to each other.
- The remaining objection is not that Expert C ICC is unchanged relative to RAW;
  it is that Expert C ICC still does not improve the core objection relative to
  the previous round: it remains too saturated and still slightly over-corrects
  white balance.
- Therefore the next fix must change the target used to fit response stats, not
  only guard the final baseline output.

New target construction:

```text
RAW/default render + Expert C ICC
  -> take most of Expert C luma/tone movement
  -> take only a small fraction of Expert C chroma-vector movement
  -> cap chroma growth relative to RAW/default
  -> anchor global R/G and B/G back toward RAW/default
  -> save as filtered pseudo-target
  -> extract response stats from RAW/default -> filtered pseudo-target
  -> evaluate baseline from filtered response stats
```

New script:

```text
scripts/build_fivek_filtered_targets.py
```

Initial V6 target parameters:

```text
luma_strength=0.82
chroma_strength=0.18
chroma_headroom=0.02
wb_anchor_strength=1.00
```

Expected behavior:

- The filtered target should be visibly closer to Expert C in exposure/tone than
  RAW/default.
- It should not inherit Expert C's full saturation increase.
- It should not inherit Expert C's global WB/color-temperature correction.
- If it is too weak, increase luma response before increasing chroma response.

Outputs to generate:

```text
outputs/fivek_auto_optimize/filtered_targets_v1_mini64_icc/
outputs/fivek_auto_optimize/response_stats_v3_filtered_mini64_icc/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target_wb_c025/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target_tone115/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_comparison/contact_sheet.png
```

## 21. ICC V6 Filtered Target Result

Filtered target command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_filtered_targets.py `
  --manifest outputs\fivek_auto_optimize\raw_cache_v2_mini64_icc\manifest.csv `
  --output-dir outputs\fivek_auto_optimize\filtered_targets_v1_mini64_icc `
  --luma-strength 0.82 `
  --chroma-strength 0.18 `
  --chroma-headroom 0.02 `
  --wb-anchor-strength 1.0
```

Generated response/baseline outputs:

```text
outputs/fivek_auto_optimize/response_stats_v3_filtered_mini64_icc/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target_wb_c025/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target_tone115/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_comparison/contact_sheet.png
```

Filtered target metrics:

```text
Expert C ICC vs RAW rgb_mae:        0.080232
filtered target vs RAW rgb_mae:     0.059612
Expert C ICC luma_delta_mean:       +0.048099
filtered target luma_delta_mean:    +0.039425
Expert C ICC chroma_delta_mean:     +0.040522
filtered target chroma_delta_mean:  +0.006971
filtered target vs Expert rgb_mae:  0.035523
```

Baseline metrics against the filtered target:

```text
candidate       luma_mae  rgb_mae   chroma_mae  chroma_delta  R/G delta   B/G delta
v6 tone          0.047629  0.047344  0.011798    +0.010850    -0.000428  -0.003292
v6 wb c0.25      0.047637  0.047169  0.008995    +0.007946    +0.000157  -0.000050
v6 tone115       0.047860  0.047587  0.012815    +0.012383    -0.000740  -0.003880
```

Interpretation:

- V6 changes the fitting target itself rather than only guarding the final
  output.
- The filtered target keeps most of Expert C's luma lift while reducing average
  chroma growth by roughly 83 percent.
- `v6 wb c0.25` is currently the best technical candidate for the user's stated
  preference: it keeps WB drift near zero and tracks the filtered target's low
  chroma growth.
- `v6 tone115` is not preferred unless the user finds V6 too weak; it increases
  tone/chroma movement without improving WB.

Manual visual validation priority:

```text
outputs/fivek_auto_optimize/response_baseline_v6_filtered_comparison/contact_sheet.png
outputs/fivek_auto_optimize/filtered_targets_v1_mini64_icc/contact_sheet.png
outputs/fivek_auto_optimize/response_baseline_v6_filtered_target_wb_c025/contact_sheet.png
```

## 22. Delete-Prep Freeze Pack V1

Reason:

- The project may delete the large local FiveK source data to recover disk
  space.
- Before deletion, the useful results must be preserved in a compact,
  reproducible, higher-precision form.
- The freeze pack must avoid carrying obsolete pre-ICC outputs or large
  low-value duplicates.

Budget decision:

```text
target budget: 20-24 GB
hard budget:   30 GB
actual:         6.98 GB
```

Generated command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_freeze_pack.py `
  --hp-count 128 `
  --gold-count 64 `
  --max-size 1536 `
  --output-dir outputs\fivek_auto_optimize\freeze_v1 `
  --max-gb 30
```

Generated high-precision assets:

```text
outputs/fivek_auto_optimize/freeze_v1/
outputs/fivek_auto_optimize/freeze_v1/manifest.csv
outputs/fivek_auto_optimize/freeze_v1/summary.json
outputs/fivek_auto_optimize/freeze_v1/contact_sheet.png

outputs/fivek_auto_optimize/freeze_v1/hp128_1536_srgb16/raw_default_srgb16/
outputs/fivek_auto_optimize/freeze_v1/hp128_1536_srgb16/expert_c_icc_srgb16/
outputs/fivek_auto_optimize/freeze_v1/hp128_1536_srgb16/filtered_target_srgb16/
outputs/fivek_auto_optimize/freeze_v1/hp128_1536_srgb16/previews/

outputs/fivek_auto_optimize/freeze_v1/gold64_original_samples/raw/
outputs/fivek_auto_optimize/freeze_v1/gold64_original_samples/expert_tiff_c/
```

Counts verified:

```text
raw_default_srgb16 TIFFs:       128
expert_c_icc_srgb16 TIFFs:      128
filtered_target_srgb16 TIFFs:   128
gold original RAW files:         64
gold original Expert TIFF files: 64
missing source pairs:             0
```

Precision verified:

```text
sample raw_default_srgb16:      uint16, shape (1023, 1536, 3)
sample expert_c_icc_srgb16:     uint16, shape (1023, 1536, 3)
sample filtered_target_srgb16:  uint16, shape (1023, 1536, 3)
```

Color/target policy:

```text
RAW/default:
  rawpy/LibRaw generic decode through src.preprocess.
  display sRGB transfer encoded to uint16 TIFF.

Expert C:
  original 16-bit TIFF read with tifffile.
  ProPhoto/ROMM transfer decoded.
  Bradford D50 -> D65 chromatic adaptation.
  matrix conversion to sRGB.
  sRGB transfer encoded to uint16 TIFF.

Filtered target:
  luma_strength=0.82
  chroma_strength=0.18
  chroma_headroom=0.02
  wb_anchor_strength=1.00
```

High-precision response stats:

```powershell
.\.venv\Scripts\python.exe scripts\extract_fivek_freeze_response_stats.py `
  --manifest outputs\fivek_auto_optimize\freeze_v1\manifest.csv `
  --output-dir outputs\fivek_auto_optimize\freeze_v1\response_stats_filtered_srgb16 `
  --bins 64 `
  --sample-stride 3
```

Generated stats:

```text
outputs/fivek_auto_optimize/freeze_v1/response_stats_filtered_srgb16/response_stats.json
outputs/fivek_auto_optimize/freeze_v1/response_stats_filtered_srgb16/response_curves.npz
outputs/fivek_auto_optimize/freeze_v1/response_stats_filtered_srgb16/per_image_stats.csv
outputs/fivek_auto_optimize/freeze_v1/response_stats_filtered_srgb16/response_curves.png
```

Disk result:

```text
freeze_v1 total files: 903
freeze_v1 total size:  6.98 GB
C: free after build:   43.32 GB
```

Deletion implication:

- The freeze pack preserves the current useful FiveK direction without keeping
  all 5000 Expert TIFFs or the full RAW tar.
- It does not replace the complete dataset for future large-scale supervised
  training.
- It is sufficient for:
  - reproducing current v6 filtered-target direction,
  - validating 16-bit/high-pixel pipeline behavior on selected samples,
  - refitting compact response curves,
  - checking future color-management regressions.

## 23. Large FiveK Source Deletion

Deletion date:

```text
2026-06-15
```

Deleted after freeze-pack verification:

```text
data/raw/fivek/expert_tiff/
data/raw/fivek/fivek_dataset.tar
```

Pre-delete verification:

```text
freeze_v1 summary missing: []
raw_default_srgb16 TIFFs:       128
expert_c_icc_srgb16 TIFFs:      128
filtered_target_srgb16 TIFFs:   128
gold original RAW files:         64
gold original Expert TIFF files: 64
freeze_v1 size before delete:    6.98 GB
```

Post-delete verification:

```text
Test-Path data/raw/fivek/expert_tiff:       False
Test-Path data/raw/fivek/fivek_dataset.tar: False
freeze_v1 files:                            903
freeze_v1 size:                             6.98 GB
C: free after deletion:                     333.4 GB
```

Remaining local FiveK source metadata:

```text
data/raw/fivek/fivek_dataset.tar.sha1
data/raw/fivek/fivek_expert_a_paths.txt
data/raw/fivek/fivek_expert_c_paths.txt
data/raw/fivek/fivek_index.html
```

Important consequence:

- Full FiveK-scale supervised experiments are no longer possible from local
  disk without re-downloading or restoring the deleted sources.
- Current local development should use:

```text
outputs/fivek_auto_optimize/freeze_v1/
outputs/fivek_auto_optimize/freeze_v1/response_stats_filtered_srgb16/
outputs/fivek_auto_optimize/filtered_targets_v1_mini64_icc/
outputs/fivek_auto_optimize/response_baseline_v6_filtered_comparison/
```
