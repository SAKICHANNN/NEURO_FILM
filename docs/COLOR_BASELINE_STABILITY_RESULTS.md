# Color Baseline Stability Results

> Created: 2026-05-27 on the Windows RTX machine.

## Current Verdict

Status: current baseline audit complete; renderer stabilization is required.

The first milestone establishes the shared benchmark configuration and render
safety metrics before changing renderer behavior. Generated outputs remain under
ignored `outputs/` paths.

## Evaluation Source Setup

Config:

```text
configs/eval_buckets.yaml
configs/eval_sources.schema.json
docs/EVAL_SOURCE_BUCKETS.md
```

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\list_eval_sources.py --manifest configs\eval_buckets.yaml --require-existing
```

Result:

```text
schema_version=1
benchmark_sets=2 buckets=10
rawpixls_velvia20_seed manifest_exists=yes
user_private_manual manifest_exists=no
```

## Safety Evaluator Smoke

Command:

```powershell
$before = Get-ChildItem outputs\color_baseline\velvia50_rawpixls20_s0p50_gamutsafe\inputs\*.jpg | Select-Object -First 1 -ExpandProperty FullName
.\.venv\Scripts\python.exe scripts\evaluate_render_safety.py --before $before --after $before --fail-on-clip --fail-on-l-ssim --pretty --output-json outputs\eval\identity_smoke\metrics.json --diff-map outputs\eval\identity_smoke\diff.png
.\.venv\Scripts\python.exe scripts\make_eval_contact_sheet.py --before $before --after $before --output outputs\eval\identity_smoke\contact_sheet.png --title "identity smoke"
```

Result:

```text
passed=true
new_clipped_pixel_count=0
L_ssim=1.000000
gradient_delta_mean=0.000000
high_frequency_delta_mean=0.000000
```

Ignored outputs:

```text
outputs/eval/identity_smoke/metrics.json
outputs/eval/identity_smoke/diff.png
outputs/eval/identity_smoke/contact_sheet.png
```

## Next Action

Implement no-clipping renderer improvements: promote gamut-safe mode for the
production path, add output margin support to the baseline CLI, keep validation
exports as PNG, and make clipping gates fail nonzero.

## Current Baseline Audit

Command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_color_baseline.py --output-dir outputs\eval\baseline_current
```

Audit parameters:

```text
source_manifest=outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/manifest.json
image_count=20
styles=8
strength=0.55
luma_strength=0.35
grain=0.012
gamut_safe=false
output_format=PNG from the current core renderer, to avoid adding JPEG compression noise to metrics
```

Summary:

| Style | New-Clip Images | Hard-Bound Images | Total New Clipped Pixels | Min L-SSIM | Mean L-SSIM | Output Bounds | Max Neutral Contam. |
|------|:---:|:---:|---:|---:|---:|:---:|---:|
| ektar_100 | 20/20 | 20/20 | 1,224,738 | 0.7596 | 0.9233 | 0..255 | 7.341% |
| hp5 | 20/20 | 20/20 | 3,115,664 | 0.7526 | 0.8771 | 0..255 | 0.030% |
| portra_400 | 20/20 | 20/20 | 898,908 | 0.7597 | 0.9236 | 0..255 | 7.389% |
| portra_800 | 20/20 | 20/20 | 1,213,274 | 0.7693 | 0.9252 | 0..255 | 13.770% |
| tri_x_400 | 20/20 | 20/20 | 3,265,263 | 0.7467 | 0.8712 | 0..255 | 0.030% |
| velvia_50 | 20/20 | 20/20 | 1,934,079 | 0.7717 | 0.9272 | 0..255 | 25.226% |
| vision3_250d | 20/20 | 20/20 | 1,162,230 | 0.7628 | 0.9248 | 0..255 | 14.822% |
| vision3_500t | 20/20 | 20/20 | 1,008,403 | 0.7652 | 0.9253 | 0..255 | 5.602% |

Generated artifacts:

```text
outputs/eval/baseline_current/summary.json
outputs/eval/baseline_current/<style>/metrics.json
outputs/eval/baseline_current/<style>/manifest.csv
outputs/eval/baseline_current/<style>/contact_sheet.png
outputs/eval/baseline_current/<style>/diff_maps/
```

Regression fixture manifest:

```text
configs/eval_regression_fixtures.json
```

Conclusion:

- The current default baseline fails the no-clipping gate for every audited image and style.
- The default Lab conversion path emits `lab2rgb` gamut clipping warnings on some images.
- L-channel structure is not stable enough for the final color-only gate; the worst styles/images are far below `L_ssim >= 0.995`.
- B&W styles are especially aggressive because current contrast logic changes luminance too strongly.
- Velvia 50 has the highest neutral contamination in this run, so chroma gain needs bounded compression and neutral protection.
