# Color Baseline Stability Results

> Created: 2026-05-27 on the Windows RTX machine.

## Current Verdict

Status: no-clipping renderer path and first artifact/color guards implemented;
safe-rich profile tuning still needs work.

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

Tune rich-but-natural per-style profiles, especially luminance preservation on
the hard seed images 09 and 11.

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

## No-Clipping Renderer Pass

Implementation changes:

- `scripts/pipeline_color_baseline.py` now supports `--output-margin`, `--format png`, and `--fail-on-clip`.
- `--gamut-safe` is formalized through `--gamut-mode source`; `--gamut-mode chroma` is available as a hue-preserving chroma compression variant.
- `--tone-rolloff` adds an optional monotonic Lab L roll-off hook.
- The CLI now saves PNG when requested or when the output suffix is `.png`; it no longer forces JPEG for every output.

Tracker completion command:

```powershell
$input = Get-ChildItem outputs\color_baseline\velvia50_rawpixls20_s0p50_gamutsafe\inputs\*.jpg | Select-Object -First 1 -ExpandProperty FullName
.\.venv\Scripts\python.exe scripts\pipeline_color_baseline.py $input --style velvia_50 --strength 0.50 --luma-strength 0.25 --grain 0 --gamut-safe --output-margin 4 --format png --fail-on-clip --output outputs\eval\noclip_smoke\velvia_50_safe.png
.\.venv\Scripts\python.exe scripts\evaluate_render_safety.py --before $input --after outputs\eval\noclip_smoke\velvia_50_safe.png --fail-on-clip --output-margin 4
```

Single-image result:

```text
after_min=4
after_max=251
new_clipped_pixel_count=0
L_ssim=0.996379
```

Full seed-set command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_color_baseline.py --output-dir outputs\eval\baseline_noclip_s0p50 --strength 0.50 --luma-strength 0.25 --grain 0 --gamut-safe --output-margin 4
```

Summary:

| Style | New-Clip Images | Hard-Bound Images | Total New Clipped Pixels | Min L-SSIM | Mean L-SSIM | Output Bounds | Max Neutral Contam. |
|------|:---:|:---:|---:|---:|---:|:---:|---:|
| ektar_100 | 0/20 | 0/20 | 0 | 0.9004 | 0.9806 | 4..251 | 2.754% |
| hp5 | 0/20 | 0/20 | 0 | 0.9244 | 0.9579 | 4..251 | 0.000% |
| portra_400 | 0/20 | 0/20 | 0 | 0.8800 | 0.9789 | 4..251 | 1.191% |
| portra_800 | 0/20 | 0/20 | 0 | 0.9031 | 0.9824 | 4..251 | 7.810% |
| tri_x_400 | 0/20 | 0/20 | 0 | 0.9217 | 0.9552 | 4..251 | 0.000% |
| velvia_50 | 0/20 | 0/20 | 0 | 0.9361 | 0.9874 | 4..251 | 17.386% |
| vision3_250d | 0/20 | 0/20 | 0 | 0.9045 | 0.9824 | 4..251 | 8.489% |
| vision3_500t | 0/20 | 0/20 | 0 | 0.9067 | 0.9834 | 4..251 | 1.575% |

Conclusion:

- The no-clipping gate now passes across all 160 seed renders with `[4, 251]` output headroom.
- L-SSIM still does not pass the final Part 1 gate on difficult images, mainly because the current renderer still modifies luminance.
- Neutral contamination remains high for Velvia 50 and some Vision3/Portra settings. This moves directly into the artifact/banding and safe-rich profile work.

## Artifact And Chroma Guard Pass

Implementation changes:

- Added `configs/color_guardrails.json` with per-style neutral protection, skin protection, chroma gain/boost caps, B&W absolute chroma caps, and dither defaults.
- `pipeline_color_baseline.py` supports `--use-guardrails`, `--neutral-protect`, `--skin-protect`, `--max-chroma-gain`, `--max-chroma-boost`, `--max-chroma-absolute`, and `--dither`.
- The safety evaluator already reports banding, high-frequency residual, gradient delta, and neutral/skin chroma contamination.
- Regression contact sheets are generated by `scripts/audit_color_baseline.py` under each ignored style output directory.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_color_baseline.py --output-dir outputs\eval\baseline_guarded_s0p50 --strength 0.50 --luma-strength 0.25 --grain 0 --gamut-safe --output-margin 4 --use-guardrails
```

Summary compared with the no-clipping pass:

| Style | New Clip | Bounds | Min L-SSIM | Mean L-SSIM | Max Neutral Contam. | Mean Chroma |
|------|:---:|:---:|---:|---:|---:|---:|
| ektar_100 | 0/20 | 4..251 | 0.8737 | 0.9789 | 0.326% | 14.617 |
| hp5 | 0/20 | 4..251 | 0.9242 | 0.9577 | 0.000% | 2.873 |
| portra_400 | 0/20 | 4..251 | 0.8584 | 0.9775 | 0.011% | 14.386 |
| portra_800 | 0/20 | 4..251 | 0.8737 | 0.9806 | 0.368% | 15.235 |
| tri_x_400 | 0/20 | 4..251 | 0.9216 | 0.9551 | 0.000% | 2.847 |
| velvia_50 | 0/20 | 4..251 | 0.9079 | 0.9859 | 5.641% | 16.392 |
| vision3_250d | 0/20 | 4..251 | 0.8716 | 0.9803 | 0.700% | 15.397 |
| vision3_500t | 0/20 | 4..251 | 0.8741 | 0.9813 | 0.108% | 14.143 |

Guardrail effect:

- Velvia 50 max neutral contamination dropped from 17.386% to 5.641%.
- Portra 800 dropped from 7.810% to 0.368%.
- Vision3 250D dropped from 8.489% to 0.700%.
- HP5/Tri-X now remain near monochrome after adding absolute B&W chroma caps.
- Worst-case L-SSIM decreased for some color stocks because the current luma transfer and gamut compression still interact on hard images. This is a tuning problem for the safe-rich profile stage, not a clipping blocker.
