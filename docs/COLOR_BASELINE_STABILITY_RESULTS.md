# Color Baseline Stability Results

> Created: 2026-05-27 on the Windows RTX machine.

## Current Verdict

Status: evaluator scaffold complete; baseline audit not yet run.

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

Run the current deterministic baseline across the seed raw.pixls set and record
per-style clipping, banding, colorfulness, L-SSIM, and failure cases.
