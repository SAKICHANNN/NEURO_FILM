# AI Color Engine Challenge Results

> Created: 2026-05-27 on the Windows RTX machine.

## Current Status

The branch now contains reproducible challengers for scheme #2 and scheme #3.
Scheme #3 has beaten `safe_lab + safe-rich` on the automatic gates; scheme #2
has not.

## Branch

```text
research/color-engines-beat-safe-lab
```

## Champion

Previous champion:

```text
safe_lab + safe-rich
```

New automatic-metric leader:

```text
local_maps_tuned_s1p2
```

The production default should still remain `safe_lab + safe-rich` until the
contact sheets are visually approved.

## Comparison Summary

| Engine | Run | Promote Count | Verdict |
|--------|-----|---:|---------|
| #1 Safe Lab | `outputs/eval/baseline_saferich/summary.json` | baseline | previous champion |
| #2 Neural LUT | `outputs/eval/color_engine_challenge/neural_lut_s300_compare.json` | 0/6 | fails L-SSIM/HF gates |
| #3 Local bounded maps | `outputs/eval/color_engine_challenge/local_maps_tuned_s1p2_compare.json` | 6/6 | automatic promote candidate |

## #3 Local/Semantic Bounded Maps

Implemented:

- `src/models/local_color_maps.py`
- `scripts/evaluate_local_color_maps.py`
- `scripts/compare_color_engines.py`

Best command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_local_color_maps.py --styles ektar_100,portra_400,portra_800,velvia_50,vision3_250d,vision3_500t --output-dir outputs\eval\color_engine_challenge\local_maps_tuned_s1p2 --local-strength 1.2
.\.venv\Scripts\python.exe scripts\compare_color_engines.py --baseline outputs\eval\baseline_saferich\summary.json --challenger outputs\eval\color_engine_challenge\local_maps_tuned_s1p2\summary.json --label local_maps_tuned_s1p2 --output outputs\eval\color_engine_challenge\local_maps_tuned_s1p2_compare.json
```

Result:

| Style | Promote | Chroma Gain | Min L-SSIM | Bounds |
|-------|:---:|---:|---:|:---:|
| ektar_100 | yes | +3.36% | 0.99518 | 4..251 |
| portra_400 | yes | +3.26% | 0.99563 | 4..251 |
| portra_800 | yes | +3.25% | 0.99517 | 4..251 |
| velvia_50 | yes | +3.51% | 0.99545 | 4..251 |
| vision3_250d | yes | +3.14% | 0.99504 | 4..251 |
| vision3_500t | yes | +3.21% | 0.99504 | 4..251 |

Safety notes:

- new clipped pixels: 0 for all six styles,
- hard-bound images: 0 for all six styles,
- neutral contamination: within tolerance for all six styles,
- high-frequency delta: within the 5% gate for all six styles.

Decision:

- `local_maps_tuned_s1p2` beats the previous champion on automatic metrics.
- It should be visually reviewed before becoming the production color default.

## #2 Image-Adaptive Neural LUT

Implemented:

- `scripts/evaluate_neural_lut.py`
- `scripts/train_neural_lut.py --basis-init-std`

Main tuning command:

```powershell
.\.venv\Scripts\python.exe scripts\train_neural_lut.py --limit 20 --styles ektar_100,portra_400,portra_800,velvia_50,vision3_250d,vision3_500t --steps 300 --image-size 96 --lut-size 17 --num-basis 6 --output-dir outputs\neural_lut\challenge_color6_s300
```

Training result:

```text
device=cuda
example_count=120
initial_l1=0.013045
final_l1=0.005296
improvement_ratio=2.46x
```

Safety comparison:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_neural_lut.py --checkpoint outputs\neural_lut\challenge_color6_s300\model.pt --styles ektar_100,portra_400,portra_800,velvia_50,vision3_250d,vision3_500t --output-dir outputs\eval\color_engine_challenge\neural_lut_s300
.\.venv\Scripts\python.exe scripts\compare_color_engines.py --baseline outputs\eval\baseline_saferich\summary.json --challenger outputs\eval\color_engine_challenge\neural_lut_s300\summary.json --label neural_lut_s300 --output outputs\eval\color_engine_challenge\neural_lut_s300_compare.json
```

Result:

- promote count: 0/6,
- no clipping and bounds passed,
- L-SSIM failed because min L-SSIM dropped to 0.99498,
- high-frequency gate failed for four styles,
- style conditioning collapsed: all six styles produced effectively the same
  output.

Second tuning attempt:

```powershell
.\.venv\Scripts\python.exe scripts\train_neural_lut.py --limit 20 --styles ektar_100,portra_400,portra_800,velvia_50,vision3_250d,vision3_500t --steps 800 --image-size 96 --lut-size 17 --num-basis 12 --basis-init-std 0.01 --output-dir outputs\neural_lut\challenge_color6_s800_b12_init
```

Result:

```text
initial_l1=0.013081
final_l1=0.005184
improvement_ratio=2.52x
```

This did not solve the collapse: the encoder still assigned almost all weight
to one basis LUT for every style. More brute-force parameter sweeps are unlikely
to help without an architecture or loss change.

## Verdict

1. `local_maps_tuned_s1p2` is the new automatic-metric winner.
2. `safe_lab + safe-rich` remains the safest production default until visual
   approval.
3. Neural LUT remains research-only; the next real fix is anti-collapse training
   or explicit style-separated basis heads, not another small parameter sweep.

## Follow-Up: Film Response Volume V1

User visual feedback on #3:

- even the strongest local-map outputs looked mostly like saturation gain,
- the output did not read as stock-specific film style.

New experiment:

- `src/models/film_response_volume.py`
- `scripts/evaluate_film_response_volume.py`

Design:

```text
safe-rich base
  -> stock-specific tone response
  -> shadow/midtone/highlight color casts
  -> hue-sector pushes
  -> neutral/skin protection
  -> gamut-safe compression
  -> [4, 251] output margin
```

Full output runs:

```text
outputs/eval/color_engine_challenge/film_response_v1_s1p0/
outputs/eval/color_engine_challenge/film_response_v1_s1p4/
```

Each stock contact sheet has three columns:

```text
before | safe-rich | response
```

Automatic comparison:

| Run | Promote Count | Boundary | Style Strength |
|-----|---:|:---:|---------------|
| `film_response_v1_s1p0` | 0/6 | 4..251, no clipping | medium |
| `film_response_v1_s1p4` | 0/6 | 4..251, no clipping | strong |

Interpretation:

- This is not a safe-baseline replacement; it intentionally changes tone and
  therefore fails the old L-SSIM/HF gates.
- It is a visual style candidate for the next branch of work: film response
  should be judged by a different metric bundle than the conservative
  content-preserving color baseline.
