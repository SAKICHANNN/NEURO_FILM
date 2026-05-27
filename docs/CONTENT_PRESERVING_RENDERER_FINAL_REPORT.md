# Content-Preserving Renderer Final Report

> Created: 2026-05-27 on the Windows RTX machine.

## Integrated CLI

Command:

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg --style velvia_50 --grain 0.006 --halation 0.05 --dust 0.04 --output output.png --write-layers --write-metrics
```

Smoke command run locally:

```powershell
$input = Get-ChildItem outputs\color_baseline\velvia50_rawpixls20_s0p50_gamutsafe\inputs\*.jpg | Select-Object -First 1 -ExpandProperty FullName
.\.venv\Scripts\python.exe scripts\render_film.py $input --style velvia_50 --grain 0.006 --halation 0.05 --dust 0.04 --output outputs\integration\render_film_smoke.png --write-layers --write-metrics
```

Smoke result:

```text
output=outputs/integration/render_film_smoke.png
bounds=[4, 251]
layers=grain, halation, dust_scratch
metrics=outputs/integration/render_film_smoke.metrics.json
```

## What Is Promoted

- `safe_lab` color engine with `--preset safe-rich` for color stocks.
- Deterministic film-effect layers as optional, separately inspectable layers.
- Ignored generated outputs for contact sheets and smoke renders.

## What Is Not Promoted

- Full-image diffusion/img2img as default.
- Neural LUT as production; it remains research after MVP smoke.
- AI grain/halation/scratch generation; deferred until deterministic layers are visually approved.

## Manual Remaining

- User/Mac visual approval of contact sheets and integrated smoke renders.
- Mac no-password SSH validation and optional Tailscale sign-in from the Windows tracker.
- Private user-photo validation, if desired.
