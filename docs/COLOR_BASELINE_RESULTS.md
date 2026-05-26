# Deterministic Color Baseline Results

> Last updated: 2026-05-26 on the Windows RTX machine.

## Decision

Use the deterministic color baseline as the next practical path.

Compared with the failed diffusion paths:

- It cannot alter geometry, faces, clothing, text, or object identity.
- It produces visible color/tonal changes on a standard digital photo.
- It supports black-and-white stocks without relying on prompt obedience.
- It runs quickly on CPU/GPU-independent image operations.

This is not a final film simulator yet. It is a content-safe baseline to improve with better statistics, tone curves, grain, halation, and real reference validation.

## Implemented

- `scripts/build_film_color_stats.py`
  - Builds robust per-style CIELAB statistics from `data/film_domain`.
  - Output: `configs/film_color_stats.json`.

- `scripts/pipeline_color_baseline.py`
  - Applies deterministic Lab mean/std transfer.
  - Preserves image geometry exactly.
  - Supports style grids, black-and-white styles, and light synthetic grain.

## Validation Artifacts

Astronaut digital-photo probes:

- `outputs/color_baseline/portra_800_astronaut_probe/contact_sheets/portra_800_contact_sheet.jpg`
- `outputs/color_baseline/velvia_50_astronaut_probe/contact_sheets/velvia_50_contact_sheet.jpg`
- `outputs/color_baseline/ektar_100_astronaut_probe/contact_sheets/ektar_100_contact_sheet.jpg`
- `outputs/color_baseline/hp5_astronaut_probe/contact_sheets/hp5_contact_sheet.jpg`
- `outputs/color_baseline/tri_x_400_astronaut_probe/contact_sheets/tri_x_400_contact_sheet.jpg`

Recommended first-pass strengths from this probe:

```text
color negative / slide: 0.55-0.75
black and white: 0.75-0.95
luma_strength: 0.45
grain: 0.01
```

## Caveats

- The style statistics are dataset-level averages, so some stocks can look too similar.
- The baseline does not yet model stock-specific H&D curves, halation, dye behavior, or scanner profiles.
- The current grain is simple Gaussian noise, not a physical film grain model.
- Better evaluation needs user-selected real digital photos, not only package sample images.
