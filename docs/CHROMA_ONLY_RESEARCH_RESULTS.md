# Chroma-Only Research Results

> Created: 2026-05-27 on the Windows RTX machine.

## Decision

Keep chroma-only residual rendering as an experimental research path. Do not
promote it over `safe_lab --preset safe-rich` until it beats the Part 1 renderer
on visual quality while preserving the same safety gates.

## Chroma Residual Scaffold

Implemented:

- `src/models/chroma_residual.py`
  - freezes source Lab L,
  - applies bounded `delta_a/delta_b`,
  - enforces output headroom.
- `scripts/smoke_chroma_residual.py`

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_chroma_residual.py
```

Expected output:

```text
mean_abs_l_delta=0.0121
bounds=[4, 251]
```

## Chroma-Only Diffusion Feasibility

Feasible interface:

```text
RGB input -> Lab
freeze L
model predicts bounded delta_a/delta_b at low resolution
edge-aware upscale / smooth
safe compositor -> RGB with output margin
```

Current decision:

- Do not train chroma diffusion yet.
- First train/evaluate Neural LUT and simple chroma residual predictors against
  the Part 1 safe-rich renderer.
- Chroma diffusion becomes worthwhile only if deterministic/Neural LUT color is
  too global or cannot handle semantic local color without artifacts.

Main risks:

- color bleeding at object boundaries,
- dirty skin/neutral contamination,
- hallucinated chroma texture,
- extra training cost with no clear win over LUT/residual maps.
