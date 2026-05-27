# FilmFX Layer Results

> Created: 2026-05-27 on the Windows RTX machine.

## Layer Foundation

Implemented:

- `src/filmfx/layers.py`
  - `FilmLayer` schema,
  - layer validation,
  - alpha/residual safety metrics.
- `src/filmfx/compositor.py`
  - alpha,
  - screen,
  - additive,
  - residual,
  - soft-light compositing.
- `scripts/pipeline_filmfx_layers.py`
- `configs/filmfx_profiles.yaml`
- `scripts/smoke_filmfx_layers.py`

Smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_filmfx_layers.py
```

Expected:

```text
bounds=[13, 243]
warm_screen alpha_max=0.08, affected_percent=44.79
blue_residual residual_abs_max=0.015
```

Decision:

- The layer compositor foundation is ready for deterministic grain, halation,
  dust, scratches, and later AI-generated RGBA/residual layers.
- No film-effect layer is enabled by default yet.
