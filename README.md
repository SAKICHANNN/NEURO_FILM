# K-MCFM — Content-Preserving Film Imaging

K-MCFM is an experimental film-imaging project focused on preserving the input photograph while applying controllable color and film-inspired effects.

## Current status

The current usable path is a deterministic renderer:

```text
input image
  -> content-safe Lab color transform
  -> optional grain / halation / dust
  -> output image
```

Earlier SDXL SDEdit, LoRA and InstructPix2Pix paths are retained as research history, not the default. Local experiments found either visible detail/identity rewriting or an impractical memory/quality trade-off on the 12GB target GPU.

The renderer is currently a safe engineering baseline, not yet a calibrated reproduction of named film stocks. Its main known limitation is an 8-bit PIL RGB input/output path; high-precision `WorkingImage`, RAW/HDR and profile-preserving output exist only partially and are the next foundation work.

## Current CLI

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --style portra_400 `
  --preset safe-rich `
  --grain 0.2 `
  --halation 0.1 `
  --output result.png
```

Inspect all current controls with:

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py --help
```

Current output should use a `.png` name: the existing save path writes 8-bit PNG regardless of extension. Correct multi-format, 16-bit and ICC-aware export is tracked as P0 foundation work.

## Ultimate direction

The primary product standard is **strong, attractive film-inspired stylization without severe glitch/artifact**. Stock/process authenticity is an optional calibrated-profile claim, not a prerequisite for every useful look.

The target is a three-part system:

1. **Style-safe core** — strong color/effect styling with a hard severe-artifact veto.
2. **FilmCase bounded ML** — after identifiability and Oracle gates, a small retriever hard-selects explicit curve/LUT experts; low-confidence/OOD inputs fall back to the global deterministic champion.
3. **Deferred calibrated / isolated Creative branches** — paired film evidence would be required only for a future calibrated claim; generative editing is excluded from FilmCase.

The autonomous FilmCase path does not depend on the owner supplying images, film/digital pairs, per-image labels or additional preference votes. Unpaired reference images can support only `film-inspired/unpaired-evidence`, not calibrated stock reproduction.

The critical path is:

```text
truth / rights / reproducibility reset
  -> WorkingImage + high-precision color I/O
  -> deterministic profile/reference renderer
  -> frozen severe-artifact + autonomous style benchmark
  -> FilmCase identifiability -> bounded cases -> Oracle gate
  -> simplest useful retrieval/ranker + physical-effects challenge
  -> Windows/Mac/CPU productization and release gates

deferred side lane: future paired Portra 400 + Velvia 50 calibration
```

Read:

- [`AGENTS.md`](AGENTS.md) — current project truth and invariants
- [`docs/ULTIMATE_EXECUTION_TRACKER.md`](docs/ULTIMATE_EXECUTION_TRACKER.md) — active task tree and gates
- [`docs/planning/ULTIMATE_ROADMAP_2026.md`](docs/planning/ULTIMATE_ROADMAP_2026.md) — full research and architecture proposal
- [`docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`](docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md) — detailed unpaired, non-generative research DAG and stop rules
- [`docs/CURRENT_STATUS_2026-05-27.md`](docs/CURRENT_STATUS_2026-05-27.md) — local diffusion/IP2P findings
- [`docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md`](docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md) — current deterministic renderer

## Hardware target

- NVIDIA RTX 5070 Ti Laptop GPU, 12GB
- Apple M5, 32GB unified memory
- CPU fallback

The Style-safe path uses low-resolution routing/parameter prediction and full-resolution deterministic rendering. Generative models are outside FilmCase and no unmeasured 12GB/M5 support is assumed.

## Data and claims

- Flickr, FilmSet, FiveK, community LoRAs and other restricted/unclear sources are research-only by default.
- An unpaired film scan is a style reference, not an input/output transformation; FilmCase must first pass source-controlled identifiability and Oracle-routing gates.
- Accurate named-stock claims require owned or explicitly cleared paired digital/film captures, process/scanner metadata and whole-roll/lab holdouts.
- FilmSet targets are Capture One recipes, not real film scans.
- The full local FiveK source archive was deleted after a verified freeze pack was retained; full-scale FiveK work requires restoring sources.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Baseline on 2026-07-10: `18 passed`.

## License

The repository does not currently contain a root `LICENSE` file. Older documents claimed MIT, but public code/profile/model/data release remains blocked until the owner confirms the intended license and third-party obligations are audited.

---

*Last updated: 2026-07-11. Current default: deterministic content-safe renderer. Target: Style-safe core + autonomous unpaired FilmCase; calibrated lane deferred.*
