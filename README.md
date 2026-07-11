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

The target now has a separate algorithm-paper lane and product lane:

1. **Style-safe core** — strong color/effect styling with a hard severe-artifact veto.
2. **Roll2Film research** — identify and apply a reusable explicit colour operator from unpaired images grouped by physical film roll. The paper must demonstrate colour transfer; a benchmark, selector or rejection policy alone is not the contribution.
3. **FARO/FilmStyleSafe support** — severe-artifact evaluation and full-resolution fallback remain product/system safeguards and research evaluation, not the primary paper.
4. **Deferred calibrated / isolated Creative branches** — controlled paired film evidence is required for future named-stock calibration; generative editing remains separate.

The no-data Roll2Film leaves do not depend on the owner supplying images,
film/digital pairs, per-image labels or additional preference votes. Public
data counts as available only after a concrete file endpoint is verified.
Unpaired reference or roll images can support only
`film-inspired/unpaired-evidence`, not calibrated stock reproduction.

The critical path is:

```text
truth / rights / reproducibility reset
  -> WorkingImage + high-precision color I/O
  -> deterministic profile/reference renderer
  -> explicit invertible colour-operator contract
  -> known-operator pseudo-roll identifiability simulator
  -> FilmSet paired-blind unpaired colour-transfer experiment
  -> BlueNeg grouped-roll information pilot
  -> simplest surviving Roll2Film inference method
  -> hidden transfer + style + artifact evaluation
  -> Windows/Mac/CPU productization and release gates

supporting lane: FilmStyleSafe/FARO evaluation and product fallback
deferred lane: future paired Portra 400 + Velvia 50 calibration
```

Read:

- [`AGENTS.md`](AGENTS.md) — current project truth and invariants
- [`docs/ULTIMATE_EXECUTION_TRACKER.md`](docs/ULTIMATE_EXECUTION_TRACKER.md) — active task tree and gates
- [`docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`](docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md) — primary algorithm-first colour-transfer hypothesis, data audit, novelty boundary and falsifiable experiment program
- [`docs/planning/FARO_RESEARCH_PROGRAM_2026.md`](docs/planning/FARO_RESEARCH_PROGRAM_2026.md) — supporting artifact evaluation and product/system-risk program
- [`docs/planning/ULTIMATE_ROADMAP_2026.md`](docs/planning/ULTIMATE_ROADMAP_2026.md) — full research and architecture proposal
- [`docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`](docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md) — retained unpaired retrieval baseline and ablations
- [`docs/CURRENT_STATUS_2026-05-27.md`](docs/CURRENT_STATUS_2026-05-27.md) — local diffusion/IP2P findings
- [`docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md`](docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md) — current deterministic renderer

## Hardware target

- NVIDIA RTX 5070 Ti Laptop GPU, 12GB
- Apple M5, 32GB unified memory
- CPU fallback

The Style-safe path uses low-resolution routing/parameter prediction and full-resolution deterministic rendering. Generative models are outside FilmCase and no unmeasured 12GB/M5 support is assumed.

## Data and claims

- The current Mac has no real same-scene digital/film pairs. Its 3,896 local Flickr JPEGs are quarantined: the historical 4,210-row lineage audit found 0 eligible rows and no usable roll/source/scanner grouping.
- The traceable local Velvia lane has only 26 unique images, and the 100 IP2P pairs are synthetic smoke data.
- FilmSet is currently absent but **downloadability is verified**, not inferred from a paper: the official Kaggle CLI listed its files and one 412KB member was fetched successfully. The ~11.26GB corpus has 5,285 RAW originals and paired Cinema/ClassNeg/Velvia Capture One targets; it can support paired-blind film-recipe transfer research, not real-film truth.
- BlueNeg is currently absent but **downloadability is verified** through its public file tree and a successful byte-range read. It offers 491 frames grouped into 53 rolls; the ~956MB initial 8-bit lanes can test the roll-group hypothesis, not digital-to-film ground truth.
- The local Flickr and remote DigitalFilm collections are quarantined despite being physically downloadable. FilmSet and BlueNeg remain research-only until their exact terms and release obligations are frozen; FiveK/community assets require their own scope and rights audit.
- An unpaired film scan is a target observation, not an input/output pair; Roll2Film must pass synthetic identifiability, shuffled-roll, matched wrong-roll and hidden-transfer gates before using it as method evidence.
- Accurate named-stock claims require owned or explicitly cleared paired digital/film captures, process/scanner metadata and whole-roll/lab holdouts.
- FilmSet targets are Capture One recipes, not real film scans.
- The full FiveK source archive and the historical freeze pack are both absent on this Mac; full-scale FiveK work requires restoring sources and still cannot establish film identity.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Baseline on 2026-07-10: `18 passed`.

## License

The repository does not currently contain a root `LICENSE` file. Older documents claimed MIT, but public code/profile/model/data release remains blocked until the owner confirms the intended license and third-party obligations are audited.

---

*Last updated: 2026-07-12. Current default: deterministic content-safe renderer. Research target: Roll2Film colour transfer. FARO/ChromaticTail supports evaluation and product safety; calibrated lane remains deferred.*
