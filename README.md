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

The renderer is currently a safe engineering baseline, not yet a calibrated reproduction of named film stocks. The compatibility default remains 8-bit; an opt-in float32 safe-Lab/effects path can now write true 16-bit sRGB PNG/TIFF. HDR, wide gamut and calibrated scene-to-display mapping remain foundation work.

## Product Look Approximation CLI

For a fresh Windows CPython 3.12 environment, install the bounded product
runtime without source builds or the research/ML dependency graph:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-product-v2.txt
```

List the authoritative product catalog without reading an image:

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py --list-product-looks
```

Render one explicitly selected film-inspired Look Approximation. The available
colour choices are `velvia_50`, `portra_400` and `ektar_100`; these names are
product look labels, not claims of calibrated stock response or physical-film
reproduction.

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py input.jpg `
  --product-look portra_400 `
  --look-amount 0.75 `
  --write-recipe `
  --output result.png
```

Optional deterministic `--grain`, `--halation` and `--dust` controls remain
available. Add `--write-layers` and `--write-metrics` when those auxiliary
artifacts are wanted; the product path publishes every requested artifact
create-only as one process-level bundle. This is not simultaneous visibility
or power-loss atomicity.

Inspect all current controls with:

```powershell
.\.venv\Scripts\python.exe scripts\render_film.py --help
```

Current output supports extension-correct 8-bit PNG, JPEG and TIFF with an embedded standard sRGB ICC profile. Add `--output-bit-depth 16` with a `.png`, `.tif` or `.tiff` output for the opt-in float32-to-uint16 path; 16-bit JPEG fails closed.

## Ultimate direction

The primary product standard is **strong, attractive film-inspired stylization without severe glitch/artifact**. Stock/process authenticity is an optional calibrated-profile claim, not a prerequisite for every useful look.

The active target is the deterministic Look Approximation product:

1. **Style-safe core** — strong bounded color/effect styling with a hard severe-artifact veto.
2. **Explicit product selection** — the user chooses one available look and a bounded amount; the product does not infer a film identity from the input.
3. **Replayable workflow** — deterministic preview/export plus hash-bound recipes, receipts and create-only publication.
4. **Deferred calibrated branch** — named-stock calibration reopens only after independently rights-cleared, source-traceable observations with usable roll/process/scanner/scene groups.
5. **Isolated Creative branch** — generative editing remains separate and cannot produce Style-safe final RGB.

Historical Roll2Film, source-only routing and after-only inference experiments
are retained as evidence, not the active product route. Public data counts as
available only after exact assets, rights and group roles are verified.
Unpaired reference or roll images can support only `film-inspired` or
`Look Approximation` claims, not calibrated stock reproduction.

The critical path is:

```text
explicit Look Approximation selection
  -> bounded deterministic render
  -> severe-artifact veto
  -> preview / export / recipe / receipt replay
  -> resource and cross-platform verification
  -> private deliverable product

supporting lane: FilmStyleSafe/FARO evaluation and product fallback
deferred lane: rights-cleared Velvia 50 / Portra 400 / Ektar 100 calibration
```

Read:

- [`AGENTS.md`](AGENTS.md) — current project truth and invariants
- [`docs/ULTIMATE_EXECUTION_TRACKER.md`](docs/ULTIMATE_EXECUTION_TRACKER.md) — active task tree and gates
- [`docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md`](docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md) — deferred stock-first evidence gates and autonomous source-admission program
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

- The current Windows host has no real same-scene digital/film pairs. Its 4,212 local `film_domain` JPEGs remain quarantined by default: the historical 4,210-row lineage audit found 0 eligible rows and no usable roll/source/scanner grouping, and the two additional files still need lineage propagation.
- The traceable local Velvia lane has only 26 unique images, and the 100 IP2P pairs are synthetic smoke data.
- FilmSet is present locally as a complete 21,140-image decompressed tree (~11.26GB): 4,657 train and 628 test identities in each input/target domain. The paper-reported 638 test count is retained as provenance, but runtime uses the archive-observed 628. It can support access-controlled paired-blind film-recipe transfer research, not real-film truth.
- The exact BlueNeg 101-file / 118,929,719-byte bounded acquisition is local. Its grouped-roll experiments did not establish transferable physical-roll information; it remains research-only and is not digital-to-film ground truth or named-stock calibration.
- The local Flickr and remote DigitalFilm collections are quarantined despite being physically downloadable. FilmSet and BlueNeg remain research-only until their exact terms and release obligations are frozen; FiveK/community assets require their own scope and rights audit.
- An unpaired film scan is a target observation, not an input/output pair; Roll2Film must pass synthetic identifiability, shuffled-roll, matched wrong-roll and hidden-transfer gates before using it as method evidence.
- Accurate named-stock claims require owned or explicitly cleared paired digital/film captures, process/scanner metadata and whole-roll/lab holdouts.
- FilmSet targets are Capture One recipes, not real film scans.
- A partial 903-file FiveK freeze is present on this Windows host, while the complete source archive is absent; full-scale FiveK work still requires a separate restore/rights decision and cannot establish film identity.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The authoritative current counts are recorded per completed node in
[`docs/ULTIMATE_EXECUTION_TRACKER.md`](docs/ULTIMATE_EXECUTION_TRACKER.md);
do not use the historical repository-wide count as a release claim.

## License

The repository does not currently contain a root `LICENSE` file. Older documents claimed MIT, but public code/profile/model/data release remains blocked until the owner confirms the intended license and third-party obligations are audited.

---

*Last updated: 2026-09-01. Current default: deterministic content-safe Look Approximation renderer. FARO/FilmStyleSafe supports evaluation and product safety; calibrated named-stock work remains a separately gated future branch.*
