# Project Structure

This document records the intended repository layout. Keep executable entrypoints stable unless every caller and document reference is updated.

## Root

```text
AGENTS.md                  Agent/project knowledge base. Read first in new sessions.
README.md                  Human-facing overview.
IMPL_PLAN.md               Active-plan pointer plus historical V3 implementation plan.
TASK_BOARD.md              Task ownership and board.
requirements*.txt          Environment requirements.
configs/                   Runtime, model, training, and style configuration.
scripts/                   Stable CLI/training/data entrypoints.
src/                       Library modules and future package code.
tests/                     Test package.
docs/                      Project documentation, audits, handoffs, and references.
agent_reminder/            Timestamped chat handoff notes for future agents.
data/                      Ignored local datasets; only skeleton paths are tracked.
loras/                     Ignored local LoRA/model weights.
outputs/                   Ignored generated outputs.
logs/                      Ignored local run logs.
```

## Docs

```text
docs/
  data/                    Data-license notes, request templates, and dataset-specific docs.
  ops/                     Cloud/resource governance, ownership ledgers, and runbooks.
  planning/                Strategic plans and research subplans that are not active task boards.
  reference/               External reference files such as PDFs.
```

Core status and experiment docs remain directly under `docs/` so the next agent can find them quickly:

- `docs/WINDOWS_TASK_TRACKER.md`
- `docs/CURRENT_STATUS_2026-05-27.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/ARCH_REDESIGN.md`
- `docs/COLOR_BASELINE_RESULTS.md`
- `docs/IP2P_GRID_SEARCH_RESULTS.md`
- `docs/SDXL_LORA_VALIDATION_RESULTS.md`

Strategic rationale and detailed scientific subplans belong under `docs/planning/`; active status, owners and next leaves remain in `docs/ULTIMATE_EXECUTION_TRACKER.md`.

## Preprocessing library

`src/preprocess/` is the product colour-ingress/output boundary. Keep reusable
colour-state and encoding logic here rather than adding format-specific helpers
to individual scripts.

```text
src/preprocess/
  types.py                Typed inspection and WorkingImage contracts.
  pipeline.py             Shared raster/RAW inspection and decode entrypoints.
  raster_decode.py        ICC/cICP-aware SDR and guarded 16-bit decode plus sRGB adapters.
  raw_decode.py           Generic LibRaw scene-linear decode.
  color_management.py     Validated sRGB/Rec.2020 linear math and BT.2020 SDR transfer.
  color_state.py          Fail-closed output-claim policy.
  output_encode.py        Profiled sRGB and cICP-tagged BT.2020 SDR encoders.
```

`WorkingImage` remains the only intended production ingress. The compatibility
path stays 8-bit, while explicit PNG/TIFF16 uses the float safe-Lab/effect path
and quantizes only in `output_encode.py`. The isolated BT.2020 SDR PNG boundary
does not imply renderer integration. Keep future HDR/wide-gamut or tone-map work
inside these same boundaries rather than adding a parallel renderer.

`src/color_engine/` owns reusable deterministic colour-operator math. Its
initial `lab.py` primitive converts supported explicit linear working spaces to
and from D65 CIELAB without clipping. `safe_lab.py` owns the pure Lab-domain
look kernel and immutable source-statistics context; RGB gamut handling,
encoding, stochastic effects and file I/O remain outside that kernel.
`gamut.py` owns explicit destination-working-space gamut policies, while
`rec2020_safe_lab.py` is an isolated colour-only research adapter and is not a
production renderer entry point.

`src/roll2film/` owns isolated research representations and audits for explicit
film-inspired colour operators. `constrained.py` composes the bounded affine,
monotone-spline and dense-LUT stages; `lut.py` owns versioned trilinear and
tetrahedral interpolation plus serialization/Jacobian primitives. These
modules are not production renderer entry points and must not absorb data
acquisition, fitting, evaluation or profile concerns.
`photometric.py` owns canonical L0 exposure/WB and roll-gauge primitives;
`sensitometry.py` owns the unintegrated linear-exposure-to-layer-density
characteristic-curve representation; `sensitometry_print.py` owns the isolated
non-duplicative density-to-print composition; `residual.py` owns the generic
fail-closed sensitometry-print plus bounded tetrahedral-LUT wrapper;
`sensitometry_gauge.py` owns the data-independent inverse-neutral coordinate
gauge. None of these modules implies a
production or calibrated stock operator.

## Inference and replay contracts

`src/inference/` owns reusable deterministic render identity and replay
contracts. Versioned JSON schemas live under `configs/schemas/`, while tracked
immutable profile instances live under `configs/render_profiles/`.

Audit-specific scoring, anonymous-sheet construction and manifest validation
belong under `src/eval/`; execution CLIs remain thin scripts and generated
visual evidence stays under ignored `outputs/` roots.

```text
src/inference/
  render_contract.py      Strict profile/recipe validation, migration, hashing and replay verification.
  tiled_render.py         Local finite-support halo planning, strict execution and core stitching.
src/filmfx/
  tiled_effects.py        Effect-owned adapters that reuse existing effect/compositor math and the generic tiler.
configs/schemas/
  render_profile_v1.schema.json
  render_recipe_v1.schema.json
configs/render_profiles/
  safe_rich_v1.json       Heuristic look-approximation migration; never calibrated stock truth.
```

Keep renderer algorithms in their established colour/effect modules. Contract
files may identify and hash an operator, but they must not become a second
implementation of that operator or load executable code from profile data.
The tiled primitive likewise executes caller-supplied local operators without
reimplementing them; global-statistic and coordinate-random operators require
an explicit context contract before integration. Safe-Lab and sparse
dust/scratch now have experimental context adapters, while grain and physical
halation remain unresolved.
Exact legacy-grain staging lives in dedicated `src/filmfx/tiled_grain.py`
because temporary-resource lifecycle and global reductions are materially
different from local effect adapters. Keep it experimental until later
resource policy and renderer integration gates pass.
Effect-specific adapters remain under `src/filmfx`; they may consume the
generic tiler but cannot move or duplicate effect algorithms into inference.

## Scripts

`scripts/` intentionally remains flat. Many commands in docs and handoff notes call these files directly, so moving them into subfolders would break reproducibility. Prefer adding a README/category index before reorganizing script paths.

Current categories:

- Setup and remote access: `setup_win*.ps1`, `install_openssh_preview_server.ps1`, `run_*_as_admin.bat`
- Data preparation: `download_data.py`, `scrape_films.py`, `build_data_manifest.py`, `build_ip2p_dataset.py`, `combine_ip2p_dataset.py`
- Training: `train_lora*.py`, `train_sdxl_lora.py`, `train_ip2p*.py`, `train_lut.py`, `train_unet.py`
- Inference and validation: `pipeline.py`, `translate.py`, `infer_lora.py`, `grid_search_*.py`, `pipeline_color_baseline.py`, `make_rawpixls_velvia_preview.py`
- Artifact verification: `verify_windows_artifacts.py`

## Safety Rules For Future Reorganization

- Do not move script entrypoints without updating all docs, handoff notes, and command examples.
- Do not commit `.env`, datasets, weights, generated outputs, or local caches.
- Prefer `git mv` for tracked-file moves so history remains readable.
- After moving any referenced file, run `rg` for the old basename and update references.
- Run lightweight checks after structural changes:

```powershell
git status --short --branch
python -m py_compile scripts/pipeline_color_baseline.py scripts/make_rawpixls_velvia_preview.py
```
