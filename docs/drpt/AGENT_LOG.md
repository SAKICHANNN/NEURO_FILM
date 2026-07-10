# Project Agent Log

Durable handoff log for non-trivial DRPT-governed work. Keep implementation truth in the active project documents and use this file to record decisions, evidence, risks and the next safe handoff.

---

## 2026-07-10 — Ultimate research and roadmap reset

- **Parent goal / node:** `ULT` / `U0.0`
- **DRPT level / mode:** L2, Mode B; root integration owner with three read-only research leaves
- **User objective:** conduct a thorough study and produce a complete plan to make K-MCFM “ultimate”
- **Primary workflow:** `dev-research-reliability`
- **Routing / secondary skills:** `codex-super-router`, `codex-super-research-harness`, `scientific-research-harness`, `codex-super-aiml-harness`, `plan-tracker-discipline`, `exa-search-preference` (Exa unavailable; native web fallback), `drpt-bi-governance`, `project-agent-log-discipline`, `project-structure-steward`

### Decisions

1. Retired diffusion-first SDXL/IP2P/SDEdit as the default architecture. Local evidence shows content/detail rewriting or infeasible 12GB training behavior.
2. Selected a calibrated hybrid architecture: deterministic high-precision Reference core, optional bounded curve/LUT/bilateral-grid predictor, isolated Creative generative mode.
3. Defined named-film identity as `stock + exposure/process + scan/print interpretation`, not one prompt or LUT per stock.
4. Selected Portra 400 and Velvia 50 as the proposed two-stock calibration pilot; no capture/purchase was authorized or started.
5. Classified Flickr, FilmSet, FiveK, FilmGrainStyle and community LoRAs as research-only by default; production claims require owned/cleared paired data.
6. Replaced a single aggregate evaluator with an initial four-part design: safety, authenticity, physical effects, and human/product performance. This was superseded on 2026-07-11 by the five-scorecard design that adds film-style salience.
7. Kept FLUX.2 Klein 4B as an optional teacher/Creative challenger. Official sources conflict on roughly 8GB versus 13GB VRAM, so 12GB compatibility is not assumed.

### Files changed

- `docs/planning/ULTIMATE_ROADMAP_2026.md` — research synthesis, architecture, data/capture, evaluation, product, risk and source matrix
- `docs/ULTIMATE_EXECUTION_TRACKER.md` — active DRPT tree, DoR/DoD, dependencies, branch gates, first 30 days and experiment queue
- `AGENTS.md` — corrected current truth and project invariants
- `README.md` — replaced failed diffusion quick start with current renderer/status and plan links
- `IMPL_PLAN.md` — added active-plan supersession banner while preserving historical V3 content
- `TASK_BOARD.md` — replaced stale diffusion board with compact current queue
- `docs/CONTENT_PRESERVING_FILM_TASK_TRACKER.md` — marked historical for active planning
- `docs/WINDOWS_TASK_TRACKER.md` — marked historical machine snapshot
- `docs/planning/GAP_ANALYSIS.md` and `docs/ONLINE_DATA_AUDIT.md` — added supersession warnings
- `docs/planning/README.md` — indexed the new strategy and active tracker
- `docs/drpt/AGENT_LOG.md` — this entry

No production code, data, model, output or user-owned untracked file was modified.

### Evidence and verification

- Audited current renderer, preprocessing, effects, neural LUT, FiveK, data/license and experiment reports.
- Confirmed `scripts/render_film.py` accepts only `safe_lab`, uses PIL RGB input and writes 8-bit PNG.
- Confirmed full FiveK sources were deleted locally after a 6.98GB freeze pack was retained.
- Confirmed no GitHub Actions workflow and 5 test files / 18 test functions.
- Ran `.\.venv\Scripts\python.exe -m pytest -q` before and after documentation work; final pre-commit result: `18 passed in 13.18s` (an earlier post-edit run passed in 9.48s).
- Ran `.\.venv\Scripts\python.exe scripts\render_film.py --help` to verify the documented current CLI surface.
- Read-only leaf checks compiled 105 Python files in memory, imported core modules and reported `pip check` clean.
- External claims were checked against papers, official repositories/model cards, standards bodies, manufacturer data sheets and official product documentation.
- Checked all changed Markdown for trailing whitespace: 0 findings.
- Checked relative Markdown link targets in all changed documents: 0 missing targets.
- Ran `git diff --check`: clean after removing Markdown hard-break whitespace.

### Unresolved risks and approvals

- Root license is absent despite historical MIT claims; owner/legal decision required.
- Current branch has no matching remote branch and contains local commits; no push authorized.
- Existing untracked `halationguide.md` and `scripts/make_velvia50_scheme_comparison_sheets.py` remain protected user work.
- Capture/lab/scanner budget, participant releases, model/data downloads, paid GPU, external contact and public release all require explicit approval.
- Exact capture cost, lab/scanner repeatability, M5 thermal performance and final numerical gates remain pilot measurements, not settled facts.

### Handoff

`U0.1` was completed in this change set. Next ready leaf is `U0.3`: manifest/leakage repair, followed by `U0.4/U4` CI/evaluation foundation. Code implementation should not begin before refreshing Git state and claiming bounded files in `docs/ULTIMATE_EXECUTION_TRACKER.md`.

---

## 2026-07-11 — Velvia 50 user-preference anchor

- **Parent goal / node:** `ULT` / `U4.7`
- **DRPT level / mode:** L1, Mode A
- **Input:** user stated the preferred numbered schemes as `53, 55, 56, 33, 09, 03, 02, 01`
- **Resolution:** mapped the numbers through `outputs/contact_sheets/velvia50_all_schemes_numbered_20260616/NUMBER_MAP.csv`
- **Decision:** retain the set as qualitative preference evidence without assuming list order is a ranking; treat `53/55/56/09/01` as full 20-render anchors and `33/03/02` as smoke-only direction cues
- **Interpretation:** all five comparable choices are deterministic baseline/gamut-safe Lab variants, supporting the calibrated deterministic core; this is preference evidence, not film-authenticity evidence
- **Files changed:** `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/planning/ULTIMATE_ROADMAP_2026.md`, `docs/drpt/AGENT_LOG.md`
- **Verification:** exact scheme labels and render counts checked against `NUMBER_MAP.csv`; no output image, mapping file or user-owned untracked file modified
- **Handoff:** include the five full anchors in the frozen U4 blind-study set and rerender them on one common held-out image set before inferring strength/style parameters

---

## 2026-07-11 — Add minimum film-style salience gate

- **Node/parent goal:** `U4.8` / `ULT`
- **Trigger:** user reported that many theoretically stronger schemes look technically clean but have little film style
- **Skills used:** `plan-tracker-discipline` primary; `project-agent-log-discipline` secondary
- **Decision:** expand evaluation from four to five independent scorecards by adding film-style salience between safety and stock authenticity
- **Files changed:** `AGENTS.md`, `TASK_BOARD.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/planning/ULTIMATE_ROADMAP_2026.md`, `docs/drpt/AGENT_LOG.md`
- **Verification intent:** future U4 evaluation must run separate color-only and full-look blind tests against neutral input, the preferred deterministic anchors and real film references; global saturation alone cannot pass
- **Risk/unknown:** salience thresholds remain a preregistered pilot decision; this feedback establishes the gate but does not yet quantify it
- **Handoff state:** U4.6 remains ready for evaluator implementation; no render, model, data or user-owned untracked artifact was modified
