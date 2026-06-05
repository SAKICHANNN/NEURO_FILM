# Docs Index

Use this file as the first stop for repository documentation. Most current
project state is intentionally kept in plain Markdown so future sessions can
resume without reading generated outputs.

## Start Here

| File | Purpose |
|------|---------|
| `../AGENTS.md` | Project identity, architecture notes, traps, and operating rules. |
| `PROJECT_STRUCTURE.md` | Intended repository layout and reorganization rules. |
| `CONTENT_PRESERVING_FILM_TASK_TRACKER.md` | Main autonomous development tracker. |
| `CURRENT_STATUS_2026-05-27.md` | Snapshot of current Windows-side status. |
| `CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md` | Final report for the integrated content-preserving renderer. |

## Current Color And Film Rendering Work

| File | Purpose |
|------|---------|
| `COLOR_BASELINE_STABILITY_RESULTS.md` | Safe-rich color baseline stability results. |
| `AI_COLOR_ENGINE_CHALLENGE_TRACKER.md` | Tracker for challengers against `safe_lab + safe-rich`. |
| `AI_COLOR_ENGINE_CHALLENGE_RESULTS.md` | Results for Neural LUT, local maps, and film response volume experiments. |
| `NEURAL_FILM_LUT_V2_TRACKER.md` | New Neural LUT V2 plan for image-specific and stock-specific film rendering. |
| `NEURAL_FILM_LUT_V2_RESULTS.md` | Results for Neural LUT V2 Scheme A/B/C distilled candidates. |
| `CHROMA_ONLY_RESEARCH_RESULTS.md` | Chroma residual research notes. |
| `NEURAL_LUT_RESULTS.md` | Neural LUT scaffold and MVP imitation results. |
| `FILMFX_LAYER_RESULTS.md` | Deterministic film-effects layer results. |
| `PHYSICAL_HALATION_V2_TRACKER.md` | Physical-prior halation V2 plan, implementation, and sweep results. |
| `HALATION_SYSTEM_SPEC.md` | Complete halation architecture, parameter, evaluation, and GUI integration specification. |
| `HALATION_GUI_READINESS_TRACKER.md` | Autonomous tracker for strict validation, presets, GUI schema, layer outputs, and evaluator readiness. |

## Diffusion, LoRA, And Historical Experiments

| File | Purpose |
|------|---------|
| `EXPERIMENT_LOG.md` | Full historical experiment log. |
| `ARCH_REDESIGN.md` | Architecture evolution from earlier designs to current direction. |
| `IP2P_GRID_SEARCH_RESULTS.md` | IP2P grid-search record. |
| `IP2P_BEST_PARAMS_PROVISIONAL.json` | Provisional best IP2P parameters. |
| `SDXL_LORA_VALIDATION_RESULTS.md` | SDXL LoRA validation results. |
| `ONLINE_DATA_AUDIT.md` | Online audit for LoRA/data/dependency gaps. |

## Data, Artifacts, And Reproducibility

| File | Purpose |
|------|---------|
| `DATA_REPRODUCTION_MANIFEST.json` | Data reproduction manifest. |
| `EVAL_SOURCE_BUCKETS.md` | Evaluation source bucket notes. |
| `WINDOWS_ARTIFACT_MANIFEST.json` | Windows artifact manifest. |
| `WINDOWS_ARTIFACT_HANDOFF.md` | Windows artifact handoff notes. |
| `WINDOWS_TASK_TRACKER.md` | Windows setup/task tracker. |

## Setup, Remote Access, And Platform Notes

| File | Purpose |
|------|---------|
| `WIN_REMOTE_SETUP.md` | Windows remote setup guide. |
| `REMOTE_ACCESS_MANUAL_STEPS.md` | Manual remote-access steps. |
| `MAC_ARM_MIGRATION.md` | Mac ARM migration notes. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `data/` | Dataset-specific notes, licenses, and request templates. |
| `planning/` | Planning notes that are not active task boards. |
| `reference/` | External references such as PDFs. |

## Maintenance Notes

- Keep active trackers and current result docs discoverable from this index.
- Do not commit generated outputs, private datasets, weights, caches, or `.env`.
- If a document becomes historical, keep it linked here rather than deleting it.
- If script paths change, update `scripts/README.md`, command examples, and
  handoff notes in the same commit.
