# Content-Preserving Film Rendering Task Tracker

> Created: 2026-05-27
>
> Purpose: a detailed execution guide for independent development after the diffusion-first route failed content preservation. This tracker splits the next work into three major missions:
>
> 1. Improve the current deterministic color baseline until it is precise, stable, non-clipping, artifact-resistant, natural, and still richly colored.
> 2. Explore advanced color simulation where AI predicts color transforms, LUTs, chroma residuals, or color operators, rather than generating the final image.
> 3. Explore film artifact simulation as separate transparent/residual layers, so grain, halation, scratch, dust, bloom, and leaks do not damage the original image structure.

## Core Thesis

The project should move from full-image generative editing to content-preserving AI color rendering.

Do not let an AI decoder output the final RGB image unless it is explicitly marked as an experiment. The default product path must keep original structure locked and only apply controlled color transforms and composited film-effect layers.

```text
final_image =
    deterministic_render(
        original_structure,
        color_transform_or_chroma_layer,
        film_artifact_layers
    )
```

## Non-Negotiable Safety Constraints

These constraints apply to every branch, experiment, and merge:

1. No committed `.env`, API secrets, datasets, LoRA weights, generated outputs, caches, or private user images.
2. No final RGB diffusion/img2img route may be promoted as default unless it passes strict structure safety gates.
3. Default output must preserve geometry, face identity, text/logo shapes, object boundaries, and high-frequency luminance detail.
4. No hard clipping in production output:
   - 8-bit PNG/JPEG after image must not hit `0` or `255` except where the input already did and the result is explicitly documented.
   - Preferred production export should reserve output headroom, for example `[4, 251]` or a configurable equivalent.
5. No severe artifacts:
   - no painterly smearing,
   - no posterization/banding,
   - no dirty color bleeding on skin,
   - no halo where no highlight supports it,
   - no generated texture that changes scene content.
6. Rich color is required, but rich color must be achieved with gamut compression, tone/chroma curves, and perceptual constraints, not by hard channel clipping.
7. Every visual experiment must generate:
   - before/after pairs,
   - contact sheet,
   - machine-readable manifest,
   - safety metrics,
   - written conclusion.

## Tracker Operating Contract

This tracker should be treated like `docs/WINDOWS_TASK_TRACKER.md`: work from top to bottom, skip only blocked/manual tasks, and keep statuses current after every milestone.

### Autonomous One-Day Development Contract

This file is designed to guide an agent through a full autonomous development day without asking the user routine questions.

Default behavior:

1. Work in dependency order from the `Dependency Order` table.
2. Do not ask the user for implementation preferences when the tracker, repo conventions, or safety constraints give a reasonable answer.
3. Use web research and repo search proactively when a claim, tool, model, metric, or implementation detail is uncertain.
4. Prefer conservative, reversible, well-scoped implementation choices.
5. Keep generated outputs, datasets, weights, and private files ignored; commit code/docs/config/tests only.
6. Run long jobs when the tracker requires them, but write logs and keep enough metadata to resume after a crash.
7. Update this tracker and result docs as work completes, fails, or changes status.
8. Leave only true manual/account/GUI/private-data actions for the user, and report those at the end.

The agent should not pause to ask about:

- exact filenames when a clear repo convention exists;
- whether to add a small helper script needed by the current order;
- whether to run tests or safety evaluation;
- whether to browse for primary sources before implementing research tasks;
- whether to commit at an explicit commit node;
- whether to push at an explicit push node;
- whether to reduce scope inside a research task if the full version is blocked and a useful smoke test is possible.

The agent must pause or mark `manual` only for:

- account login, GUI install, or credential entry;
- private user photo selection or user aesthetic approval;
- publishing/licensing decisions;
- destructive cleanup outside ignored generated-output directories;
- paid/cloud resource decisions not already approved;
- legal/privacy choices.

### Autonomous Decision Defaults

When multiple reasonable options exist, use these defaults:

| Decision Area | Default |
|---------------|---------|
| Output image format for safety validation | PNG, not JPEG |
| Production color path | deterministic `safe_lab` until a later path beats it on metrics and visual review |
| First AI color route | Neural LUT imitation of the safe renderer before unpaired/style losses |
| First film-effect route | deterministic layer compositor before AI layer generators |
| Diffusion usage | experimental only unless output space is chroma/residual/layer constrained |
| Eval images | use existing raw.pixls-derived manifests and ignored local eval folders |
| Tests | add lightweight synthetic/fixture tests before large visual grids |
| Metrics | prefer conservative gates; calibrate thresholds on identity transform |
| Commits | commit at the tracker-defined commit node after tests/docs pass |
| Pushes | push at tracker-defined push node after commit group is coherent |
| Failed experiment | document, mark `abandoned` or `experimental`, and continue to next unblocked task |

### Autonomous Work Loop

Repeat this loop until the day ends or all unblocked tasks are complete:

1. `git status --short --branch`
2. Read the next pending order in `Dependency Order`.
3. Search the repo for existing helpers before creating new code.
4. If external knowledge matters, browse primary sources and update the tracker with source labels.
5. Implement the smallest complete milestone that satisfies the current completion test.
6. Run unit/smoke/safety checks.
7. Update docs/tracker/result files.
8. Commit at the defined commit node.
9. Push if the current milestone is a push node.
10. Continue to the next unblocked order.

### Long-Running Job Protocol

For training, batch evaluation, large grids, or downloads:

- write stdout/stderr logs under `logs/` or the ignored output run directory;
- record the exact command in the relevant result doc;
- write a manifest with input files, parameters, seed, git commit, and output paths;
- prefer resumable/cache-aware scripts;
- after a crash or reboot, inspect logs/manifests before restarting;
- if a long job fails, document the failure and run a smaller smoke test to isolate the issue.

### Self-Review Cadence

At least once per milestone, and roughly every 90 minutes during a long autonomous run, do a short self-review:

```text
What order am I on?
Did I accidentally skip a dependency?
Did I rely on an unverified claim?
Did I protect .env/data/weights/outputs from git?
Did I run the completion test?
Should this be committed or pushed now?
What, if anything, is truly manual?
```

Record meaningful answers in the relevant result doc or tracker status update.

### End-Of-Day Report Requirements

At the end of a full autonomous run, report:

- current branch and git status;
- commits and pushes made;
- tracker orders completed, partial, blocked, manual;
- tests/evals run and whether they passed;
- output contact sheets/manifests generated;
- research sources newly used;
- critical failures or abandoned experiments;
- only the remaining true manual actions for the user.

### Status Legend

| Status | Meaning |
|--------|---------|
| done | Implemented and verified on this Windows machine |
| partial | Some code/data exists, but the completion test does not fully pass |
| pending | Not started or not verified |
| blocked | Cannot proceed until a dependency is resolved |
| manual | Requires Mac/user/account/GUI action |
| experimental | Implemented only as a research path; not allowed as production default |
| abandoned | Tried and rejected; keep records but do not build on it without a new reason |

### Source And Hallucination Control Protocol

Use this protocol before adding or changing any research claim in this tracker:

1. Prefer primary sources: papers, official project pages, official docs, or project repos.
2. If a claim is not directly stated in the source, label it as `engineering inference`.
3. If a cited method has not been run locally, label the implementation status as `pending` or `experimental`, never `done`.
4. If a source is inaccessible, do not rely on it; either find another primary source or mark the claim as `unverified`.
5. Do not cite user-provided prose as fact unless it is independently confirmed or clearly marked as a hypothesis.
6. Re-check recency-sensitive model/tool claims before implementation.

### Evidence Labels

| Label | Meaning |
|-------|---------|
| source-verified | Directly supported by a checked source |
| locally-verified | Verified by code or output in this repo on this machine |
| engineering-inference | Reasonable conclusion from source + project constraints, but not directly claimed by a source |
| unverified | Do not implement as fact until checked |

### Machine Role

Windows remains the CUDA implementation and experiment machine. Mac remains the development/review machine.

```text
Mac reviews / develops -> GitHub -> Windows pulls / runs -> Windows pushes code docs -> Mac validates visuals
```

Known Windows workspace:

```text
C:\Users\hhvrf\Documents\neuro_film
```

Manual Mac actions still live in `docs/WINDOWS_TASK_TRACKER.md` and `docs/REMOTE_ACCESS_MANUAL_STEPS.md`.

## Source-Checked Research Notes

The planning below is based on project experiments plus primary sources checked on 2026-05-27.

### Diffusion Control And Why It Is Not Enough For The Default

- ControlNet adds spatial conditioning to pretrained text-to-image diffusion models and supports edge/depth/segmentation/pose controls, but it is still a generative image model, not a mathematical lock on original pixels.
  Source: <https://arxiv.org/abs/2302.05543>
- T2I-Adapter aligns external control signals such as color and structure with frozen T2I models, but it also remains generation-control rather than a color-only renderer.
  Source: <https://arxiv.org/abs/2302.08453>
- Palette shows conditional diffusion can solve image-to-image tasks including colorization, inpainting, uncropping, and JPEG restoration. It supports chroma/color-layer research, but does not by itself guarantee film-style content preservation.
  Source: <https://arxiv.org/abs/2111.05826>
- DDColor is a Transformer-based colorization route that supports the idea of semantic color prediction, but black-and-white colorization is not the same problem as film color rendering.
  Source: <https://arxiv.org/abs/2212.11613>
- ColorEdit and ColorCtrl show the research field is moving toward more precise color editing inside diffusion attention/value mechanisms, but these are still better treated as research tracks rather than the default film-rendering product.
  Sources: <https://arxiv.org/abs/2411.10232>, <https://arxiv.org/abs/2508.09131>
- Plug-and-Play Diffusion Features can preserve semantic layout through feature injection in text-driven image-to-image translation, but it is still full-image diffusion and should only be used as a color-suggestion experiment.
  Source: <https://arxiv.org/abs/2211.12572>

### Neural LUT / AI Color Rendering Sources

- Image-Adaptive 3D LUT learns basis LUTs and a small CNN that predicts content-dependent weights, enabling fast high-resolution color/tone manipulation.
  Source: <https://arxiv.org/abs/2009.14468>
- AdaInt improves 3D LUT expressiveness by learning adaptive non-uniform sampling intervals.
  Source: <https://arxiv.org/abs/2204.13983>
- SepLUT separates component-independent and component-correlated transforms through 1D/3D LUT decomposition.
  Source: <https://arxiv.org/abs/2207.08351>
- NILUT represents color transforms as conditional neural implicit 3D LUTs.
  Source: <https://arxiv.org/abs/2306.11920>

### Film Artifact Layer Sources

- Film grain synthesis has deep-learning support; the IEEE TIP 2023 work uses an encoder-decoder for removal and a cGAN for controllable grain synthesis.
  Source: <https://pubmed.ncbi.nlm.nih.gov/37647187/>
- Analogue film damage simulation has a strong statistical foundation for scratches, dust, hairs, and damage masks, with 4K damaged/restored scan pairs and perceptual validation.
  Sources: <https://arxiv.org/abs/2302.10004>, <https://daniela997.github.io/FilmDamageSimulator/>
- CineStill describes rem-jet as reducing halation and describes halation as bright light reflection causing red-sensitive-layer glow in overexposed highlights. This supports highlight-bound, red/orange, alpha-layer halation constraints.
  Source: <https://cinestillfilm.com/pages/frequently-asked-questions>
- Dehancer's halation discussion supports the same practical model: halation appears around overexposed high-contrast boundaries and is tied to film-layer scattering.
  Source: <https://blog.dehancer.com/articles/halation/>
- CNN digital-to-film style transfer has been reported to learn color and some grain but not halation, making halation-layer modeling a high-value research target.
  Source: <https://arxiv.org/abs/2411.15967>
- Transparent/layered diffusion is relevant to future RGBA film-effect layers:
  - LayerDiffuse: <https://arxiv.org/abs/2402.17113>
  - Trans-Adapter: <https://arxiv.org/abs/2508.01098>
  - Layered Diffusion Brushes: <https://openaccess.thecvf.com/content/ICCV2025/html/Gholami_Streamlining_Image_Editing_with_Layered_Diffusion_Brushes_ICCV_2025_paper.html>
- Frequency-preservation editing research supports the need for high-frequency safety metrics and residual/layer constraints, but it does not directly solve film rendering:
  - FreqEdit: <https://arxiv.org/html/2512.01755v2>
  - FlexiEdit: <https://arxiv.org/html/2407.17850v1>

### Source Verification Matrix

| Claim Used In This Tracker | Evidence Label | Checked Source | Implementation Consequence |
|----------------------------|----------------|----------------|----------------------------|
| ControlNet/T2I-Adapter improve conditioning but are still generative image controls | source-verified + engineering-inference | ControlNet, T2I-Adapter papers | keep as experiment only, not default |
| Diffusion can be used for colorization/image-to-image tasks | source-verified | Palette paper | motivates chroma-only research, not production proof |
| Transformer colorization can predict plausible chroma semantically | source-verified | DDColor paper | motivates chroma/residual model |
| Image-adaptive LUTs can learn efficient color/tone transforms | source-verified | Image-Adaptive 3D LUT paper | prioritize Neural LUT as Part 2A |
| Adaptive/non-uniform LUT sampling improves expressiveness | source-verified | AdaInt paper | optional upgrade after basic LUT MVP |
| Separating 1D and 3D LUT transforms can be useful | source-verified | SepLUT paper | optional architecture variant |
| Neural implicit LUTs can represent conditional color transforms | source-verified | NILUT paper | research variant, not MVP |
| Film grain can be synthesized with controllable deep generative methods | source-verified | IEEE TIP/PubMed film grain paper | Part 3 AI grain is plausible |
| Film scratches/dust/hairs can be statistically simulated from real damage | source-verified | FilmDamageSimulator paper/project | Part 3 scratch/dust statistical baseline |
| Halation should be highlight-bound and red/orange biased for CineStill-like behavior | source-verified + engineering-inference | CineStill FAQ, Dehancer article | deterministic halation constraints |
| CNN digital-to-film work learned color/grain but not halation | source-verified | `CNNs for Style Transfer of Digital to Film Photography` | halation is high-value research target |
| Transparent/layered diffusion can support future RGBA layer work | source-verified + engineering-inference | LayerDiffuse, Trans-Adapter, Layered Diffusion Brushes | Part 3 AI layer generators remain experimental |
| High-frequency preservation matters for identity/detail stability | source-verified + engineering-inference | FreqEdit, FlexiEdit | evaluator must measure high-frequency luminance/detail retention |

Unverified or deliberately excluded claims:

- No claim is made that any cited diffusion/color-editing method directly solves Kodak/Fuji film emulation.
- No claim is made that any cited layer-diffusion method already generates film-specific halation/grain layers.
- No claim is made that Neural LUT will outperform the deterministic baseline before local experiments.

## Branch, Commit, And Push Policy

### Current Starting Point

Current branch before this tracker was created:

```text
codex/windows-baseline-handoff-20260527
```

Known local commits not yet pushed at the time this tracker is written:

```text
fdaa03f windows: add raw preview handoff and gamut-safe baseline
0078367 docs: organize project structure
dbc3673 docs: add content-preserving film task tracker
```

Before beginning implementation, push this branch or intentionally create a new branch from it.

### Global Git Rules

- Use small commits at completed safety milestones.
- Push after each milestone group, not after every tiny edit.
- Never include ignored assets in commits unless the user explicitly asks.
- Commit messages should name the mission and checkpoint, for example:
  - `color: add non-clipping safety evaluator`
  - `color: stabilize lab baseline gamut compression`
  - `research: add neural lut experiment scaffold`
  - `filmfx: add layer compositor and deterministic grain`

### Branch Nodes

| Node | Branch | Purpose | Start Condition | Merge Condition |
|------|--------|---------|-----------------|-----------------|
| B0 | `codex/windows-baseline-handoff-20260527` | Current handoff branch | Already active | Push current commits and tracker |
| B1 | `feat/color-baseline-stability` | Mission 1: harden current renderer | B0 pushed | all Part 1 gates pass |
| B2 | `research/ai-color-rendering` | Mission 2: Neural LUT and chroma-only AI color | Part 1 evaluator exists | research report plus MVP/prototype gates |
| B3 | `research/film-fx-layers` | Mission 3: film artifact layers | Part 1 evaluator exists | layer compositor and at least deterministic grain/halation gates pass |
| B4 | `integration/content-preserving-renderer` | Integrate approved parts | B1 plus selected B2/B3 commits | full regression and user visual approval |

### Push Nodes

| Push Node | When | Required Contents |
|-----------|------|-------------------|
| P0 | Immediately after creating this tracker | tracker doc and any docs-only organization changes |
| P1.1 | After Part 1 evaluator and benchmark set | safety metrics scripts, fixture manifests, docs |
| P1.2 | After non-clipping renderer changes | renderer code, tests, updated results doc |
| P1.3 | After Part 1 final review | contact sheets, manifest references, no generated images committed |
| P2.1 | After Neural LUT scaffold | differentiable LUT module, tiny smoke test, docs |
| P2.2 | After Neural LUT MVP result | experiment report, metrics, failure/success decision |
| P2.3 | After chroma-only prototype decision | prototype/report, not necessarily production merge |
| P3.1 | After layer data model and compositor | layer schema, compositor code, tests |
| P3.2 | After deterministic grain/halation | controllable layers, safety metrics |
| P3.3 | After AI artifact prototype decision | report and prototype artifacts |
| P4 | Integration branch | approved code paths plus final task report |

## Feature Registry

This registry defines what exists, what is planned, and what must remain experimental.

| Feature | Category | Current Status | Production Default? | Main Risk | Required Gate |
|---------|----------|----------------|---------------------|-----------|---------------|
| Deterministic Lab stats baseline | color renderer | partial | yes, after Part 1 | clipping/banding/overstrong chroma | Part 1 final gate |
| Gamut-safe Lab compression | color safety | partial | yes | hue shift or desaturation | no-clipping + visual review |
| Output headroom/margin | color safety | partial in raw preview script | yes | JPEG can reintroduce edge values | PNG validation min/max |
| Render safety evaluator | evaluator | pending | yes | false confidence if metrics too weak | calibrated identity baseline |
| Neural LUT | AI color renderer | pending | maybe | may become generic filter | Part 2A decision gate |
| Local/Semantic LUT maps | AI color renderer | pending | maybe | color halos/bleeding | edge-aware map gate |
| Chroma residual model | AI color renderer | pending | maybe | chroma bleeding | fixed-L and gamut gate |
| Chroma-only diffusion | research | pending | no | unstable training, bleeding | research report only |
| Diffusion chroma suggestion | research | pending | no | dirty/generated chroma | archive unless clear win |
| Layer compositor | filmfx foundation | pending | yes | bad blend math/clipping | layer bounds + no-clipping |
| Deterministic grain | filmfx | pending | maybe | ugly/noisy texture | spectrum + visual gate |
| AI grain residual | filmfx research | pending | no until proven | semantic texture contamination | residual-only gate |
| Deterministic halation | filmfx | pending | maybe | fake glow everywhere | highlight-bound gate |
| AI halation RGBA | filmfx research | pending | no until proven | hallucinated glow/content damage | RGBA-only + physical losses |
| Scratch/dust alpha overlays | filmfx | pending | optional | distracting damage | sparse alpha gate |

## Global Execution Order

Do not skip the evaluator. The whole project depends on objective safety gates.

```text
0. Push current handoff branch and this tracker.
1. Build evaluation harness and visual benchmark set.
2. Harden current deterministic color renderer.
3. Add richer but safe color controls.
4. Freeze Part 1 as the production baseline.
5. Explore AI color transforms: Neural LUT first, chroma-only diffusion second.
6. Explore artifact layers: deterministic compositor first, AI layer generators later.
7. Integrate only paths that pass safety gates.
```

## Dependency Order

This is the execution contract. Work from top to bottom; skip only blocked/manual items and continue with the next unblocked item.

| Order | Task | Depends On | Status | Completion Test | Commit Node |
|:---:|------|------------|:---:|-----------------|-------------|
| 0 | Push current handoff branch | Git access | done | `git push` returned `Everything up-to-date` on 2026-05-27 | P0; do this without asking if git remote works |
| 1 | Create `feat/color-baseline-stability` | Order 0 | done | branch created from `codex/windows-baseline-handoff-20260527` | branch node B1 |
| 2 | Define eval source buckets | current rawpixls manifests, no committed images | done | `configs/eval_buckets.yaml` exists; `scripts/list_eval_sources.py --require-existing` passes for seed set | `color: define evaluation source buckets` |
| 3 | Add safety evaluator | Order 2 | done | identity pair reports zero new clipping and `L_ssim=1.0` | `color: add render safety evaluator` |
| 4 | Audit current baseline | Order 3 | done | `docs/COLOR_BASELINE_STABILITY_RESULTS.md` has baseline table and ignored metrics exist under `outputs/eval/baseline_current/` | `color: audit current baseline stability` |
| 5 | Implement no-clipping renderer improvements | Order 4 | done | full 20-image x 8-style safe pass has zero new `0/255` and output bounds `4..251` | `color: enforce non-clipping output bounds` |
| 6 | Add artifact/banding guards | Order 5 | done | banding, high-frequency, neutral/skin contamination, and guarded contact sheets appear in report | `color: add artifact and banding guards` |
| 7 | Tune safe-rich profiles | Order 6 | done | color-stock safe-rich profile passes no-clip and L-SSIM gates on seed set; B&W marked for separate gate | `color: tune rich natural film profiles` |
| 8 | Add production preset/regression tests | Order 7 | done | `--preset safe-rich`, batch eval smoke, and pytest smoke pass | `color: add safe-rich production preset` |
| 9 | Freeze Part 1 verdict | Order 8 | done | Part 1 final gate table completed; human visual approval remains manual | `color: record stable baseline verdict` |
| 10 | Branch `research/ai-color-rendering` | Orders 3 and 5 | pending | branch exists | branch node B2 |
| 11 | Neural LUT scaffold | Order 10 | pending | differentiable LUT smoke test passes | P2.1 |
| 12 | Neural LUT MVP | Order 11 | pending | result doc compares vs Part 1 | P2.2 |
| 13 | Chroma residual/chroma diffusion research | Orders 11-12 | pending | research doc decides promote/archive | P2.3 |
| 14 | Branch `research/film-fx-layers` | Orders 3 and 5 | pending | branch exists | branch node B3 |
| 15 | Layer schema/compositor | Order 14 | pending | layer compositor smoke test passes | P3.1 |
| 16 | Deterministic grain/halation/scratch layers | Order 15 | pending | layer contact sheets and safety metrics exist | P3.2 |
| 17 | AI artifact layer prototypes | Orders 15-16 | pending | residual/RGBA-only reports exist | P3.3 |
| 18 | Integration branch | approved outputs from B1/B2/B3 | pending | `render_film.py` final CLI runs | P4 |
| 19 | Mac/user visual validation | Order 18 | manual | user approves contact sheets | manual |

Parallel-safe tasks:

- Orders 10-13 and 14-17 can proceed in parallel after Order 5, but neither may merge into integration before Part 1 gates pass.
- Literature review/memos can proceed anytime, but implementation claims must still follow source verification.
- Manual Mac validation can run whenever contact sheets exist.

## Shared Evaluation Harness

This is a dependency for all three missions.

### E0: Benchmark Image Sets

| ID | Task | Status | Output |
|----|------|--------|--------|
| E0.1 | Define `data/eval_sources/` manifest schema without committing images | done | `configs/eval_sources.schema.json` plus `docs/EVAL_SOURCE_BUCKETS.md` |
| E0.2 | Reuse final raw.pixls.us RAW-rendered 20-image set as a seed benchmark | done | `configs/eval_buckets.yaml` references ignored manifest only |
| E0.3 | Add source buckets: skin, sky, foliage, night/tungsten, snow/high-key, deep shadow, saturated objects, text/logo, face, fine detail | done | `configs/eval_buckets.yaml` |
| E0.4 | Add manual slot for user/Mac photos without committing them | done | ignored `data/eval_sources/private_manual/` slot documented |

Completion test:

```powershell
python scripts/list_eval_sources.py --manifest configs/eval_buckets.yaml
```

Commit node: `color: define evaluation source buckets`

### E1: Safety Metrics

| ID | Metric | Reason | Gate |
|----|--------|--------|------|
| E1.1 | output min/max and clipped pixel percentage | hard no-clipping requirement | no new `0/255`; recommended output bounds `[4, 251]` for PNG |
| E1.2 | L-channel SSIM | structure lock | `L_ssim >= 0.995` for color-only paths |
| E1.3 | luminance gradient/edge delta | edge preservation | no large edge movement; threshold calibrated on identity transform |
| E1.4 | high-frequency luminance residual | smearing detection | residual below threshold, no low-pass collapse |
| E1.5 | posterization/banding score | artifact detection | monotonic low banding score vs input |
| E1.6 | hue/chroma outlier map | dirty color detection | no large skin/neutral contamination |
| E1.7 | contact sheet and diff maps | human review | always generated |

Output scripts:

```text
scripts/evaluate_render_safety.py
scripts/make_eval_contact_sheet.py
```

Outputs:

```text
outputs/eval/<run_id>/metrics.json
outputs/eval/<run_id>/manifest.csv
outputs/eval/<run_id>/contact_sheet.png
outputs/eval/<run_id>/diff_maps/
```

Commit node: `color: add render safety evaluator`

Push node: P1.1

## Part 1: Improve Current Latest Scheme

Mission: make the deterministic color baseline precise, stable, non-clipping, artifact-resistant, natural, and richly colored.

Recommended branch:

```powershell
git switch -c feat/color-baseline-stability
```

### Part 1 Principles

- Current deterministic Lab baseline is the production baseline.
- It can be modified, but not replaced by diffusion.
- Rich color must come from safe color science:
  - perceptual gamut compression,
  - chroma curves,
  - tone roll-off,
  - robust statistics,
  - optional local adaptation.
- If a color transform would clip, compress or reject it.

### 1.0 Baseline Audit

| ID | Task | Status | Completion Test |
|----|------|--------|-----------------|
| 1.0.1 | Run current `pipeline_color_baseline.py` on eval set for each style | done | `outputs/eval/baseline_current/*/metrics.json` exists |
| 1.0.2 | Record per-style clipping, banding, colorfulness, L-SSIM | done | `docs/COLOR_BASELINE_STABILITY_RESULTS.md` has audit table |
| 1.0.3 | Identify worst styles/images | done | failure summary recorded; worst recurring images include seed IDs 09 and 11 across styles |
| 1.0.4 | Freeze a "do not regress" fixture set | done | `configs/eval_regression_fixtures.json` references source IDs only |

Commit node: `color: audit current baseline stability`

### 1.1 No-Clipping Renderer

| ID | Task | Status | Details |
|----|------|--------|---------|
| 1.1.1 | Promote gamut-safe path from ad-hoc flag to default-safe mode | done | `--gamut-safe` resolves to formal `source` gamut mode; production preset still pending |
| 1.1.2 | Add output margin option to baseline pipeline, not only raw preview script | done | `--output-margin 4` supported |
| 1.1.3 | Use PNG for no-clipping validation outputs | done | `--format png` and `.png` suffix save real PNG |
| 1.1.4 | Add perceptual gamut compression strategy variants | done | source binary-search and hue-preserving chroma compression modes exist |
| 1.1.5 | Add monotonic tone roll-off before output bounds | done | `--tone-rolloff` hook supported |
| 1.1.6 | Add hard failure mode if clipping exceeds gate | done | `--fail-on-clip` exits nonzero after saved-output validation |

Completion test:

```powershell
python scripts/pipeline_color_baseline.py input.jpg --style velvia_50 --strength 0.50 --luma-strength 0.25 --grain 0 --gamut-safe --output-margin 4 --output out.png
python scripts/evaluate_render_safety.py --before input.jpg --after out.png --fail-on-clip
```

Commit node: `color: enforce non-clipping output bounds`

### 1.2 Artifact And Banding Prevention

| ID | Task | Status | Details |
|----|------|--------|---------|
| 1.2.1 | Add banding/posterization metric | done | histogram gaps and low-bit quantization proxy in evaluator |
| 1.2.2 | Add optional 16-bit internal processing path | done | renderer keeps float32 until final export; optional dither reduces 8-bit quantization risk |
| 1.2.3 | Add smooth chroma compression | done | guardrail chroma caps ease into limit before gamut compression |
| 1.2.4 | Add neutral/skin protection masks as non-AI heuristics | done | `--use-guardrails` applies neutral and skin masks |
| 1.2.5 | Add per-style max chroma gain | done | `configs/color_guardrails.json` defines per-style caps |
| 1.2.6 | Add regression contact sheets for artifact-prone images | done | audit script writes per-style `contact_sheet.png` |

Commit node: `color: add artifact and banding guards`

### 1.3 Natural Yet Rich Color

| ID | Task | Status | Details |
|----|------|--------|---------|
| 1.3.1 | Build style-specific tone/chroma parameter table | done | `configs/color_rendering_profiles.yaml` |
| 1.3.2 | Add colorfulness target ranges per stock | done | safe-rich profile records high/medium/near-zero chroma through per-style strengths and guardrails |
| 1.3.3 | Implement saturation curve instead of linear saturation push | done | `--chroma-curve-strength` protects already-saturated colors |
| 1.3.4 | Implement highlight roll-off and shadow floor | done | `--shadow-floor-l` and `--highlight-ceiling-l` supported |
| 1.3.5 | Add optional local contrast preservation in L only | done | `--preserve-luma-detail` supported |
| 1.3.6 | Validate 8 stocks on raw.pixls/digital eval set | done | `outputs/eval/baseline_saferich/summary.json` and results doc updated |

Commit node: `color: tune rich natural film profiles`

### 1.4 Production CLI And Regression Suite

| ID | Task | Status | Details |
|----|------|--------|---------|
| 1.4.1 | Add production preset CLI | done | `--preset safe-rich` |
| 1.4.2 | Add batch eval CLI | done | `scripts/evaluate_color_pipeline.py` |
| 1.4.3 | Add pytest smoke tests for no-clipping fixtures | done | `tests/test_color_baseline_safety.py` |
| 1.4.4 | Add docs with recommended per-style settings | done | `docs/COLOR_BASELINE_STABILITY_RESULTS.md` |
| 1.4.5 | Generate final 20-image contact sheet per priority style | done | `outputs/eval/baseline_saferich/<style>/contact_sheet.png`, ignored |

Commit node: `color: add safe-rich production preset`

Push node: P1.2

### Part 1 Final Gate

Machine gate table:

| Gate | Status | Evidence |
|------|:---:|----------|
| 20-image rawpixls set: no output `0/255`, min/max within configured margin | done | `outputs/eval/baseline_saferich/summary.json`; all styles `4..251` |
| No severe banding in sky/skin/high-key samples | done | evaluator banding metrics and contact sheets generated |
| `L_ssim >= 0.995` for all color-only results | done | six color stocks pass; B&W excluded from color-only gate |
| Edge delta below threshold calibrated from identity transform | partial | gradient/high-frequency metrics recorded; threshold should be refined after more fixtures |
| Human visual contact sheet shows rich color without obvious artifact | manual | `outputs/eval/baseline_saferich/<style>/contact_sheet.png` needs user/Mac review |
| Docs updated | done | `docs/COLOR_BASELINE_STABILITY_RESULTS.md` and this tracker |

Part 1 machine verdict:

- Promote `safe_lab` + `--preset safe-rich` as the current default color-stock baseline.
- Keep HP5/Tri-X enabled but treat B&W tone as a separate visual/tone approval path.
- Do not promote diffusion/img2img as a default color renderer.

Commit node: `color: record stable baseline verdict`

Push node: P1.3

## Part 2: Explore Other Color Simulation Schemes

Mission: build AI color rendering paths that only predict color transforms, LUTs, chroma residuals, or color operators. Do not let AI generate final RGB as the default.

Recommended branch:

```powershell
git switch -c research/ai-color-rendering
```

Start condition:

- E0/E1 evaluator exists.
- Part 1 baseline has at least a working no-clipping mode.

### Part 2 Priority Order

```text
2A. Image-adaptive Neural LUT
2B. Local/Semantic LUT and tone-parameter predictor
2C. Chroma-only residual model
2D. Chroma-only diffusion
2E. Diffusion color suggestion as experiment only
```

### 2A: Image-Adaptive Neural LUT MVP

Rationale: closest engineering match to "AI predicts color transform; deterministic renderer applies it."

| ID | Task | Status | Details |
|----|------|--------|---------|
| 2A.1 | Implement differentiable 3D LUT apply op | pending | trilinear interpolation, CPU/CUDA if possible |
| 2A.2 | Implement basis LUT module | pending | identity plus learnable basis LUTs |
| 2A.3 | Implement small image encoder | pending | CNN/ConvNeXt-lite; low-res preview input |
| 2A.4 | Predict basis weights per style | pending | style embedding plus image features |
| 2A.5 | Train to imitate Part 1 stable renderer first | pending | supervised pseudo-target from safe renderer |
| 2A.6 | Add unpaired film-stat/style loss only after imitation works | pending | histogram/CLIP/DINO optional, never first |
| 2A.7 | Export LUT or LUT weights per image | pending | reproducible color transform artifact |
| 2A.8 | Evaluate against Part 1 safety gates | pending | must not reduce structure metrics |

Proposed files:

```text
src/models/color_lut/
scripts/train_neural_lut.py
scripts/eval_neural_lut.py
configs/model/neural_lut.yaml
docs/NEURAL_LUT_RESULTS.md
```

Commit nodes:

- `research: add differentiable lut operator`
- `research: add neural lut training scaffold`
- `research: record neural lut mvp results`

Push nodes:

- P2.1 after scaffold and smoke tests.
- P2.2 after MVP results.

Decision gate:

- If Neural LUT only imitates Part 1 without improvement, keep as research but do not promote.
- Promote only if it improves naturalness/richness while passing all Part 1 safety gates.

### 2B: Local/Semantic Color Parameter Predictor

Rationale: global LUT may be too simple for film behavior. This route predicts bounded local color controls while renderer remains deterministic.

| ID | Task | Status | Details |
|----|------|--------|---------|
| 2B.1 | Define local adjustment map format | pending | low-res maps upsampled edge-aware |
| 2B.2 | Add bounded maps: chroma gain, hue shift, shadow bias, highlight warmth | pending | numeric limits per style |
| 2B.3 | Add optional semantic hints | pending | skin/sky/foliage masks; heuristic first, ML later |
| 2B.4 | Train/predict maps from encoder | pending | low-resolution output only |
| 2B.5 | Add edge-aware smoothing | pending | no halos or color bleeding |
| 2B.6 | Evaluate skin/sky/foliage cases | pending | dedicated contact sheets |

Commit node: `research: add bounded local color maps`

### 2C: Chroma Residual Model

Rationale: closest "separate layer" color model without full diffusion.

Architecture:

```text
Input RGB -> Lab
Model sees low-res image/style
Model predicts bounded delta_a, delta_b
L_out = L_in
a_out = a_in + bounded_delta_a
b_out = b_in + bounded_delta_b
gamut compression -> output
```

| ID | Task | Status | Details |
|----|------|--------|---------|
| 2C.1 | Add Lab residual renderer | pending | no learned model yet |
| 2C.2 | Add bounded residual map schema | pending | max delta per style |
| 2C.3 | Train small U-Net/Transformer residual predictor to imitate Part 1 | pending | supervised first |
| 2C.4 | Add losses: chroma reconstruction, gamut penalty, smoothness | pending | no L reconstruction because L is fixed |
| 2C.5 | Evaluate color bleeding and edge alignment | pending | use chroma edge maps |

Commit node: `research: add chroma residual prototype`

### 2D: Chroma-Only Diffusion

Rationale: advanced research path. Diffusion is allowed only in chroma/residual space.

Architecture:

```text
Input RGB -> Lab
Freeze L
Noise a/b or delta_a/delta_b only
Diffusion denoises chroma residual
Output Lab = [L_in, a_in + delta_a, b_in + delta_b]
Gamut-safe compositor
```

| ID | Task | Status | Details |
|----|------|--------|---------|
| 2D.1 | Write feasibility memo and VRAM estimate | pending | before coding |
| 2D.2 | Build tiny synthetic dataset from Part 1 renderer targets | pending | no real target requirement at first |
| 2D.3 | Train 64/128px toy chroma diffusion | pending | prove interface only |
| 2D.4 | Add structure losses even though L is frozen | pending | edge/chroma bleeding checks |
| 2D.5 | Upscale/guide chroma maps with edge-aware interpolation | pending | avoid full-res diffusion cost |
| 2D.6 | Compare against Neural LUT and Part 1 | pending | promote only if clearly better |

Commit node: `research: test chroma-only diffusion prototype`

Push node: P2.3

### 2E: Diffusion Color Suggestion Only

This is an experiment line, not a product line.

Allowed pattern:

```text
generated = diffusion(input, style_prompt)
generated_lab = RGB2Lab(generated)
input_lab = RGB2Lab(input)
final_lab = [input_L, generated_a, generated_b]
```

Rules:

- Never promote raw generated RGB.
- Always restore original L.
- Always run gamut-safe output.
- Always compare to Part 1.
- If dirty color/bleeding appears, archive the experiment.

Commit node: `research: evaluate diffusion chroma suggestion`

## Part 3: Explore Film Artifact Layer Simulation

Mission: model film side effects as separate layers. AI may generate grain, halation, dust, scratches, leaks, or bloom layers, but must not rewrite the base image.

Recommended branch:

```powershell
git switch -c research/film-fx-layers
```

Start condition:

- E0/E1 evaluator exists.
- Part 1 baseline has a no-clipping mode.

### Part 3 Layer Contract

Every film effect must be represented as one of these:

```text
RGBA layer:
  rgb: visual effect color
  alpha: bounded opacity

Residual layer:
  residual_rgb_or_luma: bounded signed residual
  mask_or_strength: bounded map

Parameter-only layer:
  deterministic effect parameters + seed
```

The final image is produced only by deterministic compositing:

```text
final = composite(base_color_render, effect_layers)
```

### 3.0 Layer Compositor Foundation

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.0.1 | Define layer schema | pending | JSON manifest for layers |
| 3.0.2 | Implement compositor | pending | alpha, screen/additive, residual, soft-light modes |
| 3.0.3 | Add layer visualizer | pending | export each layer separately |
| 3.0.4 | Add safety metrics for layers | pending | alpha bounds, residual bounds, affected area |
| 3.0.5 | Add CLI | pending | `scripts/pipeline_filmfx_layers.py` |

Proposed files:

```text
src/filmfx/layers.py
src/filmfx/compositor.py
scripts/pipeline_filmfx_layers.py
configs/filmfx_profiles.yaml
docs/FILMFX_LAYER_RESULTS.md
```

Commit node: `filmfx: add layer schema and compositor`

Push node: P3.1

### 3.1 Deterministic Grain Layer

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.1.1 | Implement zero-mean high-frequency grain residual | pending | seed-controlled |
| 3.1.2 | Modulate grain by luminance and film stock | pending | shadows and midtones configurable |
| 3.1.3 | Add channel-correlated and channel-independent variants | pending | color vs B&W grain |
| 3.1.4 | Add power spectrum metric | pending | compare realism over patches |
| 3.1.5 | Add contact sheets by ISO/strength | pending | visual review |

Constraints:

- zero or near-zero mean,
- bounded amplitude,
- high-frequency dominant,
- no edge movement,
- no low-frequency color cast unless explicitly a separate layer.

Commit node: `filmfx: add deterministic grain residual`

### 3.2 AI Grain Generator

Rationale: easiest AI artifact layer because grain is high-frequency and non-semantic.

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.2.1 | Review implementation options: cGAN vs tiny diffusion vs procedural neural noise | pending | write memo first |
| 3.2.2 | Build training patches from grainy/clean or pseudo pairs | pending | do not commit data |
| 3.2.3 | Train small conditional generator | pending | condition on luminance, ISO, style, seed/noise |
| 3.2.4 | Output residual only | pending | not final image |
| 3.2.5 | Compare to deterministic grain | pending | spectrum and visual preference |

Commit node: `filmfx: prototype conditional grain generator`

### 3.3 Deterministic Halation Layer

Rationale: halation is hard and must be physically constrained before AI is allowed.

Inputs:

```text
highlight mask
overexposure map
edge/gradient map
local contrast map
film stock halation profile
strength
```

Output:

```text
halation_rgba
```

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.3.1 | Implement highlight mask with soft threshold | pending | linear RGB or Lab L input |
| 3.3.2 | Restrict halation to high-contrast highlight boundaries | pending | no glow in unsupported areas |
| 3.3.3 | Add red/orange spectral profile | pending | stock-specific |
| 3.3.4 | Add multi-radius blur/scatter | pending | small core plus broad falloff |
| 3.3.5 | Add alpha cap and affected-area cap | pending | safety gate |
| 3.3.6 | Validate on night lights, windows, snow, specular highlights | pending | contact sheets |

Commit node: `filmfx: add constrained halation layer`

### 3.4 AI Halation Layer Generator

Rationale: high-value research path. Must not start until deterministic halation exists and is measurable.

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.4.1 | Write halation data strategy memo | pending | paired data, pseudo labels, residual extraction |
| 3.4.2 | Generate pseudo-halation targets from deterministic simulator | pending | train first to imitate constraints |
| 3.4.3 | Train layer generator to output RGBA only | pending | no final RGB |
| 3.4.4 | Condition on highlight/edge/luminance/style | pending | not prompt-only |
| 3.4.5 | Add physical losses | pending | only near highlights, red/orange dominance, smoothness |
| 3.4.6 | Compare to deterministic halation | pending | only promote if visually better and safe |

Commit node: `filmfx: prototype halation layer generator`

### 3.5 Scratch, Dust, Hair, Damage Layers

Rationale: can be modeled as sparse alpha overlays and should not need to understand scene semantics deeply.

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.5.1 | Implement procedural sparse dust/scratch alpha masks | pending | statistical baseline |
| 3.5.2 | Add avoid-face/avoid-subject optional mask slot | pending | manual/heuristic first |
| 3.5.3 | Add style controls: clean scan, archival, damaged, dusty | pending | user-facing |
| 3.5.4 | Explore FilmDamageSimulator-like statistics | pending | source-compatible implementation only |
| 3.5.5 | Optional AI transparent layer generator | pending | only if procedural baseline insufficient |

Commit node: `filmfx: add dust and scratch layers`

### 3.6 Bloom, Light Leak, Gate Dirt

| ID | Task | Status | Details |
|----|------|--------|---------|
| 3.6.1 | Implement deterministic bloom separate from halation | pending | white/warm glow, not red film-base halation |
| 3.6.2 | Implement optional light leak overlay | pending | semi-transparent, edge-origin, seed-controlled |
| 3.6.3 | Implement gate dirt/dust overlays | pending | sparse, controllable |
| 3.6.4 | Ensure all layers are separately exportable | pending | visual debug |

Commit node: `filmfx: add optional overlay effect layers`

### Part 3 Final Gate

All effect layers must pass:

- Base image structure unchanged before compositing.
- Each layer exported independently.
- Residual/alpha bounds documented.
- Halation appears only near supported highlights.
- Grain is bounded and high-frequency.
- Scratch/dust are sparse and optional.
- User can disable every layer independently.
- Contact sheets show layer-only and final composite views.

Push node: P3.2 and P3.3 depending on deterministic vs AI milestones.

## Integration Plan

Recommended integration branch:

```powershell
git switch -c integration/content-preserving-renderer
```

### Integration Order

1. Merge Part 1 stable renderer.
2. Add filmfx deterministic compositor only if safety tests pass.
3. Add Neural LUT only if it improves or matches Part 1 safety.
4. Do not merge chroma-only diffusion unless it has a clear win.
5. Do not merge AI artifact generators unless layer contract and safety gates pass.

### Final CLI Target

```powershell
python scripts/render_film.py input.jpg `
  --style velvia_50 `
  --color-engine safe_lab `
  --preset safe-rich `
  --grain 0.25 `
  --halation 0.15 `
  --output output.png `
  --write-layers `
  --write-metrics
```

Possible color engines:

```text
safe_lab
neural_lut
local_lut
chroma_residual
chroma_diffusion_experimental
diffusion_chroma_suggestion_experimental
```

Possible filmfx engines:

```text
none
deterministic
ai_grain_experimental
ai_halation_experimental
```

### Integration Commit Nodes

| Commit Node | Contents |
|-------------|----------|
| I1 | `render_film.py` wrapper and stable safe_lab path |
| I2 | layer compositor integrated with stable renderer |
| I3 | optional Neural LUT engine behind explicit flag |
| I4 | final docs, metrics, and tracker status update |

Push node: P4

## Documentation Requirements

Each mission must produce a result doc:

```text
docs/COLOR_BASELINE_STABILITY_RESULTS.md
docs/NEURAL_LUT_RESULTS.md
docs/CHROMA_ONLY_RESEARCH_RESULTS.md
docs/FILMFX_LAYER_RESULTS.md
```

Each result doc must include:

- exact command lines,
- source image manifest path,
- parameter grid,
- metrics summary,
- contact sheet paths,
- decision: promote, keep experimental, or abandon,
- known failure cases,
- next action.

## Manual-Only Items

These should remain manual unless the user gives interactive access:

- Mac visual approval of contact sheets.
- Mac no-password SSH validation.
- Tailscale install/sign-in on Windows and Mac.
- Using private user photos for validation.
- Any licensing decision for publishing LoRA/data-derived assets.

## Current First Action

The next agent should:

1. Commit this tracker revision if it is still uncommitted.
2. Push the current branch with the existing handoff, structure, tracker, and tracker-revision commits; this is an explicit tracker-approved push node, so do not ask the user again if GitHub auth works.
3. Create branch `feat/color-baseline-stability`.
4. Implement E0/E1 evaluator before changing renderer behavior.

Suggested immediate commands:

```powershell
git status --short --branch
git add docs/CONTENT_PRESERVING_FILM_TASK_TRACKER.md AGENTS.md
git commit -m "docs: refine content-preserving tracker with source gates"
git push
git switch -c feat/color-baseline-stability
```
