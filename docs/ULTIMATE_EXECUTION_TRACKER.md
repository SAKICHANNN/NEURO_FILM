# K-MCFM Ultimate Execution Tracker

> Active planning authority from 2026-07-10.
> Strategy and research basis: `docs/planning/ULTIMATE_ROADMAP_2026.md`.
> This tracker records intended work; only rows marked `complete` are implemented facts.

---

## 1. Parent goal and task contract

**Parent goal `ULT`**: deliver a local-first, content-safe, color-managed, calibrated film-imaging product whose Reference path never rewrites geometry or identity and whose stock/process claims are supported by owned or cleared measurements.

### Definition of Ready

- current code/data/document truth is known;
- task has one owner and bounded files;
- dependencies and human-approval gates are explicit;
- baseline, hypothesis, metric, gate and stop condition are frozen before execution;
- train/validation/test and research/production data lanes cannot mix silently.

### Definition of Done

- implementation, tests and evidence report agree;
- failure and tail cases are preserved, not hand-picked away;
- affected parent/child/sibling/interface/docs nodes have been checked;
- data/model/profile provenance and hashes are recorded;
- relevant tests pass on the declared device matrix;
- scoped local commit exists; no push unless explicitly requested.

### Status vocabulary

| Status | Meaning |
|---|---|
| `complete` | Evidence and propagation are complete |
| `ready` | DoR satisfied; can be claimed next |
| `blocked` | External approval/input/dependency is missing |
| `pending` | Not ready yet |
| `research-only` | May be explored but cannot enter Reference/release |
| `retired` | Retained as history/fallback, not active architecture |

---

## 2. Architecture decision record

| Decision | State | Rationale | Revisit trigger |
|---|---|---|---|
| Deterministic full-resolution renderer is the Reference core | accepted for plan | Current diffusion paths changed detail/identity; color operators can guarantee no geometry resampling | Only if a new method passes all hard gates |
| Small model may predict bounded parameters | candidate | LUT/grid/curve prediction adds context without generating pixels | Real paired residual remains after simple baseline |
| Generative editing is isolated | accepted for plan | Useful for creative target/teacher, unsafe as identity-preserving renderer | Never merged into Reference without new architecture proof |
| Stock profile includes process and interpretation | accepted for plan | Negative/slide/B&W do not have one intrinsic display RGB look | None; schema invariant |
| Portra 400 + Velvia 50 are pilot stocks | proposed | Orthogonal negative/slide behaviors and high user value | Availability, rights or lab feasibility fails |
| FiveK is optional neutral auto-base only | accepted for plan | Expert retouch is not film identity; full local sources were deleted | Sources/rights restored and product evidence supports it |
| Community/Flickr/FilmSet assets are research-only by default | accepted for plan | Missing or limited rights; FilmSet is Capture One recipe target | Per-asset legal clearance |

---

## 3. DRPT node tree

```text
ULT  Ultimate calibrated film-imaging product
├── U0  Truth, rights and reproducibility reset
│   ├── U0.1 Reconcile active docs and archive obsolete claims
│   ├── U0.2 Decide repository license and third-party notice policy
│   ├── U0.3 Build data/artifact lineage and repair split leakage
│   └── U0.4 Establish CI, frozen benchmark and environment capture
├── U1  Color-managed high-precision foundation
│   ├── U1.1 Make WorkingImage the only render ingress
│   ├── U1.2 Implement color-state contract and fail-closed modes
│   ├── U1.3 Implement correct 8/16-bit/profile-aware export
│   └── U1.4 Tile/cache/reference backend and parity vectors
├── U2  Reference film renderer and profile system
│   ├── U2.1 Profile/recipe schema and provenance
│   ├── U2.2 Monotone sensitometry + global LUT renderer
│   ├── U2.3 Negative/slide/B&W interpretation interfaces
│   └── U2.4 Legacy safe_lab compatibility and migration
├── U3  Owned paired calibration
│   ├── U3.1 Capture protocol, rights and lab/scanner SOP
│   ├── U3.2 Portra 400 pilot
│   ├── U3.3 Velvia 50 pilot
│   └── U3.4 Frozen roll/lab/scanner holdout
├── U4  Evaluation V2
│   ├── U4.1 Content and geometry hard gates
│   ├── U4.2 Color/sensitometry authenticity scorecard
│   ├── U4.3 Grain/halation/MTF measurement
│   ├── U4.4 Blind human study protocol
│   └── U4.5 Performance/cross-platform benchmark
├── U5  Bounded-AI challenge
│   ├── U5.1 1D + 3D LUT baseline
│   ├── U5.2 SepLUT/NILUT challenge
│   ├── U5.3 Bilateral-grid challenge
│   └── U5.4 Optional masks/personalization
├── U6  Physical effects
│   ├── U6.1 Exposure-domain halation
│   ├── U6.2 Density-aware grain and MTF
│   ├── U6.3 Bloom and creative defects separation
│   └── U6.4 100MP/video consistency
├── U7  Productization
│   ├── U7.1 Stable engine API + CLI + batch
│   ├── U7.2 Desktop non-destructive workflow
│   ├── U7.3 Windows/Mac/CPU deployment
│   └── U7.4 Packaging, recovery, privacy and telemetry
└── U8  Release and expansion
    ├── U8.1 Beta and independent validation
    ├── U8.2 Legal/security/release cards and BOM
    ├── U8.3 Wave 2/3 stock expansion
    └── U8.4 Isolated tool-agent and Creative mode
```

Critical path:

```text
U0 → U1 → U2 → U3 → (U5 || U6) → U7 → U8
             ↘ U4 starts early and gates every later promotion
```

U4 is a continuous validation sibling, not an end-of-project QA phase.

---

## 4. Integrated work board

### U0 — Truth, rights and reproducibility reset

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U0.0 | complete | Ultimate research synthesis and execution tree | none | Roadmap + this tracker; 2026-07-10 |
| U0.1 | complete | Active docs agree on deterministic current baseline and retired diffusion path | U0.0 | AGENTS/README/IMPL_PLAN/TASK_BOARD and planning index agree; 2026-07-10 |
| U0.2 | blocked | Root `LICENSE`, NOTICE and public-claim decision | owner/legal choice | License file exists; dependencies/assets audited |
| U0.3 | ready | Manifest v2, research/production lanes, group split, cross-split duplicate audit | none | 100% lineage class; 0 unresolved cross-split near duplicates |
| U0.4 | ready | CI + lock capture + frozen benchmark registry | U0.1 | Clean clone tests; checksums; device/env report |
| U0.5 | ready | Correct stale FiveK/data reproduction state | U0.1 | Manifest no longer claims deleted full sources exist |

Restrictions:

- preserve untracked user files `halationguide.md` and `scripts/make_velvia50_scheme_comparison_sheets.py`;
- do not delete old experiments; mark them historical;
- do not push the current local-only branch without explicit approval;
- do not add an MIT license merely because old docs say MIT—the owner must confirm.

### U1 — Color-managed high-precision foundation

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U1.1 | pending | `render_film` consumes `WorkingImage` | U0.4 | JPEG/PNG/TIFF/RAW E2E fixtures; no early PIL 8-bit collapse |
| U1.2 | pending | `scene/display/unknown` state and Reference/Approximation policy | U1.1 | Unknown inputs warn/fail closed; metadata round-trip |
| U1.3 | pending | Correct TIFF/PNG/JPEG encoding, 8/16-bit and ICC | U1.1 | Extension=encoding; bit-depth/profile tests |
| U1.4 | pending | ACEScg or validated wide-gamut working contract | U1.2 | OCIO config/version pinned; golden transform vectors |
| U1.5 | pending | HEIF/HDR/gain-map detect/preserve or explicit rejection | U1.2 | Fixtures for supported/unsupported variants |
| U1.6 | pending | Halo-aware tile/cache renderer | U1.3 | Full-frame vs tiled tolerance; bounded memory |

Do not claim camera-accurate RAW solely from generic rawpy. Reference-grade camera paths require a known DNG/IDT/profile; generic development remains labeled.

### U2 — Reference renderer and profile system

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U2.1 | pending | Versioned profile + recipe JSON schema | U0.2, U1.2 | Schema tests, migrations, provenance card |
| U2.2 | pending | Monotone exposure/sensitometry curves | U1.4, U2.1 | Monotonicity/property tests; curve report |
| U2.3 | pending | Tetrahedral 3D LUT/global residual | U2.2 | Identity/gamut/interpolation tests |
| U2.4 | pending | Negative/slide/B&W interpretation plugin boundary | U2.1 | Three synthetic reference profiles and contracts |
| U2.5 | pending | Legacy `safe_lab` adapter | U2.1, U1.3 | Old recipes render within frozen tolerance |
| U2.6 | pending | Profile evidence labels | U2.1 | `heuristic/measured/paired/held-out` visible in CLI/API |

U2 can be implemented with synthetic/unit data. It cannot be promoted as an accurate named-stock renderer before U3/U4.

### U3 — Owned paired calibration

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U3.1 | blocked | Rights-cleared capture protocol and releases | owner budget/legal/lab approval | Signed allowed-use matrix and SOP |
| U3.2 | blocked | Scanner/lab repeatability pilot | U3.1 | Repeat scans/rolls quantify measurement floor |
| U3.3 | blocked | Portra 400 paired pilot | U3.2 | Charts + at least 3 rolls across 2 process sessions + 30–50 valid scenes minimum |
| U3.4 | blocked | Velvia 50 paired pilot | U3.2 | Same evidence; E-6/direct interpretation |
| U3.5 | pending | Freeze train/val/gold by group | U3.3, U3.4 | Hash manifest and access log before first fitting |
| U3.6 | pending | Scale to about 100 valid scenes/stock if powered | U3.5 | Pilot variance/effect-size decision, not arbitrary count |

Pilot capture minimum dimensions:

- charts + -3EV..+3EV sequence;
- D55, 3200K, representative LED and difficult mixed light;
- skin, foliage, sky, neon, highlights, deep shadow, text, fabric and architecture;
- raw digital master + raw/high-bit-depth film scan;
- film lot, exposure, process, lab, scanner, ICC, settings and repeat scan;
- scene/roll/lab/subject grouped split and retained failures.

Go/No-Go:

- if process/scan variance is larger than candidate-model differences, stop model scaling and repair measurement;
- if rights are incomplete, keep data internal or do not use it;
- if the two-stock pilot cannot distinguish stock from interpretation, repair schema/capture first.

### U4 — Evaluation V2

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U4.1 | ready | Frozen content/geometry hard-gate suite | U0.4 | face/text/edge/keypoint/tile/determinism report |
| U4.2 | pending | Color/sensitometry scorecard | U2.2, U3.2 | chart/EV/illuminant/roll/lab slices + CIs |
| U4.3 | pending | Grain/halation/MTF suite | U3.2 | NPS/radial/MTF test vectors and repeatability |
| U4.4 | ready | Blind pairwise study protocol | competitor-output rights check | preregistered questions, exclusions, analysis |
| U4.5 | pending | Windows/M5/CPU performance matrix | U1.6 | cold/warm p50/p95, RAM/VRAM, 24/100MP, batch |
| U4.6 | ready | Retire chroma-gain promotion gate | none | New evaluator does not reward saturation as identity |

Four reports stay separate:

1. content/geometry safety;
2. stock/process/interpretation authenticity;
3. physical-effect fidelity;
4. human preference and product performance.

Never collapse them into one score.

Provisional promotion rules, frozen before each experiment:

- hard safety: no new text errors; no geometry resampling; face/keypoint diagnostics within benign-control tolerance;
- authenticity: full-roll holdout improves against the strongest simpler baseline, including P95/tails;
- blind stock-match: lower bound of 95% CI > 50% against the current champion;
- content-preservation human score non-inferior to deterministic baseline by a preregistered margin;
- exact thresholds for ΔE, repeatability and latency are calibrated by the pilot, not invented after seeing test results.

### U5 — Bounded-AI challenge

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U5.0 | pending | Challenge protocol and fixed split | U3.5, U4.1, U4.2 | Signed experiment spec |
| U5.1 | pending | 1D + 33³/65³ 3D LUT baseline | U5.0 | Full scorecard and artifacts |
| U5.2 | pending | Existing SepLUT/NILUT retrained on real paired targets | U5.1 | Same compute/data/eval; no pseudo-teacher claim |
| U5.3 | pending | Low-resolution bilateral-grid predictor | U5.1 | Held-out local residual improvement; no halo |
| U5.4 | pending | Bounded masks/semantic conditioning | U5.3 failure pattern | Only if systematic region errors remain |
| U5.5 | research-only | FLUX.2 Klein/IP2P target→LUT projection | U4.1 | Isolated report; generated RGB never Reference output |
| U5.6 | pending | Winner distillation/runtime conversion | U5.1–U5.4 winner | Non-inferior FP32/FP16/CoreML/ONNX parity |

Stop at the simplest passing model. A valid result is “the analytic/3D-LUT renderer wins; delete the neural dependency.”

FLUX.2 gate:

- official sources conflict between roughly 8GB and 13GB for 4B inference;
- use official FP8/offload smoke only after explicit model-download approval if needed;
- 12GB LoRA training is unsupported by current official evidence; 24GB+ remote/costly work requires approval;
- DreamLite and noncommercial/gated candidates remain watchlist only.

### U6 — Physical effects

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U6.0 | complete | Current deterministic halation/grain baseline documented | existing code | Current reports; explicitly heuristic |
| U6.1 | pending | Halation moved to exposure/density domain | U2.2, U4.3 | Real patch radial/color/exposure fit |
| U6.2 | pending | Density/channel/resolution-aware grain | U4.3 | NPS/autocorrelation/repeat-scan comparison |
| U6.3 | pending | Stock/scanner MTF model | U4.3 | Edge/line-pair validation |
| U6.4 | pending | Bloom separated from halation | U6.1 | Separate parameter/evaluator/UI layer |
| U6.5 | pending | Creative dust/scratch/light leak labels | U2.1 | Never included in calibrated score silently |
| U6.6 | pending | 100MP/video determinism | U1.6, U6.1–U6.4 | Seam-free stills; temporal report |

The IPOL grain paper may guide a clean-room mathematical implementation. Its GPL code must not be copied into a permissive core without a deliberate licensing architecture.

### U7 — Productization

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U7.1 | pending | Stable pure recipe→render API | U2, U5/U6 winners | API compatibility tests |
| U7.2 | pending | CLI inspect/render/verify/benchmark + batch | U7.1 | End-to-end fixtures and error codes |
| U7.3 | pending | Desktop browser/preview/history/export | U7.1 | UX acceptance and accessibility checks |
| U7.4 | pending | Windows CUDA/ONNX path | U5.6 | 12GB peak and parity report |
| U7.5 | pending | Apple Core ML/Metal path | U5.6 | M5 thermal/swap/parity report |
| U7.6 | pending | CPU fallback | U7.1 | Quality parity and bounded-memory test |
| U7.7 | pending | Packaging/update/recovery/privacy | U7.2, U7.3 | Clean install, rollback and no-pixel telemetry audit |

Provisional performance budgets, to be frozen only after U4.5 baseline:

- 2048px warm preview target P95 ≤ 300ms;
- 24MP Reference target P95 ≤ 3s RTX and ≤ 5s M5;
- 100 images complete with zero errors/leaks;
- tile memory bounded and GPU failure falls back without quality change.

These are product targets, not measured current claims.

### U8 — Release and expansion

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U8.1 | pending | 20–50 photographer closed beta | U7, legal/consent | preregistered study and issue closure |
| U8.2 | pending | Datasheet, model/profile cards, SPDX/CycloneDX BOM | U0.2, U3–U7 | Release bundle audit |
| U8.3 | pending | Signed/versioned profile packs and rollback | U8.2 | signature/update/rollback tests |
| U8.4 | pending | Vision3/Ektar/Portra800 expansion | Pilot success | Each profile independently gated |
| U8.5 | pending | Tri-X/HP5 process-specific profiles | B&W schema + data | Developer/process evidence |
| U8.6 | research-only | Schema-constrained language tool agent | U7.1 | outputs validated parameter JSON only |
| U8.7 | research-only | Creative generative mode | separate license/safety review | UI/storage/metadata isolation |

Public release requires explicit human approval and written license/trademark/security review. C2PA may record the editing chain; it is not a truth detector.

---

## 5. First 30 days: ready leaves

### Week 1 — truth and guardrails

1. `U0.1` (complete 2026-07-10): reconcile active docs and mark diffusion/IP2P as retired research.
2. `U0.3`: define manifest v2; tag every current source `production/research/eval/blocked`.
3. `U4.6`: replace the chroma-gain promotion rule with four scorecards.
4. `U0.2`: ask owner to confirm intended code license and release posture.

Evidence bundle: document diff, manifest schema, duplicate/leakage report, release-blocker list.

### Week 2 — renderer ingress/egress

1. `U1.1`: route `render_film` through `WorkingImage`.
2. `U1.3`: implement true format/bit-depth/ICC-preserving export.
3. Add E2E JPEG/PNG/TIFF/RAW tests; unsupported HEIF/HDR fails closed.

Stop if the new path cannot reproduce current 8-bit legacy fixtures; repair compatibility before adding features.

### Week 3 — schema and reference skeleton

1. `U2.1`: profile/recipe schema.
2. `U2.2`: identity/monotone curves.
3. `U2.3`: identity tetrahedral 3D LUT.
4. `U2.5`: safe_lab adapter.

Evidence bundle: schema fixtures/migrations, property tests, identity golden vectors, CLI `inspect` prototype.

### Week 4 — evaluation and capture readiness

1. `U4.1`: expand the frozen safety corpus with face/text/skin/sky/ramp/texture/highlight cases.
2. `U4.4`: preregister blind-study protocol.
3. `U3.1`: finalize capture/lab/scanner/rights SOP and obtain quotes/approval.
4. `U4.5`: benchmark current renderer on exact target hardware.

No film purchase, lab booking, model download or GPU training occurs without the corresponding approval.

---

## 6. Experiment queue

| Experiment | Hypothesis | Fixed baseline | One changed variable | Promotion gate | Stop condition |
|---|---|---|---|---|---|
| EXP-COLOR-01 | Monotone curves + 3D LUT capture stock response | safe_lab + measured target | global transform | Authenticity improves on held-out roll; safety unchanged | No stable held-out gain |
| EXP-COLOR-02 | SepLUT models global residual better | EXP-COLOR-01 | representation | Tail metrics improve, complexity justified | Same/poorer result |
| EXP-LOCAL-01 | Bilateral grid fixes scene-local residual | current global winner | local grid | Local slices improve; no halo/tile seam | Artifact or no independent gain |
| EXP-FX-01 | Exposure-domain halation matches real radial behavior | current display-level halation | composition domain | Held-out point/edge profile improves | Lens/scanner confound unresolved |
| EXP-FX-02 | Density-aware grain matches real NPS | current procedural grain | density-conditioned params | Held-out NPS/ACF within repeatability | Scanner noise not separated |
| EXP-GEN-01 | FLUX.2 FP8 is a better color oracle than IP2P | IP2P fixed grid | teacher model | LUT projection gains preference without safety loss | OOM, license issue or no gain |
| EXP-RUNTIME-01 | FP16/CoreML/ONNX is color-noninferior | FP32 reference | runtime/precision | ΔE/golden parity + speed gain | Neutral/skin drift or instability |

Each experiment produces: contract, config, input manifest hash, environment, raw per-image metrics, failure gallery, aggregate with CI, decision and scoped commit. No result may update a profile merely because one contact sheet looks good.

---

## 7. Multi-agent / multi-chat ownership protocol

Before parallel writes, each leaf declares:

- owner;
- allowed files/artifacts;
- forbidden files/artifacts;
- dependencies and required commit;
- evidence bundle;
- stop condition;
- integration owner.

Default protected shared files: `AGENTS.md`, `README.md`, `IMPL_PLAN.md`, `TASK_BOARD.md`, this tracker, profile schema, lock files and data manifests. Only the integration owner edits them during a parallel wave. Other agents remain read-only or use disjoint files.

Any concurrent chat must refresh `AGENTS.md`, `git status`, recent log, this tracker and outstanding claims immediately before writing. Overlapping claims pause rather than merge implicitly.

---

## 8. Change propagation checklist

For every meaningful node change:

- [ ] classify implementation / evidence / scope / risk / interface change;
- [ ] update parent goal status;
- [ ] inspect child assumptions;
- [ ] inspect sibling baselines and shared evaluator;
- [ ] inspect data/model/profile dependencies;
- [ ] inspect CLI/API/GUI consumers;
- [ ] update tests and golden vectors;
- [ ] update active docs and archive superseded claims without deleting history;
- [ ] append `docs/drpt/AGENT_LOG.md`;
- [ ] re-run scoped checks and `git diff --check`;
- [ ] make one narrow local commit;
- [ ] record remaining approval and unknowns.

---

## 9. Approval gates

Explicit owner approval is required before:

- choosing/adding a public repository license;
- purchasing film, lab work, scanner access, data or competitor software;
- downloading large models/data or starting costly GPU/cloud training;
- sending messages, recruiting subjects/testers or publishing surveys;
- uploading private images, manifests or checkpoints;
- pushing/merging branches, opening a PR or public release;
- deleting/relocating large caches or user artifacts;
- making claims using film-stock trademarks in a commercial product.

Current approval state: research and local documentation are authorized; all items above remain unapproved.

---

## 10. Current evidence anchors

- Diffusion/IP2P failure: `docs/CURRENT_STATUS_2026-05-27.md`, `docs/SDXL_LORA_VALIDATION_RESULTS.md`, `docs/IP2P_GRID_SEARCH_RESULTS.md`
- Deterministic baseline: `docs/COLOR_BASELINE_RESULTS.md`, `docs/COLOR_BASELINE_STABILITY_RESULTS.md`
- Saturation-gate failure: `docs/AI_COLOR_ENGINE_CHALLENGE_RESULTS.md`
- Pseudo-teacher LUTs: `docs/NEURAL_FILM_LUT_V2_RESULTS.md`
- Input path: `docs/PREPROCESSING_INPUT_PIPELINE_TRACKER.md`, `src/preprocess/`, `scripts/render_film.py`
- FiveK deletion/freeze state: `docs/FIVEK_AUTO_BASE_MODEL_TRACKER.md`
- Halation evidence limits: `docs/HALATION_SYSTEM_SPEC.md`
- Data rights boundary: `docs/data/DATA_LICENSE_BOUNDARIES.md`
- Research synthesis and primary external links: `docs/planning/ULTIMATE_ROADMAP_2026.md`

---

*Tracker initialized: 2026-07-10. Next ready leaf: U0.3. Integration owner: repository owner or explicitly assigned Codex root agent.*
