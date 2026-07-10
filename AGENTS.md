# AGENTS.md — K-MCFM Project Knowledge Base

> **Current truth: 2026-07-10.** The production-capable path is a deterministic content-safe color renderer plus procedural effects. The earlier SDXL/IP2P/SDEdit direction was experimentally rejected as the default because it rewrites detail/identity or is infeasible on the 12GB target GPU.
> **Target direction:** a calibrated, color-managed Reference renderer with optional bounded-AI parameter prediction; generative editing remains isolated as Creative/R&D.

---

## 1. Project identity

| Field | Value |
|---|---|
| Project | K-MCFM — content-preserving film imaging |
| Current default | deterministic `safe_lab` / safe-rich color path + optional grain/halation/dust |
| Ultimate target | measured stock + process + scan/print interpretation profiles, high-precision RAW/HDR pipeline, bounded local color model |
| Content contract | Reference/Adaptive paths may change color and non-warping effects on a fixed pixel grid, never intentionally change geometry, objects, text or identity |
| Target hardware | M5 32GB and RTX 5070 Ti **Laptop** 12GB; CPU fallback |
| Current evidence | 18 local tests pass; deterministic renderer is usable; stock accuracy is not yet calibrated |
| License | old docs say MIT, but no root `LICENSE` exists; public release is blocked until the owner decides and adds one |

Do not describe the project as “Film Translation via InstructPix2Pix” or claim that diffusion is the current content-preserving solution.

---

## 2. Read order

| Order | File | Purpose |
|---:|---|---|
| 1 | `AGENTS.md` | Current truth and invariants |
| 2 | `docs/ULTIMATE_EXECUTION_TRACKER.md` | Active DRPT task tree, gates and next ready leaves |
| 3 | `docs/planning/ULTIMATE_ROADMAP_2026.md` | Research synthesis, target architecture and primary sources |
| 4 | `TASK_BOARD.md` | Compact active board/pointer |
| 5 | `IMPL_PLAN.md` | Active-plan pointer plus historical V3 plan |
| 6 | `docs/CURRENT_STATUS_2026-05-27.md` | Diffusion/IP2P failure and deterministic pivot |
| 7 | `docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md` | Current renderer implementation and promoted effects |
| 8 | `docs/PROJECT_STRUCTURE.md` | Repository placement and safe cleanup rules |
| 9 | `docs/EXPERIMENT_LOG.md` | Historical experiments |

The following are historical context, not active authority: `docs/ARCH_REDESIGN.md`, `docs/planning/GAP_ANALYSIS.md`, `docs/ONLINE_DATA_AUDIT.md`, and the diffusion sections below the supersession banner in `IMPL_PLAN.md`.

---

## 3. Current implementation

```text
PIL 8-bit RGB input
  -> deterministic CIELAB mean/std transfer (`safe_lab`, safe-rich guards)
  -> optional deterministic grain / physical-inspired halation / dust
  -> bounded 8-bit output
  -> PNG encoder
```

Current strengths:

- fixed-grid deterministic color transform; no generative geometry rewrite;
- stable safe-rich guardrails and fixed test artifacts;
- modular grain/halation/dust layers;
- `WorkingImage`, ICC-aware raster helpers and generic RAW decoding exist under `src/preprocess/`;
- substantial experiment logs and comparison artifacts are preserved.

Current limitations:

- `scripts/render_film.py` does not use `WorkingImage`; it converts input to PIL RGB and writes 8-bit PNG;
- no complete HEIF/HDR/gain-map or wide-gamut production path;
- no stock/process/scanner-calibrated ground truth;
- existing Lab statistics can make stocks look similar;
- current halation numbers are explicitly uncalibrated heuristics;
- current neural LUTs were mainly distilled from a pseudo-teacher;
- no CI, packaging, stable engine API, production GUI or cross-platform parity suite.

---

## 4. Target architecture

```text
RAW / Log / HDR / SDR
  -> color-state validation (`scene`, `display`, `unknown`)
  -> high-precision WorkingImage + versioned input transform
  -> optional neutral auto-base
  -> stock exposure + monotone sensitometry curves
  -> global 3D LUT / SepLUT / NILUT
  -> optional bounded bilateral-grid local residual
  -> explicit interpretation:
       color negative -> neutral scan or print chain
       motion negative -> neutral scan or print stock
       slide -> direct scan
       B&W -> developer + scan/print chain
  -> exposure/density-domain halation
  -> density-aware grain + MTF
  -> separate bloom / creative defects
  -> versioned OCIO/ACES output transform
  -> image + replayable recipe + provenance
```

Architecture rules:

1. `WorkingImage` becomes the only production ingress.
2. Unknown color state fails closed to **Look Approximation**, not Reference.
3. The full-resolution path remains deterministic and does not spatially resample in Reference/Adaptive modes.
4. A learned model may predict curves, LUT weights, grids, masks or effect parameters; it may not produce the final Reference RGB image.
5. Generative models can be low-resolution teachers or Creative output only.
6. Use the simplest candidate that passes held-out gates; a no-neural-network winner is acceptable.

---

## 5. Mode contract

| Mode | Allowed | Forbidden | Output label |
|---|---|---|---|
| Reference | calibrated fixed-grid color and non-warping effects with known input state | geometry/object/text/identity changes | `reference` |
| Adaptive | bounded parameter prediction + deterministic render | generative RGB, spatial warp | `adaptive-bounded` |
| Look Approximation | deterministic rendering from unknown/display-referred input | reference-grade authenticity claim | `look-approximation` |
| Creative | explicit generative editing | presenting output as calibrated or identity-safe | `creative-generative` |

Every render records input hash/color state, profile and model hashes, code commit, seed, parameters, output transform, evidence grade and mode.

---

## 6. Experiment truth

### Retired as default

- SD1.5 IP2P: fine-tuning produced severe painterly smearing.
- SDXL full-UNet IP2P: OOM on the 12GB GPU even at very low resolution in tested settings.
- SDXL LoRA + SDEdit: low strength had little effect; useful-looking strength rewrote faces, clothing and details.
- IP-Adapter/ControlNet: conditions do not create a pixel-identity guarantee.

Keep these artifacts for research and regression. Do not restart the same grid without a new falsifiable hypothesis.

### Research-only candidates

- existing SepLUT/NILUT/4D proxies: must be retrained on real paired targets;
- local bounded maps: old automatic gate rewarded at least 3% chroma and the user judged outputs mainly as saturation gain;
- FLUX.2 Klein 4B: promising Apache-2.0 creative/teacher challenger, but official sources conflict on roughly 8GB vs 13GB VRAM; 12GB support requires FP8/offload measurement;
- community film LoRAs: asset-level license and base-model compatibility must be audited.

---

## 7. Data and claim boundaries

| Source | Default lane | Valid use | Invalid default use |
|---|---|---|---|
| Self-owned paired digital/film captures | production candidate | calibration/training/evaluation after rights closure | none until rights and split are complete |
| Flickr film images | research-only | unpaired aesthetics/failure analysis | commercial weights or stock truth |
| FilmSet | research-only | Capture One recipe baseline/warmup | real film scan truth |
| MIT-Adobe FiveK | research-only | neutral auto-base research | film identity or automatic public-weight clearance |
| FilmGrainStyle740k | research-only | academic comparison under its terms | commercial development/training |
| Manufacturer data sheets | prior | curve/sensitivity/MTF/granularity initialization | end-to-end RGB target |
| Community LoRAs | research-only until audited | Creative comparison | Reference core |

Full FiveK RAW/TIFF sources were deleted locally on 2026-06-15 after a verified 6.98GB freeze pack was retained. Do not plan full-scale FiveK work unless sources are restored.

All new manifest rows need source URL/ID, author, license snapshot/date, rights scope, scene/roll/lab/scanner/uploader group, content and perceptual hashes, derivation lineage and allowed-use fields. Split by group; exact/perceptual cross-split leakage must be zero before training.

---

## 8. Evaluation contract

Never use one aggregate score for promotion. Maintain four independent scorecards:

1. **content/geometry safety** — dimensions, no warp, edge/keypoint location, face/text diagnostics, tile/determinism;
2. **stock/process authenticity** — exposure/density curves, chart color, illuminant/EV/roll/lab/scanner slices and complete holdouts;
3. **physical effects** — grain NPS/autocorrelation/density dependence, halation radial/color/exposure behavior, MTF and bloom separation;
4. **human/product** — blinded stock-match vs preference, confidence intervals, latency/RAM/VRAM/cross-platform stability.

L-SSIM and the current `[4,251]` range remain legacy 8-bit regression diagnostics, not universal 16-bit/HDR or film-authenticity gates. Freeze test groups and thresholds before seeing final results; report tails and failures, not only means.

---

## 9. Pilot stocks and capture order

1. Portra 400: C-41 color negative + explicit neutral scan interpretation.
2. Velvia 50: E-6 slide + direct scan interpretation.
3. Vision3 500T/250D: ECN-2, with neutral scan and print-film interpretations separated.
4. Ektar 100 / Portra 800.
5. Tri-X 400 / HP5 Plus: developer/process-specific B&W profiles.

Pilot requires controlled charts, -3EV..+3EV sequences, daylight/tungsten/LED/mixed lighting, real scenes, repeated scans, at least three independent rolls per stock spanning at least two recorded process sessions, and a whole-roll holdout. Pilot quantities are engineering starts, not statistical guarantees.

---

## 10. Immediate priorities

| Priority | Node | Work |
|---:|---|---|
| Done | U0.1 | Active docs reconciled and stale diffusion instructions marked historical on 2026-07-10 |
| P0 | U0.2 | Owner decides repository license; add legal artifacts only after approval |
| P0 | U0.3 | Repair manifest lineage and cross-split leakage |
| P0 | U0.4/U4 | Add CI, frozen benchmark and four scorecards |
| P0 | U1 | Connect `WorkingImage`, 16-bit/profile-aware I/O and color-state contract |
| P1 | U2 | Implement profile/recipe schema and deterministic reference renderer |
| P1 | U3 | Run rights-cleared Portra/Velvia calibration pilot after approval |
| P1 | U5/U6 | Challenge bounded AI and calibrate effects only after data/eval gates |
| P2 | U7/U8 | Productize, beta, expand stocks and isolate Creative mode |

The active dependencies, DoR/DoD and stop rules live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.

---

## 11. Hardware and risk gates

- RTX target is a 12GB Laptop GPU; do not use desktop 5070 Ti figures.
- Reference renderer and small predictor should fit comfortably; leave peak-memory headroom.
- FLUX.2 Klein BF16 is not assumed to fit 12GB; FP8/offload is an optional measured experiment.
- Apple unified memory success must include swap and thermal behavior, not only successful load.
- Do not start paid cloud/GPU work, large downloads, film/lab purchases, external recruiting, release or deployment without explicit approval.

---

## 12. Update protocol

Update this file when:

1. the production default or architecture boundary changes;
2. a candidate is promoted/retired by a complete evidence bundle;
3. data rights, split or source availability changes;
4. a stock profile reaches a new evidence grade;
5. hardware support is measured on a new target;
6. release/license posture changes.

For non-trivial changes, also update `docs/ULTIMATE_EXECUTION_TRACKER.md` and `docs/drpt/AGENT_LOG.md`, run propagation checks, preserve unrelated user files and make a scoped local commit.

---

*Last updated: 2026-07-10 | Current implementation: deterministic content-safe renderer | Target: calibrated hybrid film-imaging system*
