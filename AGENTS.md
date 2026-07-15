# AGENTS.md — K-MCFM Project Knowledge Base

> **Current truth: 2026-07-15.** The production-capable path is a deterministic content-safe color renderer plus procedural effects. The earlier SDXL/IP2P/SDEdit direction was experimentally rejected as the default because it rewrites detail/identity or is infeasible on the 12GB target GPU.
> **Target direction:** engineering may use the strongest rights-compatible colour-transfer method, but the primary research paper must itself perform colour transfer. The product path is now independent: fixed strong explicit experts, optional transform-aware hard selection, image-level validated support/strength and a severe-artifact veto. Roll2Film remains an offline research challenger whose special group-information hypothesis is not established until fixed-sample matched controls pass. ChromaticTail/FilmStyleSafe is supporting evaluation, FARO is a product safety/system wrapper, FilmCase opens only after diversity and Oracle gates, calibrated profiles are deferred, and generative RGB editing remains separately labeled Creative work.

---

## 1. Project identity

| Field | Value |
|---|---|
| Project | K-MCFM — content-preserving film imaging |
| Current default | deterministic `safe_lab` / safe-rich color path + optional grain/halation/dust |
| Ultimate target | strongly stylized film-inspired output under an explicit severe-artifact budget; independent deterministic product winner, conditional Roll2Film algorithm research, high-precision RAW/HDR rendering and deferred calibrated profiles |
| Product standard | maximize visible style and preference subject to a hard severe-artifact veto |
| Content contract | stylization may be strong, but confirmed severe face/text/object corruption, geometry failure, banding, seams, clipping or unstable color artifacts block promotion |
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
| 3 | `docs/planning/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_INTEGRATION_20260715.md` | Current evidence adjudication, product/research split and corrected execution order |
| 4 | `docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md` | Conditional algorithm-first colour-transfer hypothesis, data gates, novelty boundary and experiment DAG |
| 5 | `docs/planning/FARO_RESEARCH_PROGRAM_2026.md` | Supporting artifact evaluation and product/system-risk program; historical paper priority is superseded |
| 6 | `docs/planning/ULTIMATE_ROADMAP_2026.md` | Earlier strategic synthesis, product architecture and primary sources |
| 7 | `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md` | FilmCase baseline hypotheses and ablations, subordinate to diversity/Oracle gates |
| 8 | `TASK_BOARD.md` | Compact active board/pointer |
| 9 | `IMPL_PLAN.md` | Active-plan pointer plus historical V3 plan |
| 10 | `docs/CURRENT_STATUS_2026-05-27.md` | Diffusion/IP2P failure and deterministic pivot |
| 11 | `docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md` | Current renderer implementation and promoted effects |
| 12 | `docs/PROJECT_STRUCTURE.md` | Repository placement and safe cleanup rules |
| 13 | `docs/EXPERIMENT_LOG.md` | Historical experiments |

The following are historical context, not active authority: `docs/ARCH_REDESIGN.md`, `docs/planning/GAP_ANALYSIS.md`, `docs/ONLINE_DATA_AUDIT.md`, and the diffusion sections below the supersession banner in `IMPL_PLAN.md`.

---

## 3. Current implementation

```text
WorkingImage linear-sRGB input
  -> explicit temporary sRGB8 compatibility adapter
  -> PIL 8-bit RGB legacy renderer
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

- `scripts/render_film.py` now uses `WorkingImage` as ingress, but immediately crosses an explicit legacy sRGB8 adapter and still writes 8-bit PNG;
- no complete HEIF/HDR/gain-map or wide-gamut production path;
- no stock/process/scanner-calibrated ground truth;
- existing Lab statistics can make stocks look similar;
- current halation numbers are explicitly uncalibrated heuristics;
- current neural LUTs were mainly distilled from a pseudo-teacher;
- CPU-safe CI exists; packaging, stable engine API, production GUI and cross-platform parity suite remain incomplete.

---

## 4. Target architecture

```text
RAW / Log / HDR / SDR
  -> color-state validation (`scene`, `display`, `unknown`)
  -> high-precision WorkingImage + versioned input transform
  -> optional neutral auto-base
  -> FilmCase eligibility/OOD:
       global deterministic fallback or hard-selected bounded case expert
  -> style/stock exposure + monotone curves
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
3. The full-resolution Style-safe/Calibrated path remains deterministic and does not spatially resample.
4. A learned model may predict curves, LUT weights, grids, masks or effect parameters; it may not directly produce the final Style-safe RGB image.
5. FilmCase cannot depend on new user images, film/digital pairs, per-image labels or additional preference votes.
6. Unpaired film images are style references, never input/output pairs; FilmCase output stays labeled `film-inspired/unpaired-evidence`.
7. FilmCase uses hard Top-1/medoid selection by default; low confidence, OOD or unknown state falls back to the global deterministic champion.
8. Generative image models are excluded from FilmCase and can only exist in a separately approved Creative branch.
9. Use the simplest candidate that passes held-out gates; a no-neural-network winner is acceptable.
10. The research method must render a transformed image through an explicit colour operator; a metric, benchmark, selector or rejection policy alone is not the primary paper contribution.
11. Roll2Film treats a roll as a repeated-measure weak-supervision group. Without roll/process/scanner metadata, its inferred transform is a `roll-look`, not a stock response.

---

## 5. Mode contract

| Mode | Allowed | Forbidden | Output label |
|---|---|---|---|
| Style-safe | strong deterministic/bounded color and effects | confirmed severe glitch/artifact | `film-inspired` |
| Calibrated Reference | Style-safe render plus owned/cleared stock/process evidence | unsupported authenticity claims | `calibrated-reference` |
| Creative | explicit generative editing | presenting output as artifact-safe or calibrated | `creative-generative` |

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

- Roll2Film: unpaired roll-set repeated measures -> shared explicit roll-look operator -> hidden transfer evaluation; fixed-budget affine method controls pass, but physical-roll information remains unestablished until matched real-roll controls;
- FilmCase: source-controlled identifiability → bounded case bank → Evaluator Oracle → simplest generic or transform-aware retrieval → hard sparse router/OOD fallback;
- existing SepLUT/NILUT/4D proxies: may challenge the Style-safe frontier on rights-cleared preference targets; real paired targets are required only for calibrated claims;
- local bounded maps: old automatic gate rewarded at least 3% chroma and the user judged outputs mainly as saturation gain;
- FLUX.2 Klein 4B: excluded from FilmCase; a future Creative-only candidate requiring separate instruction/approval, with unresolved 12GB support;
- community film LoRAs: asset-level license and base-model compatibility must be audited.

---

## 7. Data and claim boundaries

| Source | Default lane | Valid use | Invalid default use |
|---|---|---|---|
| Self-owned paired digital/film captures | deferred calibration candidate | future calibration after a new explicit scope and rights closure | active FilmCase dependency or current user ask |
| Local Flickr film images | quarantined | internal aesthetics/failure analysis only | paper training, released weights, commercial weights or stock truth |
| FilmSet | locally present, research-only | paired-blind Capture One recipe transfer and supervised upper bound | real film scan truth |
| BlueNeg | remote-verified, research-only, custom attribution license | roll-group information pilot using metadata and small preview lanes | digital-to-film ground truth or clean named-stock target |
| MIT-Adobe FiveK | partial local freeze only, auxiliary | neutral auto-base research after a separate restore decision | film identity or automatic public-weight clearance |
| FilmGrainStyle740k | research-only | academic comparison under its terms | commercial development/training |
| Manufacturer data sheets | prior | curve/sensitivity/MTF/granularity initialization | end-to-end RGB target |
| Community LoRAs | research-only until audited | Creative comparison | Reference core |

Current Windows audit, 2026-07-15: `data/film_domain` contains 4,212 JPEGs, while the earlier 4,210-row legacy lineage audit remains 0 eligible/4,210 quarantined and must not be silently generalized to new files. `data/processed/manifest.jsonl` exists. The complete decompressed FilmSet image tree is local: 21,140 files / 11,262,805,356 bytes, with 4,657 identities per train domain and **628** per test domain. The paper-reported 638 is retained as a publication contradiction, never used at runtime. A partial FiveK freeze is present; it is not the complete source.

BlueNeg remains absent but remote-verified through its public file tree and a successful range read; its metadata records 491 frames, 53 rolls and 13 film-type strings, while its initial 8-bit preview plus pseudo-ground-truth lanes are about 956MB. FilmSet local internal use needs no new download approval, but its final 628 lockbox requires access controls and release rights remain unresolved. On 2026-07-15 the owner pre-authorized necessary data downloads; BlueNeg still waits on the scientific CT3/metadata/whole-roll-split gate rather than another permission request. Neither resource changes the claim boundaries above.

All new manifest rows need source URL/ID, author, license snapshot/date, rights scope, scene/roll/lab/scanner/uploader group, content and perceptual hashes, derivation lineage and allowed-use fields. Split by group; exact/perceptual cross-split leakage must be zero before training.

---

## 8. Evaluation contract

The product objective is constrained optimization:

> **Maximize film-style strength and user preference among candidates with no confirmed severe glitch/artifact on the frozen gold set.** Report artifact rate and confidence intervals on the wider stress set instead of claiming universal zero defects.

Maintain five independent scorecards:

1. **severe artifact veto** — face/limb/object/text corruption, broken geometry, VAE smearing, posterization, banding, large unintended clipping, tile seams, color blocks, repeated textures or temporal flicker;
2. **style salience and appeal** — visibly stylized, recognizably film-inspired, and preferred over the technically clean bland baseline;
3. **content retention** — graded edge/keypoint/face/text/texture diagnostics after the severe veto, not a requirement to stay visually close to the input;
4. **calibrated authenticity** — stock/process/scan evidence only when a profile claims `calibrated`; film-inspired looks may ship without this label;
5. **physical/product quality** — effect plausibility, determinism, latency, memory and cross-platform stability.

Promotion order is severe-artifact veto → maximize style/appeal → check graded content and product quality. Authenticity is a conditional gate for calibrated claims, not the universal product objective. Intended grain, halation, bloom and strong tone/color are not artifacts by themselves; they fail only when they create objectionable corruption or instability. L-SSIM and `[4,251]` remain legacy diagnostics, not universal style gates.

For autonomous FilmCase research, `53/55/56/09/01` are the complete frozen owner-preference anchors; `33/03/02` are smoke-only cues. The 2026-07-15 external audit found 53/55/56 near-collinear after same-input normalization and visually flagged ID 11 outputs 09/53/55/56 for red-speckle/posterization failure. Treat this as external autonomous evidence: freeze it into the diversity/worst-case tests, but do not call it completed internal blind adjudication. Repeated blind Codex vision audits, nuisance-matched controls and full-resolution adjudication replace new owner votes. These results must be labeled autonomous visual evidence, not population preference. External human validation is deferred to U8.

---

## 9. Deferred calibrated stock order

This is a future evidence design, not an active request for user data and not a FilmCase dependency:

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
| Done | U5.CT0 | Reframe primary publication work as Roll2Film colour-transfer algorithm; demote benchmark/system work to support |
| P0 | U5.CT1/U5.CT3 | Fixed-budget affine and L2 controls pass; finish explicit L0/gauge/shaper closure and matched real-roll controls before any roll-information promotion |
| Done/in progress | U5.CT2/U5.CT4 | FilmSet archive/pair-blind/628 lockbox freeze passed; CT2 continues only for BlueNeg metadata and whole-roll gates |
| P0 | U0.2 | Owner decides repository license; add legal artifacts only after approval |
| Done/blocked | U0.3 | Legacy 4,210-row audit completed fail-closed with 0 eligible rows; new remote-data grouping contracts move to U5.CT2 |
| Done/ongoing | U0.4/U4 | CI/registry foundation is complete; maintain severe-artifact/style evaluation as Roll2Film support and product QA, not the primary paper |
| P0 | U1 | Connect `WorkingImage`, 16-bit/profile-aware I/O and color-state contract |
| P1 | U2 | Implement profile/recipe schema and deterministic reference renderer |
| Deferred | U3 | Reopen paired Portra/Velvia calibration only after a future explicit scope |
| Conditional | U5.CT5–U5.CT8 | FilmSet data gate passed; run internal pair-blind development only after nonlinear CT3 and matched-strength baseline freeze, then continue the simplest survivor |
| P0 product | U5.FC1–U5.FC8/U6 | Build fixed explicit champion/bank independently; FilmCase routing requires ≥2 modes plus an Oracle gap, otherwise use champion + bounded strength |
| P2 | U7/U8 | Productize, beta, expand stocks and isolate Creative mode |

The active dependencies, DoR/DoD and stop rules live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.

---

## 11. Hardware and risk gates

- RTX target is a 12GB Laptop GPU; do not use desktop 5070 Ti figures.
- Style-safe renderer and small predictor should fit comfortably; leave peak-memory headroom.
- FLUX.2 Klein is outside FilmCase; if a future Creative instruction reopens it, BF16 is not assumed to fit 12GB and FP8/offload must be measured.
- Apple unified memory success must include swap and thermal behavior, not only successful load.
- Necessary dataset downloads are owner-authorized as of 2026-07-15, but must still have a scientific gate, source/licence snapshot, storage/retention plan and bounded scope; this does not justify the 290GB BlueNeg full archive while the 0.956GB pilot suffices. Do not start paid cloud/GPU work, film/lab purchases, external recruiting, release or deployment without separate explicit approval.

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

*Last updated: 2026-07-12 | Current implementation: deterministic content-safe renderer | Research target: Roll2Film colour transfer | Product target: Style-safe deterministic/operator core | Calibrated lane deferred*
