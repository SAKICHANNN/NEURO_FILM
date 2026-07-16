# AGENTS.md — K-MCFM Project Knowledge Base

> **Current truth: 2026-07-15.** The production-capable path is a deterministic content-safe color renderer plus procedural effects. Ultimate success now requires stock-first learning from verifiable real photographic-film scans; specific `film_stock_id` experts are primary, while historical/unknown-stock film is a separate auxiliary class. FilmSet/Capture One, camera simulations, LUTs and pseudo-teachers are controls only.
> **Target direction:** compare stock-specific global, hierarchical, retrieval and bounded conditional explicit operators. Roll, process, scanner, source and content are nested nuisance/group variables. The physical-roll-only Roll2Film hypothesis remains closed under current BlueNeg evidence. GPU models may predict bounded parameters but never directly generate final RGB. See `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md` and `docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`.

---

## 1. Project identity

| Field | Value |
|---|---|
| Project | K-MCFM — content-preserving film imaging |
| Current default | deterministic `safe_lab` / safe-rich color path + optional grain/halation/dust |
| Ultimate target | as many evidence-backed, distinguishable stock-specific real-film-derived experts as practical, visibly stylized under a severe-artifact budget, with unseen roll/source validation; historical/unknown-stock is separate and cannot replace named stocks |
| Product standard | maximize visible style and preference subject to a hard severe-artifact veto |
| Content contract | stylization may be strong, but confirmed severe face/text/object corruption, geometry failure, banding, seams, clipping or unstable color artifacts block promotion |
| Target hardware | M5 32GB and RTX 5070 Ti **Laptop** 12GB; CPU fallback |
| Current evidence | 170 local tests pass; the 104-file connected audit closes both stock-learning edges: global RGB is nonsignificant while low-frequency scene colour/content predicts labels, and Ektar source geometry is 92.86% separable. No training/operator fitting starts; only a metadata-only full-YFCC shared-author expansion remains open |
| License | old docs say MIT, but no root `LICENSE` exists; public release is blocked until the owner decides and adds one |

Do not describe the project as “Film Translation via InstructPix2Pix” or claim that diffusion is the current content-preserving solution.

---

## 2. Read order

| Order | File | Purpose |
|---:|---|---|
| 1 | `AGENTS.md` | Current truth and invariants |
| 2 | `docs/ULTIMATE_EXECUTION_TRACKER.md` | Active DRPT task tree, gates and next ready leaves |
| 3 | `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md` | Active real-film success condition, evidence ledger, data/algorithm gates and RF execution tree |
| 4 | `docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md` | Active stock-first source matrix, first pilots, experiment DAG, gates and failure branches |
| 5 | `docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md` | Primary stock label hierarchy, evidence grades, separate named/historical coverage and current data ledger |
| 6 | `docs/planning/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_INTEGRATION_20260715.md` | Prior evidence adjudication and product/research split |
| 7 | `docs/planning/FARO_RESEARCH_PROGRAM_2026.md` | Supporting artifact evaluation and product/system-risk program; historical paper priority is superseded |
| 8 | `docs/planning/ULTIMATE_ROADMAP_2026.md` | Earlier strategic synthesis, product architecture and primary sources |
| 9 | `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md` | FilmCase baseline hypotheses and ablations, subordinate to diversity/Oracle gates |
| 10 | `TASK_BOARD.md` | Compact active board/pointer |
| 11 | `IMPL_PLAN.md` | Active-plan pointer plus historical V3 plan |
| 12 | `docs/CURRENT_STATUS_2026-05-27.md` | Diffusion/IP2P failure and deterministic pivot |
| 13 | `docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md` | Current renderer implementation and promoted effects |
| 14 | `docs/PROJECT_STRUCTURE.md` | Repository placement and safe cleanup rules |
| 15 | `docs/EXPERIMENT_LOG.md` | Historical experiments |

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
12. `film_stock_id` is the highest-level real-film expert class. Physical roll, process, scanner, source and content are nested controls, not replacements for stock identity.
13. `historical-film/unknown-stock` is a valid independent expert/stress lane, but it never counts toward named-stock coverage or substitutes for a specific stock.
14. At inference the user selects the target stock; content-aware routing stays inside that stock and never guesses a film identity for the input digital photograph.

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

- Roll2Film: fixed-budget method controls and FilmSet transfer pass, but BlueNeg correct-roll gains change sign across held-out rolls and the roll-cluster interval crosses zero; physical-roll information is not established and CT7 stops;
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
| LOC FSA/OWI colour archive | public domain; metadata/pilot/group verified | `historical-film/unknown-stock` expert, ageing/scanner nuisance and stress lane | named-stock coverage, physical-roll, pure Kodachrome, calibrated stock or scanner-out truth |
| MIT-Adobe FiveK | partial local freeze only, auxiliary | neutral auto-base research after a separate restore decision | film identity or automatic public-weight clearance |
| FilmGrainStyle740k | research-only | academic comparison under its terms | commercial development/training |
| Manufacturer data sheets | prior | curve/sensitivity/MTF/granularity initialization | end-to-end RGB target |
| Community LoRAs | research-only until audited | Creative comparison | Reference core |

Current Windows audit, 2026-07-15: `data/film_domain` contains 4,212 JPEGs, while the earlier 4,210-row legacy lineage audit remains 0 eligible/4,210 quarantined and must not be silently generalized to new files. `data/processed/manifest.jsonl` exists. The complete decompressed FilmSet image tree is local: 21,140 files / 11,262,805,356 bytes, with 4,657 identities per train domain and **628** per test domain. The paper-reported 638 is retained as a publication contradiction, never used at runtime. A partial FiveK freeze is present; it is not the complete source.

BlueNeg metadata/licence/inventory and the exact 101-file / 118,929,719-byte pixel acquisition are local at revision `b038a1ae...`; all LFS hashes pass. The frozen four-roll `Kodak Gold 100-5` development/confirmatory and nested-LOO diagnostics are complete: correct physical-roll support loses to content-similar wrong-roll retrieval across all four raw roll means. The full 956MB lanes and 290GB archive remain absent. This is provisional single-stock archive/restoration evidence below transferable `S2`, not 13-film-type generalization or stock calibration.

Commons SF0.5, 2026-07-16: 36 derivative-only files / 27,812,816 bytes are hash/decode clean with zero exact or dHash<=4 pairs and no confirmed severe visual corruption. Only Ektar100 passes the source gate (26 files, eight normalized authors, 30.77% largest share). Superia has 2 files/2 authors; Gold has 8/4 and 62.5% largest share plus a repeated same-author content cluster. All remain `S0`; one passing stock cannot open learning.

Commons SF0.6A, 2026-07-16: a metadata-only sweep of 10 further exact categories / 393 rows yields one new pass, UltraMax400 (84 files, 15 normalized authors, 35.71% largest share, 51 strict derivative-rights rows). Ektar+UltraMax is still below the three-stock comparative minimum; SF0.6B continues and no new pixels or learning are allowed yet.

Commons SF0.6B, 2026-07-16: Kodachrome64 adds the third metadata source pass (51 files, 22 authors, 25.49% largest share, 15 strict derivative rows). One GFDL1.2 row is metadata-only and excluded from pixels. UltraMax/Kodachrome pixels still require SF0.7; three metadata passes do not establish stock signal.

Commons SF0.7, 2026-07-16: 51 UltraMax/Kodachrome derivatives / 42,255,167 bytes are decode/hash/duplicate and visually clean. UltraMax passes with 37 files/eight authors/32.43% largest share. Kodachrome's strict pool collapses to 14 files/two authors/85.71% and obvious bridge-versus-stone content clusters, so it stops. Ektar+UltraMax are only two pixel-passing stocks; no learning opens.

YFCC SF0.8A, 2026-07-16: the bounded YFCC15M metadata freeze contains 7,350,000 rows in ten Parquet files / 1,738,329,293 bytes and no image payloads. A deterministic CC-BY-2.0 exact-text audit retains 51 Velvia50 rows across 26 Flickr UIDs; prospective process/HDR exclusions leave 45/23. This opens only a maximum-32 live-page/pixel pilot. YFCC user text is weak label evidence, UID is not person identity, Portra generation strings remain mixed, and no third pixel stock or stock signal is established.

YFCC SF0.8B, 2026-07-16: live page/licence, decode and dimension checks retain 25 Velvia50 files / 4,575,435 bytes across 14 UIDs at 16% largest share. Two repeated audits find zero exact or dHash<=4 duplicates. Two contact sheets plus full-resolution clipping risks show no confirmed severe glitch and broad content. Velvia50 is the third provisional unpaired `S0` pixel stock; learning remains forbidden until SF1.0 stock-versus-source/content/colour controls pass.

YFCC SF1.0A/A2, 2026-07-16: exact YFCC Ektar metadata passes 26/10 while YFCC UltraMax fails at 68.75% one-UID dominance. Live rights retain a 16-file/5-UID YFCC Ektar bridge with zero duplicates and no severe visual artifact. The connected graph is YFCC Ektar--Velvia plus Commons Ektar--UltraMax. The bridge is diagnostic only and learning remains forbidden until connected nuisance controls pass.

SF1.0B, 2026-07-16: both same-source stock edges fail. Commons RGB is 59.82%/p=.280 while 4x4 scene colour is 80.36%/p=.024. YFCC RGB is 65.71%/p=.188 while luma/HOG/geometry are 72--76% and 4x4 scene colour is 89.29%/p=.002. Same-stock Ektar source geometry is 92.86%/p=.016. Current pools are closed for stock learning; larger models are forbidden. A full-YFCC metadata-only shared-author gate is the sole data expansion.

FSA/OWI RF0.3 audit, 2026-07-15: the Commons mirror exposed 1,031 pages but strict filtering/canonicalization retained 558 public-domain LOC scan identifiers after removing 419 repeated representations. A 64-image/20.57MB visual pilot had zero decode, exact-duplicate or dHash<=4 failures and no confirmed severe glitch. Creator/location grouping leaves 457 learning-eligible records, 101 unknown-creator stress records and 43 conservative location guard groups. Phase C was stopped and retained at 258 downloaded derivatives / 81,016,399 bytes when the stock-first objective became authoritative. Training is not allowed; these are archive-scan leakage groups, not physical rolls or calibrated Kodachrome.

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
| Done/closed | U5.CT0 | Roll2Film algorithm hypothesis was correctly framed and tested; current paper route closes because BlueNeg matched-roll replication is ambiguous |
| Done | U5.CT1/U5.CT3/U5.CT6 | Fixed-budget method controls pass, but real-roll correct-group gain changes sign; no roll-information promotion |
| Done | U5.CT2/U5.CT4 | FilmSet and BlueNeg metadata/licence/access/whole-roll contracts pass |
| Done/limited | RF0.1/RF0.2 | FILM-R acquired and visually/integrity audited; use only as real-film unknown-look/nuisance evidence, never stock or clean-target truth |
| Done/data stop | RF1.1/RF1.2 | RF1.2 elevates retrieval as a challenger; RF1.1 proves FILM-R cannot separate family from content and forbids classifier training |
| Done/closed | RF1.4A | Preview route closed: GA/Konica structurally unidentified; NPH/Gold roll-null not rejected and nuisance shortcut wins |
| Done/pass | RF1.4B0 | Gold100 official alignment/support: 47 exact pairs across six rolls; no colour fit or stock-response claim |
| Done/limited pass | RF1.4B1 | All metric/visual gates pass for Gold archive preview-to-display mapping; bounded 3x3 affine beats SepLUT overall; no stock-response claim |
| Done/provisional S0 | SF0.8B | YFCC Velvia50 passes live rights, pixels, grouping, duplicate, content and vision gates with 25 files/14 UIDs; no stock-response claim |
| Done/closed current pools | SF1.0 | Connected diagnostics confirm source/content/scene-colour shortcuts; no operator fitting |
| Done/diagnostic bridge | SF1.0A/A2 | YFCC Ektar passes metadata/live pixel gates at 16 files/5 UIDs; YFCC UltraMax stops; connected design is available |
| Done/closed | SF1.0B | Both stock edges fail; Ektar source fingerprint is strong; current pools remain aesthetic/failure evidence only |
| P0 metadata only | SF1.1 | Freeze/download/filter the 65.64GB public YFCC100M SQLite index for same-author exact-stock support; no pixels |
| Done/closed | RF2.S0 | Direct Gold archive-matrix transplant passes OOD coverage but fails style, non-basic residual and clipping; do not add capacity |
| Done/metadata pass | SF0.4 | Commons froze 658 metadata rows; three exact-stock categories pass, with no pixels downloaded and no stock-signal claim |
| Done/one-stock pass | SF0.5 | 36 Commons derivatives are clean; Ektar passes source groups, Superia/Gold stop; learning remains forbidden |
| P0 named-stock sources | SF0.6 | Obtain at least two further exact stocks that independently pass derivative-rights, author-group and content gates |
| P0 historical auxiliary | RF0.3/RF1.3 | Complete bounded FSA/OWI pixels, deterministic border/content audit and creator/location holdouts as `historical-film/unknown-stock`; never count it as a named stock |
| Data-gated | RF2.S/RF2.H | Compare stock-specific CPU global/hierarchical/retrieval/conditional experts separately from the historical/unknown expert after their own RF1 gates |
| Conditional | RF3 | Train lightweight bounded GPU challengers only after data/identifiability gates pass |
| P0 | U0.2 | Owner decides repository license; add legal artifacts only after approval |
| Done/blocked | U0.3 | Legacy 4,210-row audit completed fail-closed with 0 eligible rows; new remote-data grouping contracts move to U5.CT2 |
| Done/ongoing | U0.4/U4 | CI/registry foundation is complete; maintain severe-artifact/style evaluation as Roll2Film support and product QA, not the primary paper |
| P0 | U1 | Connect `WorkingImage`, 16-bit/profile-aware I/O and color-state contract |
| P1 | U2 | Implement profile/recipe schema and deterministic reference renderer |
| Deferred | U3 | Controlled named-stock calibration remains unavailable; do not weaken real-film RF requirements |
| Done/auxiliary | U5.CT5–U5.CT8 | FilmSet one-shot complete: Cinema/ClassNeg pass digital recipe control, Velvia fidelity fails; no Ultimate promotion |
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

*Last updated: 2026-07-16 | Current implementation: deterministic content-safe renderer | Research target: stock-first bounded explicit operators | Product target: Style-safe deterministic/operator core | Calibrated lane deferred*
