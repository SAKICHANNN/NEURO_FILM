# AGENTS.md — K-MCFM Project Knowledge Base

> **Current truth: 2026-07-17.** The production-capable path is a deterministic content-safe color renderer plus procedural effects. Ultimate success now requires stock-first learning from verifiable real photographic-film scans; specific `film_stock_id` experts are primary, while historical/unknown-stock film is a separate auxiliary class. FilmSet/Capture One, camera simulations, LUTs and pseudo-teachers are controls only.
> **Target direction:** compare stock-specific global, hierarchical, retrieval and bounded conditional explicit operators. A stock may conditionally expose latent `Mode A/B/C`, but multi-mode structure is an unproved, data-gated hypothesis and `K=1` remains a formal outcome. Roll, process, scanner, source and content are nested nuisance/group variables. The physical-roll-only Roll2Film hypothesis remains closed under current BlueNeg evidence. Unpaired digital-to-film operator identification remains unresolved. GPU models may predict bounded parameters but never directly generate final RGB. See `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md`, `docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md` and `docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md`.

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
| Current evidence | 716 local tests pass. U1.2B rejects real raster transparency instead of falsely claiming discarded alpha is preserved; fully opaque alpha strips with an explicit warning. U1.2C rejects multi-frame/page raster inputs instead of silently taking frame zero. U1.4A adds validated no-clamp D65 linear-sRGB/linear-Rec.2020 math with `2.38e-7` extended float32 roundtrip error. U1.4B adds deterministic BT.2020 SDR RGB16 PNG CICP ingress/egress with exact samples and `1.44e-5` linear roundtrip error. U1.4C1A/B add working-space D65 Lab primitives and one pure Lab look kernel while preserving both frozen hashes and eight-style full/tiled parity. C1C synthetic gamut/wide-colour gates pass, but C2 real-image OOD closes the operator: 66/96 renders exceed the frozen 0.5% new Rec.2020 boundary gate, worst 6.7808%; visual review was forbidden and no integration opens. U1.5A/B now reject recognized HDR/gain-map signals through bounded JPEG APP/PNG text traversal rather than relying only on file-edge sampling; U1.5C additionally pins two exact libultrahdr MPO gain-map references that fail closed through the existing multi-frame boundary without a source change. This is not complete ISO 21496-1 or HDR support. HDR/ACES and arbitrary profiles remain open. U1.6A-D/F prove finite-support execution, byte-identical two-pass safe-Lab, simple-halation, sparse dust and exact temporary-disk-staged legacy grain; U1.6G1/G2 add exact shape-stable global grids and float32 streaming percentiles, G3 freezes faithful colour/density DAG lifetimes, and G4A proves byte-exact coordinate gradient windows. G4B closes batch-sensitive G1-v1 row staging; G4C passes a separately versioned explicit-float32 reducer, and G4D binds its real providers. G4E fails one frozen legacy gate; G4F retains v2 only as a safe/non-empty opt-in research candidate. G4G measures repeat-identical 24MP execution/composite at 0.963-0.983GiB peak. G4H adds an isolated direct-module adapter with 36/36 parity/lifetime policies and zero production imports/schema changes. G4I measures two repeat-identical local 100MP adapter runs at 4.066-4.072GB process-tree RSS and 123.8-131.8s worker time; complete renderer and production integration remain closed. U1.6E/G0 close invalid shortcuts. Physical halation and complete renderer remain open. U2.1A provides verified replay identity; U2.6A exposes the existing evidence ledger through deterministic validated read-only API/CLI, with safe-rich still none/none/heuristic/non-calibrated. U2.5B adds an explicit profile-driven safe-Lab path with exact current-style parity and pre-output fail-closed provenance validation while leaving the default unchanged. U2.4A adds only an isolated synthetic-test interpretation boundary; real negative/slide/B&W operators and production integration remain absent. SF2.3 finds only Ektar100/UltraMax400 individually eligible and one shared author, so the Commons union has zero redundant edges. SF2.4R finds a technically strong 385-stock Newgrain catalogue and group/process schema, but published Terms prohibit automated querying/scraping/mining; the source closes before any retained audit or pixel request. RF2.C0 retains one Ektar100/fixed-e0 spektrafilm Look Approximation as an external comparison control (style 8.03, non-basic residual 7.34, 0% new clipping, no severe visual failure); auto exposure spans roughly 0.2x--4.2x scene luma and is negative adaptation evidence, not stock distinction. U5.R1A now provides strict FilmStyleSafe annotation, VLM/human isolation, conservative 3+3 adjudication, hidden-split leakage checks and non-binding power tooling; all current U4/RF2.C0 evidence is A0-only. U5.R1C3 affine weak-passes at 2/3 and quadratic at 1/3 with 53/55/56 retained; R1C3D spatial cross-fit remains 2/3, so the conditioned-SCIS route closes with no detector or gate. Current pixels remain closed for learning; training, operator fitting and LSM remain forbidden |
| Latest algorithm evidence | 1008 local tests pass. U5.R2AC1's independent shared-resource diffusion policy is deterministic and structurally clean (exact identity/neutral/repeat, bounded range, no ringing/crosstalk, finite support), but its strongest local effect over the global reaction control is `.0019468`, just below the frozen `.002` floor; parameter rescue, AC2, photographs and integration close. U5.R2AB1 separately finds strong historical DoRF responses but closes their bank because the closest survivors are near-duplicates (`.001016 < .02`). U5.R2AA1 closes the Kodak datasheet chain as strong/non-basic but nuisance-unidentified. |
| Latest source evidence | 1004 local tests pass. U5.R2AC0 retains Filmulator only as a generic shared-developer spatial-mechanism prior: three colour layers share developer through reaction, diffusion, reservoir exchange and agitation. The audited source is GPLv3-or-later while K-MCFM has no decided root licence, so copying, porting, linking and compatibility claims are blocked; only a new synthetic first-principles contract may open, with no stock claim. U5.R2AB0 separately retains DoRF as an internal historical response prior, but AB1 closes a response bank for insufficient diversity. |
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
| 6 | `docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md` | Conditional within-stock mode hypothesis, K=1 branch and LSM0-LSM8 gates |
| 7 | `docs/data/LATENT_STOCK_MODE_EVIDENCE_REGISTRY.md` | Strictly separated observed-evidence and latent-inference ledgers |
| 8 | `docs/planning/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_INTEGRATION_20260715.md` | Prior evidence adjudication and product/research split |
| 9 | `docs/planning/FARO_RESEARCH_PROGRAM_2026.md` | Supporting artifact evaluation and product/system-risk program; historical paper priority is superseded |
| 10 | `docs/planning/ULTIMATE_ROADMAP_2026.md` | Earlier strategic synthesis, product architecture and primary sources |
| 11 | `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md` | FilmCase baseline hypotheses and ablations, subordinate to diversity/Oracle gates |
| 12 | `TASK_BOARD.md` | Compact active board/pointer |
| 13 | `IMPL_PLAN.md` | Active-plan pointer plus historical V3 plan |
| 14 | `docs/CURRENT_STATUS_2026-05-27.md` | Diffusion/IP2P failure and deterministic pivot |
| 15 | `docs/CONTENT_PRESERVING_RENDERER_FINAL_REPORT.md` | Current renderer implementation and promoted effects |
| 16 | `docs/PROJECT_STRUCTURE.md` | Repository placement and safe cleanup rules |
| 17 | `docs/EXPERIMENT_LOG.md` | Historical experiments |

The following are historical context, not active authority: `docs/ARCH_REDESIGN.md`, `docs/planning/GAP_ANALYSIS.md`, `docs/ONLINE_DATA_AUDIT.md`, and the diffusion sections below the supersession banner in `IMPL_PLAN.md`.

---

## 3. Current implementation

```text
WorkingImage linear-sRGB input
  -> explicit float32 sRGB look adapter
  -> deterministic float32 CIELAB mean/std transfer (`safe_lab`, safe-rich guards)
  -> optional deterministic grain / physical-inspired halation / dust
  -> bounded float output
  -> one final quantization to sRGB8 PNG/JPEG/TIFF or opt-in sRGB16 PNG/TIFF with embedded ICC
```

Current strengths:

- fixed-grid deterministic color transform; no generative geometry rewrite;
- stable safe-rich guardrails and fixed test artifacts;
- strict opt-in v1 profile/recipe identity with asset/input/output hash verification;
- exact finite-support halo-aware tiled execution primitive with bounded callback windows;
- experimental two-pass safe-Lab tiling with full-frame context and coordinate-exact legacy dither;
- experimental finite-support simple-halation tiled composite with derived halo;
- exact neutral-axis output for current HP5/Tri-X B&W profiles;
- modular grain/halation/dust layers;
- `WorkingImage`, ICC-aware raster helpers and generic RAW decoding exist under `src/preprocess/`;
- substantial experiment logs and comparison artifacts are preserved.

Current limitations:

- `scripts/render_film.py` now uses one float32 safe-Lab/effect path for both output depths, but remains SDR/sRGB;
- generic RAW enters as explicit linear-sRGB/scene-linear and renders end to end, but still lacks a calibrated scene-to-display tone map;
- no complete HEIF/HDR/gain-map or wide-gamut production path; recognized unsupported inputs now fail closed instead of silently decoding an SDR base;
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
       stock global deterministic fallback
       or, only after LSM gates, one hard latent mode + bounded case expert
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
15. `H-LSM-1` is a hypothesis: an evidence-backed stock may have multiple stable latent modes, but no current stock has proved `K>1`; `K=1` closes latent routing and keeps global champion plus bounded strength.
16. Mode discovery is forbidden until stock evidence, connectivity, stock identifiability, pixel/rights and leakage gates pass. Raw appearance, CLIP/content, uploader, geometry or scanner/source clusters are not film modes.
17. Without independent metadata, latent modes use `Mode A/B/C`; exposure, EI, illuminant, push/pull, process and scanner interpretations remain `unknown` or `hypothesis_only` and never backfill observed metadata.
18. Mode-space evidence and within-mode content retrieval use separate representations. Unpaired operator candidates remain `film-inspired/unpaired-evidence`, not identified digital-to-film or calibrated stock response.

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
- Latent stock modes: conditional LSM0-LSM8 hypothesis under stock-first. Current pools are not eligible; `K=1`, strength-only, content/source nuisance, no Oracle gain or severe artifacts all close routing without stopping Ultimate;
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
| Newgrain | terms-blocked source reconnaissance | public catalogue/schema facts only | automated metadata/pixel acquisition, shared-user graph, fitting or training without written permission |

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

SF1.1, 2026-07-16: the exact 65,644,027,904-byte YFCC100M SQLite passes SHA-256 and its 7,826-part S3 ETag. Two full scans are byte-identical at SHA-256 `19fd20b3...`; Ektar100/Velvia50 passes metadata connectivity with 780/240 rows, 122/62 UIDs and 16 shared UIDs. Ektar/UltraMax fails because UltraMax has only 27 UIDs. This opens SF1.2 bounded page-only live-rights feasibility only; image payloads, operator fitting, training, LSM and stock claims remain forbidden.

SF1.2, 2026-07-16: 61 sequential bounded HTML requests confirm eight shared UIDs with a current CC BY 2.0 page for both Ektar100 and Velvia50, passing the frozen five-author gate. No image URL was requested. This opens only the separately frozen 38-candidate/512MiB SF1.3A pixel and integrity pilot; operator fitting, training and LSM remain forbidden.

SF1.3A, 2026-07-16: 37 bounded derivatives pass hash/decode/dimension, exact/dHash<=4 duplicate and autonomous visual gates: 21 Ektar100, 16 Velvia50 and all eight bilateral UIDs. Content remains heterogeneous/unbalanced and this is not stock evidence. Only the preregistered leave-one-UID-out SF1.3B diagnostic opens; training, operator fitting and LSM remain forbidden.

SF1.3B, 2026-07-16: leave-one-UID-out global RGB is 56.25%/p=.464; standardized RGB and 4x4 scene colour each reach 68.75%, while geometry reaches 62.5%. RGB minus best nuisance is -12.5 points with 95% CI [-31.25%, 0]. The shared-author pool is unidentified and closed for learning; do not add model capacity, fit operators or cluster modes.

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
| Done/metadata pass | SF1.1 | Full YFCC Ektar/Velvia shared-author gate passes with 16 UIDs; exact source integrity and repeat audit complete; no pixels |
| Done/rights pass | SF1.2 | Eight shared authors have a current CC BY page for both stocks; 61 HTML requests, no image access |
| Done/pixel pass | SF1.3A | 37 clean bounded Ektar/Velvia derivatives, eight bilateral UIDs, zero duplicate pairs, no confirmed severe artifact; no fitting/training |
| Done/closed | SF1.3B | Shared-author Ektar/Velvia RGB fails held-out-UID identifiability and loses to nuisance controls; pool closed for learning |
| Done/closed | SF2.0A | 63/63 Apollo 7 metadata pages validate and support/filter bridge pass, but only one of two required shared content tags passes; content-confounded, no pixels/fitting/training/LSM |
| Done/closed | SF2.0B0 | STS098 exact-code rows pass at 168/329/10, but Velvia has only two rolls below frozen four-roll minimum; insufficient support, no B1/photo/images/fitting/training/LSM |
| Done/closed | SF2.0C0 | Cross-mission exact-code census finds 25 missions but only STS098 joins Velvia50 and Portra400NC; no candidate mission, no photos/pixels/fitting/training/LSM |
| Done/limited pass | SF2.9R | Five multi-family IT8 reference archives pass exact integrity/parse and 288-patch spectral support, but calibrated-target manufacture and missing common recorder input close stock operator fitting; physical/nuisance evidence only |
| Done/closed | SF2.1A | Openverse returns 960 metadata rows, but Ektar pagination repeats 19 identities and only UltraMax passes per-stock support; no live pages/pixels/fitting/training/LSM |
| Done/closed | SF2.2R | Smithsonian fixed-shard reconnaissance finds family-level Kodachrome/Ektachrome evidence split across owning units, not an exact connected stock graph; reject 6.64GB full metadata transfer, no pixels/fitting/training/LSM |
| Done/closed | SF2.3 | Three immutable Commons snapshots yield 294 strict rows but only Ektar100/UltraMax400 pass per-stock support and share one author; zero two-author edges, no live preflight/pixels/fitting/training/LSM |
| Done/pass | U2.1A | Strict v1 profile/recipe schemas, exact safe-rich migration, deterministic ICC bytes and committed replay verification pass; no calibrated profile claim |
| Done/boundary pass | U2.4A | Isolated interpretation request/plugin/result boundary passes with three synthetic-test-only witnesses; real operators, profiles and renderer integration remain pending |
| Done/pass | U2.5/U2.5B | Explicit profile-driven safe-Lab adapter exactly matches current legacy styles and validates provenance before output; default and schemas remain unchanged |
| Done/pass | U2.6/U2.6A | Validated deterministic read-only API/CLI expose declared profile evidence; safe-rich remains none/none/heuristic/non-calibrated and no schema or renderer changed |
| Done/pass | U1.6A | Finite-support tiled primitive has exact coverage, zero committed Gaussian full/seam error and bounded expanded windows; complete renderer/global context/100MP remain open |
| Done/pass | U1.6B | All eight safe-rich styles and a fixed real-raster crop have byte-identical full/tiled safe-Lab output; effects, integration, streaming and total-memory/100MP remain open |
| Done/pass | U1.6C | Active simple-halation real-raster composite has `5.96e-08` full/seam error and sRGB8 byte parity; physical halation, integration and 100MP remain open |
| Done/pass | U1.6D | Compact zero-halo event context reproduces legacy dust/scratch float and sRGB8 bytes exactly; stress visuals remain conspicuously heuristic, so no realism/default/grain/100MP claim opens |
| Done/closed | U1.6E | Simple-halation screen and white-alpha dust commute algebraically; reversed output differs only `1.19e-07`, inside the frozen `1e-6` equivalence gate, so no combined adapter is retained |
| Done/pass | U1.6F | Exact colour/B&W legacy grain uses private two-field memmap staging with byte parity and cleanup; 0.018 smoke passes mechanics while 0.35 stress is visually rejected; no physical/default/100MP claim |
| Done/closed | U1.6G0 | Physical/density halation is not finite-halo local: sigma-52 full-grid resample differs by `5.81e-4` even with halo 156; require percentile/global-resample/field-DAG primitives before staging |
| Done/pass | U1.6G1 | Versioned original-grid area/downsample/direct-blur/bilinear staging gives byte-identical full/tiled sigma-52 output with zero max/seam error at tile 37/64; no effect integration, legacy parity, physical or 100MP claim |
| Done/pass | U1.6G2 | Exact two-pass float32 radix percentiles match NumPy-linear bytes on the million-value formal stream and 49,793 random finite bit patterns with bounded histogram memory; no effect integration or 100MP claim |
| Done/static-plan pass | U1.6G3 | AST-backed colour/density DAGs preserve 8/2 and 6/1 inventories, classify three global fields per family and audit transitive lifetimes/resources; integration remains closed pending exact gradient windows and row-chunked global staging |
| Done/pass | U1.6G4A | One-pixel original-coordinate scalar windows reproduce full NumPy gradient bytes across tiles, rows, edges and corners; integration remains closed pending row-chunked global staging |
| Done/closed | U1.6G4B | Bounded row reads work, but G1-v1 coarse/reconstruction bytes depend on row-batch height through `tensordot`; do not rewrite v1 or select lucky chunks; open a separately versioned deterministic reducer |
| Done/numerical-field pass | U1.6G4C | Explicit increasing-index float32 reduction and exact terminal bounds reproduce full/row/tiled bytes for five frozen fields; limited real-field visual gate passes, but no effect or 100MP claim opens |
| Done/static binding pass | U1.6G4D | G4A/G4C exported providers resolve exactly to G3 requirements; both family plans become statically ready without node/lifetime/resource drift; executor remains absent |
| Done/staged-v2 pass, legacy promotion closed | U1.6G4E | Bounded default density executor matches its v2 reference with zero seams and sRGB8 parity; one frozen legacy alpha gate fails, so retain research-only and do not integrate/default-promote |
| Done/opt-in research retention pass | U1.6G4F | Fresh `07/12/16` pass non-empty/selectivity/clipping/runtime and zero-severe gates; exactly 2/3 show plausible unamplified localized effect; integration/default/legacy claims remain closed |
| Done/local 24MP measured pass | U1.6G4G | Two child runs repeat hashes at 0.963-0.983GiB peak and 28.6-29.7s; failure cleanup passes; complete renderer, integration and 100MP remain open |
| Done/isolated adapter pass | U1.6G4H | 36/36 exact compositor-parity/lifetime policies pass; private layer, failure propagation and zero production imports/schema drift verified; no renderer/CLI/100MP claim |
| Done/local 100MP adapter pass | U1.6G4I | Two exact 100MP runs repeat source/output/metadata hashes at 4.066-4.072GB peak RSS and 123.8-131.8s; failure cleanup passes; no complete-renderer/production claim |
| Done ontology / data-gated | LSM0/LSM1 | Freeze observed-vs-latent semantics, K=1 and conditional feasibility matrix; no current stock is mode-study eligible and no clustering/training opens |
| Done/closed | RF2.S0 | Direct Gold archive-matrix transplant passes OOD coverage but fails style, non-basic residual and clipping; do not add capacity |
| Done/external control only | RF2.C0 | Isolated spektrafilm Ektar100/fixed-e0 passes automatic and 3-round/9-image visual gates as a stylised non-basic Look Approximation; no external code/profile/LUT integration, teacher, stock truth or calibration claim |
| Done/support tooling | U5.R1A | Strict FilmStyleSafe colour/style/structural ontology, 3+3 label aggregation, VLM/human isolation, A0/A1/B0-B4 leakage audit and non-binding power worksheet pass; R1B design only opens |
| Done/closed | U5.R1C3/R1C3D | Pairwise affine reaches 2/3, quadratic 1/3 and fixed spatial cross-fit remains 2/3 with 53/55/56 retained; current conditioned-SCIS closes with no detector/gate or further capacity/tuning |
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

*Last updated: 2026-07-17 | Current implementation: deterministic content-safe renderer | Research target: stock-first bounded explicit operators | Product target: Style-safe deterministic/operator core | Calibrated lane deferred*
