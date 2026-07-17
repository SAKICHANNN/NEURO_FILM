# TASK_BOARD.md — Algorithm-first film colour-transfer and product path

> **Ultimate priority reset, 2026-07-15:** real-film scans now own the final
> success criterion. See `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md`.
> Specific `film_stock_id` experts are primary; historical/unknown-stock film
> is a separate auxiliary class and cannot replace or count as a named stock.
> See `docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`. FilmSet CT5/CT8 is an
> auxiliary digital-recipe control only.

> Updated 2026-07-15. Detailed DoR/DoD, dependencies, gates and evidence live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.
> SDXL/IP2P/SDEdit is a retired production direction and an optional research/Creative comparator only.

---

## Current truth

| Area | State | Evidence/next action |
|---|---|---|
| Deterministic renderer | current default | `safe_lab`/safe-rich + optional grain/halation/dust |
| Diffusion/IP2P | retired as default | detail/identity drift; tested SDXL full-UNet OOM on 12GB |
| Neural LUT/local maps | research-only | pseudo-teacher or saturation-gate evidence is insufficient |
| Roll2Film research | **ambiguous / not promoted** | BlueNeg correct-roll gains are `-0.220/+0.309`; the roll-cluster interval crosses zero, so CT7 ML/set inference stops |
| FARO/ChromaticTail | supporting evaluation/product wrapper | severe-artifact evaluation, fixed-policy audit and fallback; no standalone primary benchmark paper |
| FilmCase ML | conditional product/research branch | open only if strength-adjusted bank diversity and an applicability/safety Oracle gap exist; otherwise champion + bounded strength wins |
| Latent stock modes | hypothesis only; data-gated | no stock has proved `K>1`; LSM0 freezes semantics, LSM1 requires stock/connectivity/identifiability/rights gates, and `K=1` remains a formal branch |
| Input pipeline | float32 + fail-closed HDR pass | one float32 main path; HEIF/AVIF and recognized HDR/gain-map signals reject before silent SDR fallback |
| Product standard | decided | strongly stylized output with no severe glitch/artifact |
| Current local data | FilmSet, BlueNeg bounded pilot, stock-pilot v1, FILM-R and FSA/OWI pilot | stock-pilot v1: 189 hash+decode verified files; Gold display-candidate; NPH/Konica/GA post-negation-preview diagnostics only; FSA/OWI remains historical/unknown-stock |
| Named-stock coverage | **not established** | BlueNeg Kodak Gold is provisional single-stock evidence below transferable `S2`; no second stock is promoted |
| Historical/unknown coverage | one qualified auxiliary archive lane | LOC FSA/OWI is real historical film but never fills a named-stock slot |
| User contribution | not a current dependency | no new owner images, pairs, labels or votes are required for the no-data/FilmSet/BlueNeg stages |
| Stock accuracy | stock-first research active; calibration deferred | authoritative labels and unseen roll/source controls can support graded `S1/S2` real-film-derived experts; paired controlled evidence is required for `S3 calibrated-reference` |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | 233 CPU tests pass after U1.2A embedded-ICC fail-closed repair |

---

## Ready queue

| Order | Node | Task | Status | Approval |
|---:|---|---|---|---|
| 0 | U0.1 | Reconcile active README/status/docs; archive stale diffusion instructions | complete | completed 2026-07-10 |
| 0a | U5.FC0 | Freeze autonomous unpaired non-generative FilmCase plan | complete | completed 2026-07-11 |
| 0b | U5.R0 | Freeze FARO novelty audit and publication research program | complete | completed 2026-07-11 |
| 0c | U5.CT0 | Correct publication priority to Roll2Film colour-transfer algorithm and audit data/nearest work | complete | completed 2026-07-12 |
| 1 | U5.CT1 | Freeze invertible explicit-operator API and known-operator pseudo-roll simulator | L2 spline core passes; full CT1 still needs explicit L0/gauge/shaper closure | preserve analytic inverse/Jacobian and bake parity |
| 2 | U5.CT2/U5.CT4 | Freeze FilmSet pair blindness and BlueNeg metadata/licence/whole-roll acquisition | complete | BlueNeg full archive forbidden; exact bounded manifest only |
| 3 | U5.CT3 | Extend fixed-budget E0 from affine to L2 truth and stronger nuisance/prior/scanner factorials | L2 method controls pass; roll information not established | matched real-roll evidence remains |
| 4 | U5.CT5 | Build matched-strength deterministic/classical baselines and run FilmSet internal paired-blind development | complete on internal confirmatory: fixed Lab/pooled-L2 recipe bank passes all-238 full-res severe veto; final 628 remains sealed | frozen decision in `configs/roll2film_ct5_fullres_decision.json` |
| 6 | U5.CT6 | Run BlueNeg correct-roll matched-control pilot | complete: ambiguous across two held-out rolls | roll information not established |
| 7 | U5.CT7/U5.CT8 | Stop amortized set inference; close one-shot FilmSet auxiliary confirmation | complete: Cinema/ClassNeg pass digital recipe control; Velvia fidelity fails; no severe confirmed | cannot promote Ultimate or real-film claims |
| 1 | RF0.1/RF0.2 | Acquire, verify and gate CC-BY FILM-R real scans | complete/limited | 88 files verified; only `real-film-derived/unknown-look`; restoration is nuisance, not truth |
| 2 | RF1.1 | Test FILM-R signal vs content/damage/source nuisance | complete: unidentified | no comparable cross-content families; classifier correctly stopped |
| 3 | RF1.2 | Diagnose BlueNeg roll sign reversal without added capacity | complete: nuisance | wrong-roll content retrieval beats correct roll; physical-roll hypothesis stays closed |
| 1 | RF0.4/SF0.1 | Build registry and freeze exact four-pilot acquisition | complete | 189 files / 227,287,697 bytes; Gold 400-5 failed whole-test-roll preflight and was replaced by GA 100 5095 |
| 1a | SF0.2 | Download/hash-verify frozen stock pilot pixels in isolated root | complete | 189/189 size+LFS hashes; report SHA-256 `de286950...`; zero external files; no decode |
| 1b | SF0.3 | Decode integrity, duplicate/border/content and support-matrix audits | complete | 189 RGB PNG; Gold display-candidate; NPH/Konica/GA preview-identifiability only; 8/8 contact sheets reviewed |
| 1c | RF1.4A | Run preview leave-one-roll-out shortcut/null gates | complete: closed | GA/Konica structural fail; NPH/Gold p=0.191, simple-global ties, nuisance 10/11 |
| 1d | RF1.4B0 | Gold100 official bbox/proxy alignment gate | complete: pass | 47/47 pairs, six rolls, exact dimensions, restricted pickle, no colour fit |
| 1e | RF1.4B1 | Gold100 display-proxy paired-transform identifiability | complete: metric+visual pass | 47 pairs/6 rolls; all frozen gates pass; 3x3 affine 5.307 beats SepLUT 5.380, so nonlinear capacity is not selected |
| 1f | RF2.S0 | Gold archive-display matrix transplant falsification | complete: closed | OOD passes; style/residual/clipping fail at all fixed strengths; direct archive-matrix transplant is bland/basic and unsafe |
| 1g | SF0.4 | Commons multi-source named-stock metadata audit | complete: pass | 3 exact stocks/317 rows pass; 341-row Velvia remains family-only; zero pixels |
| 1h | SF0.5 | Commons permissive derivative pilot | complete: one-stock pass | 36 clean derivatives; only Ektar 26/8 authors passes; Superia/Gold stop; no training |
| 1i | SF0.6A | Exact-stock source expansion | complete: UltraMax metadata pass | 393 rows/10 stocks/zero pixels; UltraMax 84 files/15 authors/51 strict rows; need one more stock |
| 1j | SF0.6B | Third-stock source sweep | complete: Kodachrome64 pass | 51 files/22 authors/15 strict rows; five stocks stop; zero pixels |
| 1k | SF0.7 | UltraMax + Kodachrome64 pixel audit | complete: UltraMax pass, Kodachrome stop | 51 clean files; UltraMax 37/8 authors; Kodachrome 14/2 authors/85.71% |
| 1l | SF0.8A | YFCC15M stock metadata | complete | 7.35M rows; repeat-identical exact-text audit; Velvia50 51/26 UID raw, 45/23 after prospective contamination exclusions |
| 1m | SF0.8B | Velvia50 third pixel stock | complete: provisional S0 | 25 files/14 UIDs; live rights, duplicates, source gate and vision pass; no stock-response claim |
| 1n | SF1.0A | YFCC same-source support | complete | Ektar 26/10 passes; UltraMax 16/5 but 68.75% dominance fails; connected bridge possible |
| 1o | SF1.0A2 | YFCC Ektar pixel bridge | complete: diagnostic | 16 files/5 UIDs; live rights, zero duplicates and vision pass; exact minimum group warning |
| 1p | SF1.0B | Connected-design nuisance identifiability | complete: closed | RGB fails; scene colour/content/source controls dominate; no operator fitting |
| 1q | SF1.1 | Full YFCC shared-author metadata | complete: one metadata edge passes | 65.64GB SHA/S3 verified; repeat-identical scans; Ektar/Velvia 16 shared UIDs; no pixels |
| 1q2 | SF1.2 | Shared-author live-rights feasibility | complete: pass | 61 HTML pages; eight bilateral CC BY authors; no image requests |
| 1q3 | SF1.3A | Bounded shared-author pixel/integrity pilot | complete: pass | 37 clean pixels; eight bilateral UIDs; zero duplicates; no confirmed severe artifact |
| 1q4 | SF1.3B | Shared-author stock-identifiability diagnostic | complete: closed | RGB 56.25%/p=.464; best nuisance 68.75%; pool closed for fitting/training/LSM |
| 1q5 | SF2.0A | Apollo 7 SO-368/SO-121 stock-magazine metadata feasibility | complete: content-confounded close | 63/63 valid pages; support/filter bridge pass, only one of two shared content tags passes; no image request/fitting/training/LSM |
| 1q6 | SF2.0B0 | NASA/JSC STS098 VELVI/5775/5776 exact-code metadata snapshot | complete: insufficient-roll close | rows 168/329/10, but Velvia has only 2 rolls below frozen 4; no B1/photo/images/fitting/training/LSM |
| 1q7 | SF2.0C0 | NASA/JSC cross-mission exact-code roll-connectivity census | complete: no-candidate close | 25 missions audited; only STS098 joins both primaries and Velvia remains 2 supported rolls below frozen 4; no photos/pixels/fitting/training/LSM |
| 1q8 | SF2.1A | Openverse exact-stock shared-creator metadata graph | complete: contract-mismatch close | 960 rows; Ektar has 19 repeated identities, only UltraMax passes per-stock gate; no live pages/pixels/fitting/training/LSM |
| 1r | LSM0 | Freeze within-stock latent-mode ontology and epistemic contract | complete after propagation | hypothesis only; observed/latent ledgers separate; no stock has proved multiple modes |
| 1s | LSM1 | Build per-stock data/connectivity feasibility decisions | data-gated; current pools ineligible | no clustering/training; SF1.1 remains an unchanged upstream metadata gate |
| 1t | LSM2-LSM8 | Residual/operator identifiability, mode existence, fixed bank, Oracle, routing and product validation | conditional | requires full stock/connectivity/identifiability/pixel-rights pass; K=1/no Oracle/severe artifact closes routing |
| 2 | RF0.3/RF1.3 | Preserve bounded FSA/OWI historical/unknown lane | sealed partial, auxiliary | 258 derivatives / 81,016,399 bytes retained; never count as named-stock coverage or block RF0.4 |
| 4 | RF2.S/RF3 | Per-stock CPU expert ladder, then bounded GPU challengers if justified | data-gated | requires that stock's RF1.4 pass; no cross-stock averaging or RGB generator |
| 4a | RF2.H | Independent historical/unknown CPU expert | data-gated on RF1.3 | separate coverage ledger and claim class |
| 8 | U5.CT9 | Optional controlled named-stock calibration | future | new capture/lab scope and approval |
| 9 | U5.R0T | Audit theorem-level gap in FARO certification | deferred optional support | not a primary paper dependency |
| 10 | U0.3 | Legacy manifest-v2 lineage audit | complete; reference-derived lane blocked | 4,210/4,210 quarantined; new public-data contracts live under U5.CT2 |
| 11 | U0.4 | CPU-safe CI, environment capture and explicit non-gold registry | complete | current U4 evaluation assets remain supporting work |
| 12 | U5.R1A/U4.1/U4.2 | Maintain chromatic ontology/look rubric and stress evidence as CT evaluation support | ready support | none for local schema/tooling; participants later require approval |
| 13 | U1.1/U1.3B | Make `WorkingImage` the only render ingress and remove early sRGB8 quantization | complete | one float32 main path; exact sRGB8 colour parity, <=1-code effect parity and real RAW audit pass |
| 14 | U1.2–U1.6/U2.5A | Color-state contract, 16-bit/ICC export, HDR/HEIF, B&W safety and tiled foundation | U1.6A pass; U1.6B frozen/ready | implement safe-Lab two-pass context/dither parity without CLI integration; effects, streaming/total memory, HDR/HEIF/wide-gamut and calibrated mapping remain pending |
| 15 | U2.1–U2.5 | Profile/recipe schema and deterministic reference renderer | U2.1A complete; U2.1 in progress | strict v1 profile/recipe, exact migration and hash verifier pass; broader API/operator schemas pending; no current profile calibrated |
| 16 | U5.R2–U5.R7/U5.FC1–U5.FC8/U6 | FARO/FilmCase baselines, product fallback and artifact-safe effects | supporting/conditional | GPU/cost and participant gates only if later needed |
| 17 | U3.1–U3.4 | Optional Portra 400 + Velvia 50 calibrated profile lane | deferred | not an active user ask or dependency |
| 18 | U7/U8 | Product, beta, release and stock expansion | pending | release/legal approval |

---

## Human decisions required

1. Confirm intended code/profile/data release model and repository license.
2. Approve paid GPU, external participant contact, merge or public release before it occurs. Necessary research-data downloads and pushes are pre-authorized for the active autonomous goal.

Still-photo desktop is the default first contract. Paired capture and external human validation remain deferred rather than repeatedly requested.

---

## Protected current work

Do not touch or stage the existing user deletion of `data/raw/.gitkeep`; it is
outside this research-document leaf.

The active autonomous goal explicitly authorizes scoped commits and pushes. Preserve unrelated user work and never merge or release without a separate gate.

---

## Success gate

The deterministic product route advances independently. The research route is
stock-first: each concrete stock must beat pooled, wrong-stock, generic
historical, retrieval, shuffled-label and matched simple-enhancement controls
while preserving a frozen style floor and artifact ceiling. Coverage,
nuisance-capacity, scanner/process/source/content and shortcut tests are
mandatory. Failure closes or downgrades that stock branch; a generic old-film
expert cannot replace it. External population preference remains approval-gated, and
stock-specific `S1/S2` claims require authoritative labels and independent
roll/source controls; `S3 calibrated-reference` requires controlled paired
whole-roll evidence. Historical/unknown-film evidence never counts toward a
named-stock milestone.

---

*Active parent: `ULT > RF stock-first real-film mainline` | SF1.3B is closed and LSM1 remains ineligible; U1.3B and U2.5A product-safety leaves are complete; the next local product leaf returns to remaining U1.2/U1.4/U1.5 colour-state and format gaps; FSA/OWI remains sealed auxiliary | Integration owner: repository owner or explicitly assigned root agent*
