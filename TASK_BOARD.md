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
| Input pipeline | float32 + strengthened fail-closed HDR pass | one float32 main path; HEIF/AVIF and recognized HDR/gain-map signals reject before silent SDR fallback; bounded JPEG APP/PNG text traversal closes the prior edge-sampling blind spot |
| Product standard | decided | strongly stylized output with no severe glitch/artifact |
| Current local data | FilmSet, BlueNeg bounded pilot, stock-pilot v1, FILM-R and FSA/OWI pilot | stock-pilot v1: 189 hash+decode verified files; Gold display-candidate; NPH/Konica/GA post-negation-preview diagnostics only; FSA/OWI remains historical/unknown-stock |
| Named-stock coverage | **not established** | BlueNeg Kodak Gold is provisional single-stock evidence below transferable `S2`; no second stock is promoted |
| Historical/unknown coverage | one qualified auxiliary archive lane | LOC FSA/OWI is real historical film but never fills a named-stock slot |
| User contribution | not a current dependency | no new owner images, pairs, labels or votes are required for the no-data/FilmSet/BlueNeg stages |
| Stock accuracy | stock-first research active; calibration deferred | authoritative labels and unseen roll/source controls can support graded `S1/S2` real-film-derived experts; paired controlled evidence is required for `S3 calibrated-reference` |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | 584 CPU tests pass through U1.4B; validated Rec.2020 math and isolated BT.2020 SDR PNG boundary added without renderer claim expansion |

---

## Ready queue

| Order | Node | Task | Status | Approval |
|---:|---|---|---|---|
| 0 | U0.1 | Reconcile active README/status/docs; archive stale diffusion instructions | complete | completed 2026-07-10 |
| 0a | U5.FC0 | Freeze autonomous unpaired non-generative FilmCase plan | complete | completed 2026-07-11 |
| 0b | U5.R0 | Freeze FARO novelty audit and publication research program | complete | completed 2026-07-11 |
| 0c | U5.CT0 | Correct publication priority to Roll2Film colour-transfer algorithm and audit data/nearest work | complete | completed 2026-07-12 |
| 1 | U5.CT1/CT1C | Freeze invertible explicit-operator API and known-operator pseudo-roll simulator | analytic primitives pass; HDR shaped-bake parity closed | L0/gauge/shaper retained; 33/65 max errors .0351/.00899 fail; no gate relaxation or integration |
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
| 1q9 | SF2.2R | Smithsonian institutional stock-metadata reconnaissance | complete: no-DoR close | seven-unit fixed shard probes find disconnected family-level stock text, not an exact connected graph; 6.64GB full transfer rejected; no pixels/fitting/training/LSM |
| 1q10 | SF2.3 | Frozen Commons-snapshot union shared-author connectivity | complete: insufficient-connectivity close | 294 strict rows; only Ektar/UltraMax eligible and share one author, leaving zero two-author edges; no live preflight/pixels/fitting/training/LSM |
| 1q11 | SF2.4R | Newgrain exact-stock/shared-photographer source reconnaissance | complete: terms-blocked no-DoR | 385-stock public catalogue and rich group/process schema are promising, but published Terms prohibit automated queries/scraping/mining; stopped before a formal audit, retained snapshot or image access |
| 1q12 | SF2.5R | PROV digitised-negative/film-stock-register source reconnaissance | complete: physical-register data stop | 6,716 digital collection items exist, but 30/30 linked register volumes are physical-only with no digital/IIIF fields; no images requested; reopen only on register transcription/digitisation |
| 1q12 | RF2.C0 | Isolated spektrafilm external spectral control | complete: one external control only | Ektar100/fixed-e0 passes style/non-basic/clipping and 3-round visual gates; auto exposure is unstable nuisance evidence; no integration, teacher, stock or calibration claim |
| 1r | LSM0 | Freeze within-stock latent-mode ontology and epistemic contract | complete after propagation | hypothesis only; observed/latent ledgers separate; no stock has proved multiple modes |
| 1s | LSM1 | Build per-stock data/connectivity feasibility decisions | data-gated; current pools ineligible | no clustering/training; SF1.1 remains an unchanged upstream metadata gate |
| 1t | LSM2-LSM8 | Residual/operator identifiability, mode existence, fixed bank, Oracle, routing and product validation | conditional | requires full stock/connectivity/identifiability/pixel-rights pass; K=1/no Oracle/severe artifact closes routing |
| 1u | U1.6G4B/U1.6G4C | Close batch-sensitive G1-v1 row staging; validate a new deterministic reducer | G4B closed; G4C numerical/field pass | v1 preserved; five fields pass cross-chunk/tile/repeat and limited visual compatibility; no effect integration |
| 1v | U1.6G4D | Bind G4A/G4C capabilities | complete: static pass | exact providers resolve G3; no graph execution or resource drift |
| 1w | U1.6G4E | Freeze and implement the simplest staged density executor | complete: v2 pass / legacy promotion closed | exact/seam-safe v2 execution; `u41-14` legacy alpha max gate fails; retain unconnected research module |
| 1x | U1.6G4F | Audit explicit new-operator product value/safety | complete: opt-in research retention pass | 07/12/16 automatic pass; 2/3 localized visual pass; zero severe; legacy/default/integration claims closed |
| 1y | U1.6G4G | Measure 24MP memory/runtime and orchestration readiness | complete: local measured pass | 0.963-0.983GiB peak; repeat hashes/cleanup pass; no renderer integration or 100MP claim |
| 1z | U1.6G4H | Research adapter/orchestration parity and lifetime contract | complete: isolated adapter pass | 36/36 parity policies, ownership/failure/static isolation pass; no CLI/default/100MP claim |
| 1aa | U1.6G4I | Measure isolated adapter at 100MP | complete: local adapter pass | two exact runs at 4.066-4.072GB and 123.8-131.8s; hashes/cleanup pass; no complete renderer/integration claim |
| 1ab | U1.2B | Make raster alpha ingress truthful | complete: fail-closed pass | RGBA/LA/palette transparency rejects; opaque alpha exact-equivalent strip; no matte/preservation claim |
| 1ac | U1.2C | Make multi-frame raster ingress truthful | complete: fail-closed pass | GIF/TIFF multi-frame rejects before decode and renderer emits no output; no animation claim |
| 1ad | U1.4A | Establish validated wide-gamut math primitive | complete: primitive pass | matrix/roundtrip/neutral/provenance gates pass; no renderer/HDR/ACES/file-output claim |
| 1ae | U1.4B | Add one standards-backed wide-gamut file boundary | complete: file-boundary pass | exact deterministic RGB16 PNG CICP and roundtrip/fail-closed gates pass; no production/HDR/arbitrary-profile claim |
| 1af | U1.4C0-C2 | Make the colour operator working-space aware | first operator closed at C2 automatic gate | C1 math/legacy parity retained; real-image OOD has 66/96 over 0.5% new Rec.2020 boundary gate, so no visual run or integration |
| 1ag | U1.5C | Pin real libultrahdr gain-map ingress regressions | complete: real-reference pass | two exact CC-BY-4.0 MPO fixtures reject before working pixels and renderer output through U1.2C; 53 focused/652 full tests; no duplicate parser or broad HDR claim |
| 1ah | U2.6A | Expose existing profile evidence labels through validated read-only API/CLI | complete: pass | safe-rich reports none/none/heuristic/non-calibrated; deterministic/fail-closed gates and 656 tests pass; no schema/render/evidence-grade change |
| 1ai | U2.5B | Add explicit profile-driven safe-Lab compatibility adapter | complete: pass | 8/8 sRGB8 and 3/3 sRGB16 exact parity; invalid profile zero outputs; default/schemas/operators unchanged; 675 tests pass |
| 1aj | U2.4A | Freeze safe negative/slide/B&W interpretation plugin boundary | complete: boundary pass | 3/3 synthetic-test-only witnesses and fail-closed controls pass; no real operator, production eligibility or renderer/schema/profile integration; 698 tests |
| 1ak | U5.R2A | Build the versioned constrained global colour-operator primitive | complete: numerical pass, not a safety claim | exact tetrahedral/reference/replay gates and 704 tests pass; smooth blue-to-purple counterexample passes all numerical constraints, so severe-artifact evaluation stays independent; no fitting/integration |
| 1al | U5.R2B | Freeze and evaluate the strongest eligible global operator frontier | complete: B0 challenger retained | margin-4 anchor56 passes style/non-basic/clipping and all-nine full-res veto; anchor09 fails ID11 red-speckle; raw anchors clip and safe-rich is bland; no production/stock claim |
| 1am | U5.R2C | Freeze B1 complete-policy empirical-ceiling and annotation budget | complete: local tooling, actual B1 gated | byte-identical 54N--78N workload, strict panel/lineage/policy/all-scene validators and 716 tests; binding N null; no B1 pixels/recruitment |
| 1an | U5.R2D | Audit statistics-to-explicit-LUT ML route without stock shortcuts | complete: properties pass, shortcut rejected | 2,304-D descriptor and 726 tests pass; permutation error 4.26e-14 coexists with palette L2 111.69 and an exact-reference/1.0 absent-colour two-LUT counterexample |
| 1ao | U5.R2D1 | Benchmark bounded statistics-conditioned operators on known synthetic truth | complete: canonical info only | exact delta oracle improves 54.3% vs target-only, but interaction is 125% worse than global and captures negative style; 737 tests, zero group leakage, all predictions valid |
| 1ap | U5.R2D2 | Test imperfect canonicalizers and hard operator retrieval | complete: unidentified, retrieval closed | zero of four practical canonicalizers pass; median pairwise disagreement 0.1567 vs 0.03; Top-1/Top-3 fail; 743 tests |
| 1aq | U5.R2E0 | Build a clean-room density-domain explicit look primitive | complete: numerical/diversity pass | five bounded non-affine witnesses; exact partition/replay; positive sampled Jacobians; pairwise RMSE 0.0732--0.1470; 748 tests; no visual/stock/product claim |
| 1ar | U5.R2E1 | Evaluate bounded density witnesses on frozen gold/stress frontier | complete: one B0 density challenger retained | cyan-shadow/warm-highlight s0.50 blind-wins 2/3; style 13.49, non-basic 7.39, zero new clipping; 27/27 full-res renders severe-clean including ID11; no production/stock claim |
| 1as | U5.R2F0 | Audit CanonCGT canonical-pivot explicit-LUT external ML route | complete: E2E control feasible, SSL unavailable | Apache source + 20.9MB E2E weight fixed; exact smoke is strong but mostly basic; LUTs unbounded; 2-byte SSL placeholder; no film/stock claim |
| 1at | U5.R2F1 | Run bounded external CanonCGT E2E reference-condition safety pilot | complete: reference-sensitive, raw policies safety-fail | bank pairwise DE 8.53; 6/9 style+non-basic pass but 0/9 pass clipping/range; exact 81x2 and replay; no visual shortlist |
| 1au | U5.R2F2 | Test data-independent bounded projection of predicted explicit LUTs | complete: no automatic survivor | node clip retains style but fails structure/safety; both safe contractions pass structure/range 9/9 but style/non-basic 0/9; exact 243x2; no visual shortlist |
| 1av | U5.R2G0 | Refit immutable ML LUT controls into a bounded explicit operator family | complete: style retained, boundary safety fail | style 9/9, non-basic 6/9, sensitivity 10.07, +4.77 vs F2; clipping 0/9 and no survivor; exact 81x2 |
| 1aw | U5.R2G1 | Apply one fixed RGB8 quantization-headroom adapter to immutable G0 operators | complete: no survivor, branch closed | structure/range/style 9/9 and retention pass; clipping 0/9 because codes 1/254 remain inside frozen epsilon; exact 81x2 |
| 1ax | U5.R2H0 | Audit film-specific spectral/sensitometric prior algorithms and obtainable data | complete: Velvia-only feasibility | Velvia 50 has all three public graph families; Portra/Ektar lack separated CMY bases and full print inputs; `docs/U5_R2H0_SPECTRAL_SENSITOMETRIC_SOURCE_AUDIT.md` |
| 1ay | U5.R2H0A | Velvia 50 datasheet spectral witness numerical identifiability pilot | complete: invalid/unidentified | strong base effect, but neutral gate fails and D65-equal metamer spread massively dominates; no visual review or rescue |
| 1az | U5.R2H0C | Measured-natural-reflectance conditional spectral variability source audit | complete: CAVE research-only pass | official 405,988,465-byte 31-band archive; internal aggregates only, no redistribution or H0A rescue |
| 1ba | U5.R2H0C1 | CAVE integrity and conditional-support feasibility | complete: bounded empirical-prior candidate | 992/992 official CRC; 263 grouped pairs; output DE 1.38/5.47 median/p95, bootstrap pass; no identified spectrum/stock claim |
| 1bb | U5.R2H0C2 | Group-held-out simple measured-spectrum canonicalizer comparison | complete: retain hard T=1 | 36.13% coverage, 76.60% wins, hard .339/2.975 vs smooth .894/3.939 selected error; T2/T3 stability fail |
| 1bc | U5.R2H0C3 | External measured-spectrum replication | complete: failed, broad prior closed | 184/1,732 selected; 6.52% wins; smooth .271 vs hard 1.561 median; every chapter direction and full-policy p95 reject |
| 1bd | U2.2A | Monotone exposure-to-layer-density sensitometry primitive | complete: numerical pass | 3.73e-14 roundtrip, exact neutral anchor, positive Jacobian and distinct layers; U2.2B composition audit next, no stock/integration claim |
| 1be | U2.2B | Non-duplicative sensitometry-to-print composition | contract frozen; implement next | reuse dye/print/paper only; capture/negative stage exactly once; bounded/non-affine/Jacobian/parity gates, no stock claim |
| 1ay | U5.R2H1 | Test hard content routing between two project-owned safe operators | complete: retain hard 1-NN for A0 visual diagnostic | composite BA 70.67%, p=.010, 45.61% regret closure; style view passes, non-basic view fails; no preference/stock claim |
| 1az | U5.R2H2 | Audit exact hard-1NN routed outputs visually and at full resolution | complete: exact router closed | 0/41 routed severe and 0/10 new severe; every blind round is stable 6 routed wins / 4 global wins, below frozen 7/10; no retuning |
| 2 | RF0.3/RF1.3 | Preserve bounded FSA/OWI historical/unknown lane | sealed partial, auxiliary | 258 derivatives / 81,016,399 bytes retained; never count as named-stock coverage or block RF0.4 |
| 4 | RF2.S/RF3 | Per-stock CPU expert ladder, then bounded GPU challengers if justified | data-gated | requires that stock's RF1.4 pass; no cross-stock averaging or RGB generator |
| 4a | RF2.H | Independent historical/unknown CPU expert | data-gated on RF1.3 | separate coverage ledger and claim class |
| 8 | U5.CT9 | Optional controlled named-stock calibration | future | new capture/lab scope and approval |
| 9 | U5.R0T | Audit theorem-level gap in FARO certification | deferred optional support | not a primary paper dependency |
| 10 | U0.3 | Legacy manifest-v2 lineage audit | complete; reference-derived lane blocked | 4,210/4,210 quarantined; new public-data contracts live under U5.CT2 |
| 11 | U0.4 | CPU-safe CI, environment capture and explicit non-gold registry | complete | current U4 evaluation assets remain supporting work |
| 12 | U5.R1A/U4.1/U4.2 | Maintain chromatic ontology/look rubric and stress evidence as CT evaluation support | U5.R1A complete; R1B design complete | R1B2 ready for A0 inventory/prototypes; evidence remains A0-only; participants later require approval |
| 12a | GH0 | Cursor Ultimate Goal harness (rule/skill/state/stop hook) | complete | Goal ACTIVE; stop loop_limit 10; see `docs/drpt/CURSOR_GOAL_PROTOCOL.md` |
| 12b | U5.R1B | FilmStyleSafe failure-suite design/membership contract | complete | contract+validator+tests; no generator corpus yet |
| 12c | U5.R1B2 | Provisional A0 inventory + synthetic operator card scaffold | complete | 7 cards; placeholder hashes; 386 tests; no pixels |
| 12d | U5.R1B3 | Bind real U4.1/U4.2/ID11 A0 identities | complete | 5/7 bound; verified local SHA-256; synth/hardneg unbound |
| 12e | U5.R1B4 | Explicit synthetic highlight-chroma-island prototype | complete | deterministic PNG; autonomous vision notes large magenta disk ≠ fine speckle |
| 12f | U5.R1B5 | HF-speckle synthetic v1 + RF2.C0 Ektar/fixed-e0 bind | complete | dots `73af7d4e...`; Ektar control bound; hardneg still unbound |
| 12g | U5.R1B6 | Bind legitimate-local hard-negative A0 member | complete | bloom/halation `87f111a2...`; 9/9 A0 bound; R1C planning opened |
| 12h | U5.R1C | Conventional-metric failure study / SCIS v0 A0 pilot | complete | conv gap confirmed; SCIS candidate unpromoted |
| 12i | U5.R1C2 | Refine SCIS v0.1 for sparse HF / HF residual | complete | perfect vs hardneg/external; style contamination remains |
| 12j | U5.R1C3 | Conditioned style-control SCIS calibration; 53/55/56 retained | complete weak-pass | affine 2/3, quadratic 1/3; no candidate selected |
| 12k | U5.R1C3D | Minimal spatial cross-fit absorption diagnostic | complete fail/route closed | remains 2/3; sparse HF missed; no more conditioned-SCIS capacity/tuning |
| 13 | U1.1/U1.3B | Make `WorkingImage` the only render ingress and remove early sRGB8 quantization | complete | one float32 main path; exact sRGB8 colour parity, <=1-code effect parity and real RAW audit pass |
| 14 | U1.2–U1.6/U2.5A | Color-state contract, 16-bit/ICC export, HDR/HEIF, B&W safety and tiled foundation | U1.6A-D/F/G1/G2/G3/G4A pass; U1.6E/G0 direct shortcuts closed | next row-chunked global staging; streaming/total memory, HDR/HEIF/wide-gamut remain pending |
| 15 | U2.1–U2.6 | Profile/recipe schema and deterministic reference renderer | U2.1A/U2.4A/U2.5/U2.6 pass; U2.1/U2.4 in progress | strict replay/profile/evidence and isolated interpretation boundary pass; real interpretation/operators remain pending; no profile calibrated |
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

*Active parent: `ULT > RF stock-first real-film mainline` | SF1.3B and conditioned-SCIS remain closed; LSM1 remains ineligible; U1.5C/U2.5/U2.6 product gaps and U2.4A software boundary pass without overclaim; real interpretations remain absent; next selection stays evidence-gated | Integration owner: repository owner or explicitly assigned root agent*
