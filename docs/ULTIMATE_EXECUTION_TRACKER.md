# K-MCFM Ultimate Execution Tracker

> Active planning authority from 2026-07-10; publication priority corrected on 2026-07-12.
> Primary research basis: `docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`. Product strategy: `docs/planning/ULTIMATE_ROADMAP_2026.md`.
> This tracker records intended work; only rows marked `complete` are implemented facts.

---

## 1. Parent goal and task contract

**Parent goal `ULT`**: learn as many evidence-backed, distinguishable and
artifact-safe stock-specific explicit colour experts as practical from
verifiable real photographic-film scans. `film_stock_id` is the primary class;
roll, process, scanner, source and content are nested controls. Historical or
unknown-stock film is a separate auxiliary class and never substitutes for or
counts toward named-stock coverage. FilmSet and other digital simulations
remain controls. Calibrated reproduction is a stricter subset requiring
controlled stock/process/scanner evidence.

**Real-film authority, 2026-07-15:**
`docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md` supersedes any priority or
success condition below that treats FilmSet, camera simulations, recipes, LUTs
or pseudo-teachers as final truth.

**Stock evidence authority, 2026-07-15:**
`docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md` defines label grades,
stock/nuisance hierarchy, separate named/historical coverage and the RF0.4
entry gate. Unknown historical archives cannot satisfy a named-stock node.

**Autonomy invariant**: the active no-data Roll2Film and Style-safe work cannot
depend on the owner supplying images, film/digital pairs, per-image labels, new
preference votes or manual annotation. Public online data is an allowed source
when concrete file access is verified and rights/size/claim gates are recorded;
mere paper mentions or placeholder links do not count. The frozen preference
set `53/55/56/33/09/03/02/01` is the only current owner-preference evidence.
External human validation is a later gated activity, not a prerequisite for
the no-data research DAG.

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
| `deferred` | Explicitly outside the active dependency path; revisit only after a new trigger or approval |
| `research-only` | May be explored but cannot enter Reference/release |
| `retired` | Retained as history/fallback, not active architecture |

---

## 2. Architecture decision record

| Decision | State | Rationale | Revisit trigger |
|---|---|---|---|
| Product objective is strong style under a severe-artifact constraint | user-authoritative | User explicitly defined this as the true standard on 2026-07-11 | Only an explicit user product-goal change |
| Active research cannot depend on new user data or labels | user-authoritative | User explicitly stated that no film/digital pairs or other assistance will be supplied | Only an explicit user capability/scope change |
| Deterministic full-resolution renderer is the Style-safe core | accepted for plan | Current diffusion paths produced severe detail/identity artifacts; bounded color operators offer a stronger style/artifact frontier | Only if a new method passes the same severe-artifact and preference gates |
| Small model may predict bounded parameters | candidate | LUT/grid/curve prediction adds style/context with controlled artifact risk | Preferred deterministic family cannot reach the style frontier |
| Roll2Film special group-information claim is closed on current evidence | failed challenger | Fixed-budget mechanics pass, but BlueNeg correct-roll support loses to content-similar wrong-roll retrieval and does not replicate | Reopen only with new independent same-stock rolls and a preregistered test |
| Fixed explicit experts can independently win the product | accepted for plan | The product objective is style under a severe veto, not proving that ML is necessary | Replace only if a learned bounded method wins the same frozen policy comparison |
| FilmCase/FARO are gated product-system work and supporting research | accepted for plan | Hard selection may help applicability/safety only if the bank has multiple real modes and an Oracle beats the champion | Close FilmCase if diversity or Oracle gates fail; retain FARO/product QA |
| Unpaired evidence may support graded stock-specific real-film-derived claims, never calibration | accepted for plan | Authoritative stock labels plus independent roll/source and nuisance controls can support `S1/S2`; a film scan alone cannot identify a calibrated response | Only controlled paired measurement can promote `S3 calibrated-reference` |
| Generative image models are excluded from FilmCase | accepted for plan | User explicitly excluded the class; FilmCase renderer and supervision remain non-generative | Only a new explicit user instruction |
| Generative editing is isolated | accepted for plan | User excluded it from FilmCase; current paths are also artifact-prone | Future Creative work requires a separate explicit instruction and approval |
| Stock profile includes process and interpretation | accepted for plan | Negative/slide/B&W do not have one intrinsic display RGB look | None; schema invariant |
| First 2-4 stock pilots are selected by RF0.4 evidence, not preference names alone | user-authoritative | Specific stocks are primary, but labels, rolls/sources, content overlap, rights and attainable grade must pass before selection | RF0.4 registry comparison promotes each pilot separately |
| FiveK is optional neutral auto-base only | accepted for plan | Expert retouch is not film identity; only a partial 903-file freeze is present on the current Windows host and the complete source is absent | Complete sources/rights restored and product evidence supports it |
| Local Flickr-derived assets are quarantined | observed | 4,212 JPEGs exist on the current Windows host, but the 4,210-row legacy audit found 0 eligible rows and no roll/source/scanner grouping; two additional files lack propagated eligibility | New durable lineage, rights and group audit |
| FilmSet is local; BlueNeg remains approval-gated | accepted for plan | FilmSet supplies local hidden film-recipe pairs; BlueNeg supplies 53 real roll groups, but neither is named-stock digital/film truth | FilmSet needs manifest/access freeze, not download; BlueNeg needs no-data gates, approval and licence snapshot |
| Real-film scans are the Ultimate P0 evidence | user-authoritative | Digital recipes proved method control but cannot establish film learning | Only a future explicit goal change |
| Specific `film_stock_id` is the primary class | user-authoritative | The owner requires learning real, concrete film stocks; generic old film may coexist but cannot replace them | Only a future explicit goal change |
| Historical/unknown film is a separate auxiliary class | user-authoritative | Old film is useful for archive look and stress evidence, but it is not a named stock | Never count it toward named-stock coverage |
| FilmSet is auxiliary only | user-authoritative | Cinema/ClassNeg/Velvia targets are Capture One recipes, not physical-film scans | Never promote RF/Ultimate nodes from FilmSet |
| Roll2Film is one challenger, not the sole answer | user-authoritative | BlueNeg correct-roll effects reverse sign and CI crosses zero | Compare global, hierarchical, retrieval and bounded conditional operators |

---

## 3. DRPT node tree

Stock-first RF overlay (authoritative over the older U5-oriented tree below):

```text
ULT
`- RF  Stock-first real-film evidence and expert mainline
   |- RF0.3 / RF1.3  Historical/unknown FSA/OWI auxiliary lane
   |- RF0.4           Authoritative stock registry and first-pilot audit
   |- RF1.4           Per-stock label/content/nuisance identifiability
   |- RF2.H           Historical/unknown explicit expert, separate coverage
   |- RF2.S           Stock-specific CPU explicit expert ladder
   `- RF3-RF5         Per-stock bounded GPU challenge, confirmation and product
```

```text
ULT  Ultimate strongly stylized, artifact-safe film-imaging product
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
├── U2  Style-safe renderer and profile system
│   ├── U2.1 Profile/recipe schema and provenance
│   ├── U2.2 Monotone sensitometry + global LUT renderer
│   ├── U2.3 Negative/slide/B&W interpretation interfaces
│   └── U2.4 Legacy safe_lab compatibility and migration
├── U3  Optional owned paired calibration lane
│   ├── U3.1 Capture protocol, rights and lab/scanner SOP
│   ├── U3.2 Portra 400 pilot
│   ├── U3.3 Velvia 50 pilot
│   └── U3.4 Frozen roll/lab/scanner holdout
├── U4  Evaluation V2
│   ├── U4.1 Severe artifact gold/stress gates
│   ├── U4.2 Autonomous style salience and appeal scorecard
│   ├── U4.3 Content/effect quality diagnostics
│   ├── U4.4 Deferred external human validation
│   ├── U4.5 Performance/cross-platform benchmark
│   └── U4.6 Conditional calibrated-authenticity scorecard
├── U5  Bounded-AI challenge
│   ├── U5.CT0 Algorithm-first colour-transfer reframe and data/novelty audit
│   ├── U5.CT1 Explicit invertible operator contract and pseudo-roll simulator
│   ├── U5.CT2 Metadata-only FilmSet/BlueNeg grouping and pair-blinding contracts
│   ├── U5.CT3 Known-operator group-size/nuisance identifiability gate
│   ├── U5.CT4 Local FilmSet evidence freeze and frozen paired-blind split
│   ├── U5.CT5 Unpaired pseudo-roll transfer baselines and Roll2Film solver
│   ├── U5.CT6 Approval-gated BlueNeg correct-roll matched-control pilot
│   ├── U5.CT7 Amortised set inference and complete ablation
│   ├── U5.CT8 Hidden transfer/style/artifact evaluation
│   ├── U5.CT9 Optional controlled named-stock calibration
│   ├── U5.FC0 Autonomous FilmCase research and claim contract
│   ├── U5.FC1 Lineage, identifiability, anchors and evaluator freeze
│   ├── U5.FC2 Bounded transform/case bank
│   ├── U5.FC3 Oracle routing-value hard gate
│   ├── U5.FC4 Nonlearned retrieval baselines
│   ├── U5.FC5 Transform-aware asymmetric reranker
│   ├── U5.FC6 Hard sparse router, confidence and OOD fallback
│   ├── U5.FC7 Optional bounded local residual
│   ├── U5.FC8 Frozen ablation and promotion decision
│   ├── U5.1 1D + 3D LUT baseline
│   ├── U5.2 SepLUT/NILUT challenge
│   ├── U5.3 FC7 subleaf: bilateral-grid implementation
│   ├── U5.4 FC7 subleaf: optional masks
│   ├── U5.5 Isolated generative R&D outside FilmCase
│   └── U5.6 Winner runtime conversion
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
U0 ─┬→ U1 → U2 deterministic transform foundation ─┐
    └→ U4 artifact/style/evaluator foundation ─────┼→ U5 Roll2Film + supporting FilmCase/FARO → U7 → U8
                                                   └→ U3 remains a deferred optional evidence lane
```

U4 is a continuous validation sibling, not an end-of-project QA phase.

---

## 4. Integrated work board

### RF — real-film Ultimate mainline (priority reset 2026-07-15)

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| RF0 | in progress | Real-film evidence ledger, source/rights/group gates | none | Reopen report plus hashed manifests |
| RF0.1 | complete | Acquire FILM-R v2, verify 88 files and CC BY 4.0 snapshot | reopen report | 88/88 files, 437,570,872 bytes, all remote MD5 and local SHA-256 verified |
| RF0.2 | complete/limited | Fail-closed FILM-R manifest, visual audit and group ceiling | RF0.1 | `docs/data/REAL_FILM_FILMR_V2_GATE.md`; unknown roll/process/scanner; `real-film-derived/unknown-look` only |
| RF0.3 | sealed auxiliary partial | FSA/OWI public-domain archive: metadata, 64-image visual pilot and creator/location grouping | RF0.1 + RF1.1 stop | 558 canonical records; 258 derivatives / 81,016,399 bytes retained; no further download while named-stock P0 advances |
| RF0.4/SF0.1 | complete | Authoritative registry plus exact four-pilot BlueNeg acquisition freeze | current evidence ledger | 13/53/491 cross-check; 189 files / 227,287,697 bytes; Gold 400-5 rejected after whole-test-roll sealing; `docs/REAL_FILM_STOCK_PILOT_ACQUISITION_FREEZE.md` |
| SF0.2 | complete | Download/hash-verify the exact isolated four-pilot manifest without decoding | SF0.1 | 189/189 LFS hashes; 227,287,697 bytes; zero external lane files; report SHA-256 `de286950...`; old BlueNeg pilot untouched; `docs/REAL_FILM_STOCK_PILOT_DOWNLOAD_RESULTS.md` |
| SF0.3 | complete/corrected | Decode integrity, duplicate/border/content and stock×roll×proxy support audits | SF0.2 | hardened metadata cross-check; 189/189 RGB PNG; Gold display-candidate; others post-negation-preview diagnostics only; report v2 SHA-256 `a93257ee...`; 8/8 contact sheets reviewed; `docs/REAL_FILM_STOCK_PILOT_INTEGRITY_RESULTS.md` |
| RF1.1 | complete: unidentified | Real-film signal/nuisance/content separability audit | RF0.2 | only Velvia spans two supported content cells; no comparable cross-content families; no classifier trained |
| RF1.2 | complete: nuisance | BlueNeg nested leave-one-frame-out sign-reversal diagnostic | current CT6 + RF0.2 | retrieval-wrong beats correct roll on all four raw roll means; `docs/ROLL2FILM_BLUENEG_NESTED_LOO_RESULTS.md` |
| RF1.3 | deferred auxiliary | Freeze FSA/OWI creator-out and location-group-out real-film holdouts | RF0.3 grouping | partial pixels retained; resume only without blocking named-stock work; groups are not physical rolls |
| RF1.4A | complete: preview route closed | Post-negation-preview label/content/nuisance identifiability | SF0.3 | GA/Konica structural fail; NPH/Gold primary 0.727, roll-null p=0.191, simple-global ties, nuisance shortcut 0.909; `docs/REAL_FILM_STOCK_IDENTIFIABILITY_RESULTS.md` |
| RF1.4B0 | complete: pass | Gold100 official bbox/matrix alignment and support gate | RF1.4A | 47/47 exact bbox/proxy pairs, six rolls, restricted pickle, byte-identical report `f562141a...`; no colour fit; `docs/REAL_FILM_GOLD_PROXY_ALIGNMENT_RESULTS.md` |
| RF1.4B1 | complete: metric+visual pass, simple matrix wins | Gold100 display-proxy paired-transform identifiability | RF1.4B0 | 47 pairs/6 LOO rolls; SepLUT passes all frozen gates, but bounded 3x3 affine is simpler and better overall (5.307 vs 5.380 Delta E76); no severe visual artifact; `docs/REAL_FILM_GOLD_TRANSFORM_CONSISTENCY_RESULTS.md` |
| RF2.S0 | complete: closed | Freeze all-roll Gold archive-display matrix and test bounded transplant on existing digital gold/stress images | RF1.4B1 | OOD coverage passes, but style 6.25<7, non-basic residual 1.94<4.9 and worst clipping 8.70%; weaker strengths worsen style; no visual run; `docs/REAL_FILM_GOLD_MATRIX_TRANSPLANT_RESULTS.md` |
| SF0.4 | complete: metadata pass | Audit Commons exact Ektar100/Superia X-TRA400/Gold200 plus Velvia-family control | RF2.S0 close | 658 rows; 3 exact stocks pass source/rights/uploader gates; zero pixels; snapshot `f7ee9db...`; `docs/REAL_FILM_COMMONS_STOCK_SOURCE_AUDIT_RESULTS.md` |
| SF0.5 | complete: one-stock pass, learning closed | Audit derivative-only Commons pixels and true source groups | SF0.4 | 36 files/27.81MB clean; Ektar 26/8 authors passes, Superia 2/2 and Gold 8/4 fail; `docs/REAL_FILM_COMMONS_STOCK_PIXEL_PILOT_RESULTS.md` |
| SF0.6A | complete: one new metadata pass | Audit ten exact Commons stock categories for true-author and derivative-rights support | SF0.5 | UltraMax400 passes 84/15 authors/51 strict rows; 9 stop; zero pixels; `docs/REAL_FILM_COMMONS_STOCK_EXPANSION_RESULTS.md` |
| SF0.6B | complete: Kodachrome64 pass | Audit exact Kodachrome25/64, Ektachrome Elite100/200 and Vision3 50D/250D | SF0.6A | Kodachrome64 51/22 authors/15 strict rows; five stop; zero pixels; `docs/REAL_FILM_COMMONS_THIRD_STOCK_RESULTS.md` |
| SF0.7 | complete: UltraMax pass, Kodachrome stop | Audit derivative-only UltraMax400/Kodachrome64 pixels | SF0.6B | 51 files/42.26MB clean; UltraMax 37/8 authors passes, Kodachrome 14/2 fails; `docs/REAL_FILM_COMMONS_MULTI_STOCK_PIXEL_RESULTS.md` |
| SF0.8 | ready: third pixel stock | Obtain a third rights-complete, source-diverse stock or freeze a prospective balanced-group hypothesis | SF0.7 | no post-hoc threshold weakening; third stock must independently pass pixels/content/vision before learning |
| RF2.H | data-gated on RF1.3 | CPU historical/unknown-stock explicit expert, reported outside named-stock coverage | RF1.3 | useful independent archive look without stock claims |
| RF2.S | data-gated on RF1.4 | Per-stock CPU global/hierarchical/retrieval/conditional explicit expert ladder | RF1.4 per stock | stock-specific gain and distinctiveness after matching saturation/contrast/style |
| RF3 | conditional per stock | GPU bounded parameter challengers | a specific RF2.S residual value | curves/LUT/router/grid only; no RGB generator or cross-stock averaging |
| RF4 | pending | Unseen stock/roll/source style and severe-artifact confirmation | RF2.S/RF3 | all RF-G0..RF-G8 gates, per stock; named/historical coverage separate |
| RF5 | pending | Product integration and OOD fallback | RF4 | deterministic replay and provenance |

### U0 — Truth, rights and reproducibility reset

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U0.0 | complete | Ultimate research synthesis and execution tree | none | Roadmap + this tracker; 2026-07-10 |
| U0.1 | complete | Active docs agree on deterministic current baseline and retired diffusion path | U0.0 | AGENTS/README/IMPL_PLAN/TASK_BOARD and planning index agree; 2026-07-10 |
| U0.2 | blocked | Root `LICENSE`, NOTICE and public-claim decision | owner/legal choice | License file exists; dependencies/assets audited |
| U0.3 | complete (reference lane blocked) | Isolated manifest-v2 lineage/eligibility audit; all uncertain rows quarantined | none | 4,210/4,210 classified; 0 source groups/0 eligible rows; 140 near pairs retained inside quarantine; evidence: `docs/FILMCASE_U03_LINEAGE_AUDIT.md` |
| U0.4 | complete | CPU-safe CI, checksum baseline, local environment capture and explicit non-gold benchmark registry | U0.1 | `configs/reproducibility_baseline.json`, verifier, CI workflow, 25 local tests; evidence: `docs/REPRODUCIBILITY_BASELINE.md` |
| U0.5 | complete, corrected again | Reconcile host-specific local bytes and verified remote access | U0.1 | 2026-07-15 Windows audit: FilmSet decompressed tree 21,140 images / 11,262,805,356 bytes, four train branches of 4,657 and four test branches of 628; `manifest.jsonl` and partial FiveK freeze present; `film_domain` 4,212 JPEGs but prior eligibility remains fail-closed; BlueNeg absent/remote-verified |

Restrictions:

- preserve and do not stage the existing user deletion of `data/raw/.gitkeep`;
- do not delete old experiments; mark them historical;
- do not push the current local-only branch without explicit approval;
- do not add an MIT license merely because old docs say MIT—the owner must confirm.

### U1 — Color-managed high-precision foundation

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U1.1 | in progress | `render_film` consumes `WorkingImage` | U0.4 | Production ingress now uses `load_working_image` and records decode provenance; an explicit legacy sRGB8 adapter remains until U1.3. JPEG/PNG/TIFF/RAW E2E fixtures and removal of the adapter remain pending. |
| U1.2 | pending | `scene/display/unknown` state and Reference/Approximation policy | U1.1 | Unknown inputs warn/fail closed; metadata round-trip |
| U1.3 | pending | Correct TIFF/PNG/JPEG encoding, 8/16-bit and ICC | U1.1 | Extension=encoding; bit-depth/profile tests |
| U1.4 | pending | ACEScg or validated wide-gamut working contract | U1.2 | OCIO config/version pinned; golden transform vectors |
| U1.5 | pending | HEIF/HDR/gain-map detect/preserve or explicit rejection | U1.2 | Fixtures for supported/unsupported variants |
| U1.6 | pending | Halo-aware tile/cache renderer | U1.3 | Full-frame vs tiled tolerance; bounded memory |

Do not claim camera-accurate RAW solely from generic rawpy. Reference-grade camera paths require a known DNG/IDT/profile; generic development remains labeled.

### U2 — Style-safe renderer and profile system

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U2.1 | pending | Versioned profile + recipe JSON schema | U1.2 | Schema tests, migrations, provenance card |
| U2.2 | pending | Monotone exposure/sensitometry curves | U1.4, U2.1 | Monotonicity/property tests; curve report |
| U2.3 | pending | Tetrahedral 3D LUT/global residual | U2.2 | Identity/gamut/interpolation tests |
| U2.4 | pending | Negative/slide/B&W interpretation plugin boundary | U2.1 | Three synthetic reference profiles and contracts |
| U2.5 | pending | Legacy `safe_lab` adapter | U2.1, U1.3 | Old recipes render within frozen tolerance |
| U2.6 | pending | Profile evidence labels | U2.1 | `heuristic/measured/paired/held-out` visible in CLI/API |

U2 internal/local schema and renderer research is not blocked by the repository-license decision. U0.2 remains mandatory before publishing code, schema assets, profiles or weights. U2 can ship strongly stylized `film-inspired` profiles only after U0.2 and U4 artifact/style gates. It cannot use `calibrated` or strong named-stock reproduction claims before U3 and the conditional authenticity gate.

### U3 — Optional owned paired calibration lane

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U3.1 | deferred | Rights-cleared capture protocol and releases | future owner budget/legal/lab trigger | Signed allowed-use matrix and SOP |
| U3.2 | deferred | Scanner/lab repeatability pilot | U3.1 | Repeat scans/rolls quantify measurement floor |
| U3.3 | deferred | Portra 400 paired pilot | U3.2 | Charts + at least 3 rolls across 2 process sessions + 30–50 valid scenes minimum |
| U3.4 | deferred | Velvia 50 paired pilot | U3.2 | Same evidence; E-6/direct interpretation |
| U3.5 | deferred | Freeze train/val/gold by group | U3.3, U3.4 | Hash manifest and access log before first fitting |
| U3.6 | deferred | Scale to about 100 valid scenes/stock if powered | U3.5 | Pilot variance/effect-size decision, not arbitrary count |

The user has explicitly stated that no paired captures or other data contribution will be supplied. U3 is therefore not an active ask, blocker or fallback for U5. It may be reopened only by a future explicit scope and resource change.

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
| U4.1 | in progress | Fail-closed frozen-set/adjudication module plus provisional 9/32 seed | U0.4 | All required content categories now covered, including a research-only FilmSet face stress input; final-gold promotion remains blocked by absent complete per-candidate adjudications; evidence: `docs/FILMCASE_U41_EVALUATION_FOUNDATION.md` |
| U4.2 | in progress | Three-round blinded style/appeal protocol plus normalized same-input anchor replay | U4.1 | 54 color-only anchor/control renders and anonymous 3-pass protocol exist; final scorecard remains gated on recorded blind reviews and U4.1 finalization; evidence: `docs/FILMCASE_U42_BLIND_AUDIT_FOUNDATION.md` |
| U4.3 | in progress | Diagnostic-only high-frequency chroma-island report plus future content/effect suite | U4.1 | Red-highlight diagnostic foundation is implemented and explicitly non-veto; face/text/edge/texture and effect diagnostics remain pending; evidence: `docs/FILMCASE_U43_DIAGNOSTICS_FOUNDATION.md` |
| U4.4 | deferred | External blind pairwise human validation | U8 beta + participant/rights approval | preregistered questions, exclusions and analysis; not a FilmCase research dependency |
| U4.5 | pending | Windows/M5/CPU performance matrix | U1.6 | cold/warm p50/p95, RAM/VRAM, 24/100MP, batch |
| U4.6 | deferred | Conditional calibrated-authenticity scorecard | U2.2, U3.2 | chart/EV/illuminant/roll/lab slices + CIs for profiles claiming calibrated |
| U4.7 | complete | Record the user's known Velvia 50 preference anchor | numbered 56-scheme sheet | Mapping and evidence weights recorded; 2026-07-11 |
| U4.8 | complete | Record the true standard: strong style without severe glitch/artifact | user feedback | Product objective and promotion order updated; 2026-07-11 |
| U4.9 | complete | Initial visual audit of preferred and technical-comparator schemes | existing contact sheets + full-resolution stress cases | `docs/VISUAL_STYLE_AUDIT_2026-07-11.md`; 2026-07-11 |

Five reports stay separate:

1. severe glitch/artifact veto;
2. style salience and user appeal;
3. graded content/effect quality diagnostics;
4. calibrated authenticity when that claim is requested;
5. product performance and reliability.

Never collapse them into one score.

Provisional promotion rules, frozen before each experiment:

- severe artifact veto: zero confirmed severe face/text/object/geometry, smearing, banding, large unintended clipping, seam, color-block, repeated-texture or flicker failures on the frozen gold set; report rate + CI on the wider stress set;
- style objective: among survivors, maximize blinded autonomous style strength/appeal evidence against the neutral/bland baseline and the current preferred anchors; this is not presented as population preference;
- graded quality: moderate/local defects reduce ranking but only preregistered severe defects cause automatic rejection;
- calibrated claims only: full-roll holdout and stock-match evidence apply when a profile is labeled calibrated;
- exact thresholds and severity examples are frozen before final evaluation, not chosen after seeing candidate results.

These U4 rules remain product/exploratory promotion rules and supporting
Roll2Film evaluation. They do not define the primary algorithmic claim. FARO's
complete-policy endpoint remains valid only for a later system/product claim;
it evaluates every independent scene including fallback, uses a marginal
scene-level risk bound plus non-trivial coverage, and never estimates
preference only on a method-dependent survivor subset.

#### Known user preference anchor — Velvia 50

Source artifact: `outputs/contact_sheets/velvia50_all_schemes_numbered_20260616/ALL_SCHEMES_NUMBERED.png`; mapping authority: sibling `NUMBER_MAP.csv`.

The user reported the preferred set `53, 55, 56, 33, 09, 03, 02, 01` on 2026-07-11. The list order is preserved as stated but is not treated as a ranking or converted into cardinal score gaps.

| Number | Scheme label | Render count | Evidence use |
|---:|---|---:|---|
| 53 | `velvia50_digital20_s0p72_gamutsafe` | 20 | full-set preference baseline |
| 55 | `velvia50_rawpixls20_s0p50_gamutsafe` | 20 | full-set preference baseline |
| 56 | `velvia50_rawpixls20_s0p58_gamutsafe` | 20 | full-set preference baseline |
| 33 | `evaluate_color_pipeline_smoke` | 1 | direction cue only |
| 09 | `baseline_noclip_s0p50` | 20 | full-set preference baseline |
| 03 | `baseline_current_smoke2` | 1 | direction cue only |
| 02 | `baseline_current_smoke` | 2 | direction cue only |
| 01 | `baseline_current` | 20 | full-set preference baseline |

Interpretation:

- the five comparable full-set choices (`01/09/53/55/56`) all belong to deterministic baseline or gamut-safe Lab families; this strengthens the decision to keep the deterministic family as the preference champion;
- the three smoke choices (`02/03/33`) are not promotion evidence because they cover only one or two images;
- these choices measure personal aesthetic preference, not stock/process authenticity;
- future blind studies should include `53/55/56/09/01` as anchors, normalize them onto one frozen image set, and test whether their apparent ranking survives randomized labels and new scenes.

User feedback also states that many theoretically stronger candidates look technically clean but have little film style. Therefore:

- a candidate cannot be promoted merely for better safety, clipping, SSIM, ΔE or smoothness if observers cannot perceive a film signature;
- “film-style salience” is not “maximum effect strength”: generic saturation, crushed contrast, noise or orange highlights do not pass by themselves;
- evaluate a color-only pass with grain/halation disabled and a full-look pass with effects enabled, so effects cannot hide a weak color model;
- a salient but stock-inaccurate result may ship as a clearly labeled `film-inspired` style after artifact/product gates; it cannot use a calibrated stock claim.

### U5 — Algorithmic colour transfer plus bounded product baselines

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U5.CT0 | complete | Roll2Film algorithm-first problem, current-data audit, claim ladder, nearest-work boundary and research DAG | user correction + local/public evidence | `docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`; 2026-07-12 |
| U5.CT1 | in progress; L2 spline core passes | Versioned invertible explicit-operator contract plus known-operator pseudo-roll simulator | U1/U2 interfaces; no external data | Affine plus strictly monotone rational-quadratic splines pass inverse/Jacobian/serialization and 33-cube/65-cube bake checks. Explicit L0/gauge/shaper closure remains. Evidence: `docs/ROLL2FILM_E0_L2_FIXED_BUDGET_V3.md` |
| U5.CT2 | complete for FilmSet and BlueNeg metadata | FilmSet/BlueNeg eligibility, grouping, licence snapshot and pair-blinding contract | U0.3 | FilmSet access freeze passes. BlueNeg exact revision/licence/inventory and whole-roll split pass for a bounded 101-file pilot; only four Kodak Gold 100-5 rolls have same-film matched controls. Evidence: `docs/ROLL2FILM_FILMSET_EVIDENCE_FREEZE.md`, `docs/ROLL2FILM_BLUENEG_METADATA_FREEZE.md` |
| U5.CT3 | in progress; fixed-budget affine and L2 controls pass | Fixed-total-sample group, coverage, partition, nuisance, prior-swap and scanner identifiability study | CT1 | L2 exact partition equivalence; true nuisance boundaries, independent support and correct single-operator groups beat paired controls at fixed 4,096 pixels. Prior shift, flexible normalization and scanner/profile composition expose confounding. `roll_information=not_established`; matched real-roll controls remain. Evidence: `docs/ROLL2FILM_E0_L2_FIXED_BUDGET_V3.md`. |
| U5.CT4 | complete | Freeze local FilmSet archive and 4,657-internal/628-final paired-blind corpus | CT2 | 21,140-file tree hashed; 2,096/2,096/465 content-disjoint train/dev identities; training loader rejects lockboxes; final 628 has zero decoded payloads and remains sealed. Evidence: `docs/ROLL2FILM_FILMSET_EVIDENCE_FREEZE.md` |
| U5.CT5 | complete on internal confirmatory | Matched-strength deterministic/classical baselines plus Roll2Film solver on FilmSet internal dev | CT2/CT3/CT4 | Universal pooled L2 fails. Fixed recipe bank: Lab for Cinema; pooled L2 for ClassNeg/Velvia. All 238 confirmatory identities pass the full-resolution severe visual veto with strong style; raw excursion frequency is separated from magnitude. Official 628 remains sealed. Evidence: `docs/ROLL2FILM_CT5_CONFIRMATORY_RESULTS.md`, `docs/ROLL2FILM_CT5_FULLRES_VISUAL_AUDIT.md` |
| U5.CT6 | complete; ambiguous | BlueNeg 8-bit preview/pseudo-GT correct-roll matched-control pilot | CT2/CT3 pass | Correct-roll gain changes sign across held-out rolls (`-0.220/+0.309`); roll-cluster CI crosses zero. Roll information is not established. Evidence: `docs/ROLL2FILM_BLUENEG_CONFIRMATORY_RESULTS.md` |
| U5.CT7 | stopped by CT6 gate | Optional permutation-invariant amortised set inference and complete ablation | CT6 pass required | Do not run: only one of two rolls benefits, so added capacity would overfit four rolls rather than establish group information |
| U5.CT8 | execution policy frozen; final still sealed | Final 628 transfer plus style/artifact evaluation | CT5–CT7 decisions frozen | Exact source/operator hashes, fixed primaries/basics, bootstrap/style gates and axis-extreme visual veto are frozen before first decode. Evidence: `docs/data/ROLL2FILM_CT8_FINAL_628_EXECUTION_CONTRACT.md` |
| U5.CT9 | deferred | Controlled named-stock/process/scan calibration | new explicit capture/lab scope | Multiple rolls/process-scan sessions and whole-roll holdout; only then Level-C language |
| U5.R0 | complete, supporting | FARO literature/novelty audit, problem formulation and system DAG | user research reframe + current evidence | `docs/planning/FARO_RESEARCH_PROGRAM_2026.md`; 2026-07-11; paper priority superseded 2026-07-12 |
| U5.R0T | deferred optional support | Dedicated statistical novelty audit and theorem-or-close decision | separate future FARO system-paper scope | Distinguish LTT, two-stage/joint selective certificates, noisy-label and non-exchangeable CRC before any statistical-method claim |
| U5.R1A | ready support | ChromaticTail/FilmStyleSafe ontology, annotation schema, split and sampling/preregistration worksheet | U4 foundation | Supporting transformation-induced chromatic-severe labels, legitimate-local hard negatives and traceable look rubric for CT5-CT8/product QA |
| U5.R1B | pending | Stress generator plus real cross-algorithm failure suite | U5.R1A | Parent-scene grouping, no final-test leakage, current known failures represented |
| U5.R1C | pending | Conventional-metric failure study plus prospective intent-aware SCIS v0 | U5.R1B | A0 development and A1 hidden low-FPR sensitivity on unseen transform families/legitimate-local hard negatives; pass/fail novelty decision |
| U5.R2A | pending | Versioned numerically constrained operator contract | U1/U2 renderer foundation | Identity/curve/LUT property tests, golden vectors and explicit non-safety counterexample |
| U5.R2B | pending | Strong global operator frontier | U5.R2A, U5.R1 | Frozen strongest eligible global policy under identical renderer/export, both risk gates and target-look gate |
| U5.R2C | pending | B1 complete fixed-bank cross-rater empirical-ceiling policy and annotation budget | U5.R2B | Best qualified nonsevere candidate or identity on every scene; supported all-scene tie-score gain or adaptive branches stop |
| U5.R3 | blocked on U5.R2C pass | Fixed-bank transform-conditioned counterfactual scorer | U5.R2C pass | Beats source similarity/reward/transform-only baselines and reduces empirical-ceiling regret without shortcuts |
| U5.R4A | blocked on U5.R3 | Adopted LTT/two-stage/joint certificate for the complete deterministic fixed-`K` policy | U5.R1, U5.R3 | Reproduced overall/selected-risk, look, coverage and utility guarantees include proposer, scene-hash seeds, final-resolution audit and identity fallback |
| U5.R4T | blocked on U5.R0T gap | Optional new statistical method | U5.R0T distinct gap | New theorem and tighter matched-valid frontier; otherwise branch closes as redundant |
| U5.R5 | blocked on U5.R4A | B3 hidden end-to-end policy test | U5.R4A + participant approval | Both risk UCBs, look LCB, style-qualified coverage LCB and all-scene tie-score LCB conjunction passes |
| U5.R6 | blocked on U5.R5 | Optional proposer/local/physical/counterexample-monitor extensions | U5.R5 | Independent frontier gain at identical `K`/compute without breaking either risk or style gate |
| U5.R7 | blocked on U5.R5 | B4 external replication and full paper ablation | U5.R5 + participant approval | New source clusters/raters, label-error sensitivity, explicit marginal-guarantee limits and publication decision |
| U5.FC0 | complete | Autonomous FilmCase research/claim contract | user constraints + current evidence | `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`; 2026-07-11 |
| U5.FC1 | pending | Lineage/rights eligibility, source-controlled identifiability, normalized anchors and evaluator freeze | U0.3, U0.4, U4.1, U4.2 | strong/weak/fail/ambiguous/invalid decision; no unresolved split leakage |
| U5.FC2 | pending | Explicit bounded transform and CaseRecord/case-bank representation | U5.FC1, U5.1 | Replayable CCM/curves/LUT experts; case evidence and cross-application matrix |
| U5.FC3 | pending | Evaluator-Oracle routing-value hard gate | U5.FC2 | Oracle significantly beats global champion or FilmCase stops |
| U5.FC4 | pending | Nonlearned context/photometric/semantic retrieval baselines | U5.FC3 pass | Oracle-gap closure, route stability and negative controls |
| U5.FC5 | pending | Transform-aware asymmetric applicability reranker | U5.FC4 gap remains | Beats generic retrieval, closes ≥80% Oracle gap, no shortcut/collapse |
| U5.FC6 | pending | Hard sparse routing, confidence calibration and OOD fallback | U5.FC4 or U5.FC5 winner | Stable Top-1/medoid routing; low confidence fails to global champion |
| U5.FC7 | pending | Optional bounded local residual | U5.FC6 + preregistered local failure | Independent local gain; no halo/seam/speckle |
| U5.FC8 | pending | Frozen ablation and promotion/stop decision | U5.FC6; U5.FC7 optional | Simplest survivor, complete cards, failure gallery and product handoff |
| U5.0 | retired | Generic style challenge protocol | replaced by U5.FC0/U5.FC1 | Preserved as historical node; no independent work |
| U5.1 | pending | Preferred deterministic CCM/curves + 1D/3D LUT global frontier | U5.FC1, U2.2, U2.3 | Frozen global champion with no severe gold-set artifacts |
| U5.2 | pending | SepLUT/NILUT representation challenger | U5.1 | Same inputs/eval; beats simpler frontier without severe artifacts |
| U5.3 | pending | U5.FC7 implementation subleaf: low-resolution bilateral grid | U5.FC7 entry gate | Held-out local residual improvement; no halo |
| U5.4 | pending | U5.FC7 implementation subleaf: bounded masks/semantic conditioning | U5.3 failure pattern | Only if systematic region errors remain |
| U5.5 | research-only | Generative target/Creative comparison outside FilmCase | separate future explicit instruction + U4.1 | Not run by the autonomous non-generative FilmCase plan |
| U5.6 | pending | Winner distillation/runtime conversion | U5.FC8 winner | Non-inferior FP32/FP16/CoreML/ONNX parity |

The stock-first RF tree is the publication/research parent. Its method must
infer and apply explicit colour operators for evidence-backed stocks;
benchmark, selection and rejection work cannot substitute for transfer.
Roll2Film's physical-roll group-information hypothesis is a retained negative
result under current BlueNeg evidence, not the active parent. A film scan is
never silently treated as an input/output pair, an unknown archive is never
called a stock, and an uncalibrated stock-derived look is never called a
measured stock response. Failure of a per-stock RF1/RF2 gate closes or
downgrades that stock branch rather than promoting a generic benchmark.

FARO, ChromaticTail/FilmStyleSafe, FilmCase, source retrieval, hard expert
selection and the FC/R nodes remain evaluation assets, engineering/system
baselines and possible later product work. Complete-policy certification is
existing statistical machinery unless a separately scoped R0T/R4T audit proves
a distinct extension. Candidate-level risk calibration never licenses a final
selected policy.

The current algorithm/data/experiment authority is
`docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`. Supporting FARO
system/evaluation context remains in
`docs/planning/FARO_RESEARCH_PROGRAM_2026.md`; the earlier CaseRecord schema and
FilmCase baselines remain in
`docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`.

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
| U8.2 | pending | Datasheet, model/profile cards, SPDX/CycloneDX BOM | U0.2, U4–U7 | Release bundle audit; U3 additionally required for calibrated packs |
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
2. `U5.FC0` (complete 2026-07-11): freeze the autonomous, unpaired, non-generative FilmCase research contract.
3. `U0.3` (complete, reference lane blocked): isolated manifest-v2 audit classified all 4,210 rows; no source/uploader/roll/scanner group was recoverable, so all 4,210 rows are quarantined and reference-derived FilmCase is closed pending auditable provenance. See `docs/FILMCASE_U03_LINEAGE_AUDIT.md`.
4. `U5.CT0` (complete 2026-07-12; re-gated 2026-07-15): keep colour transfer as the paper requirement, but treat Roll2Film as a challenger until fixed-budget evidence promotes it; product work remains independently winnable.
5. `U5.CT1`: freeze the explicit invertible operator contract and known-operator pseudo-roll simulator.
6. `U5.CT2`: build metadata-only FilmSet/BlueNeg grouping, eligibility and pair-blinding contracts.
7. `U4.1/U4.2`: maintain severe-artifact examples and style/appeal scorecard as CT evaluation support; `U4.6` remains calibrated-only.
8. `U0.2`: ask owner to confirm intended code license and release posture.

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

### Week 4 — colour-transfer identifiability and supporting evaluation

1. `U5.CT3`: run group-size/coverage/nuisance curves on known synthetic operators.
2. Stop the data-bearing paper route immediately if roll grouping adds no identifiable information.
3. `U4.1/U4.2`: expand the supporting safety/style corpus with face/text/skin/sky/ramp/texture/highlight cases.
4. `U4.5`: benchmark current renderer on exact target hardware for product evidence.
5. Keep `U3` deferred; it is not an owner-data request or dependency.

No film purchase, lab booking, model download or GPU training occurs without the corresponding approval.

---

## 6. Experiment queue

| Experiment | Hypothesis | Fixed baseline | One changed variable | Promotion gate | Stop condition |
|---|---|---|---|---|---|
| EXP-CT-00 | Correct structure adds information beyond total sample count only through controlled support/nuisance | known synthetic matrix/spline/LUT operators | group/partition at fixed total pixels and matched colour support | affine controls pass; next require L2 recovery plus robustness to stronger nuisance/prior/scanner factorials | gain is only more pixels, nonlinear truth fails, nuisance absorbs style or prior swap destabilizes the operator |
| EXP-CT-01 | Target-set inference can recover an unpaired film-recipe transfer | FilmSet internal train identities with all pair access destroyed | correct pseudo-group versus pooled/shuffled groups | internal dev gain over strong unpaired baselines at matched style; one final 628 confirmation after policy freeze | no meaningful internal gap closure, style collapse, severe failure or final non-replication |
| EXP-CT-02 | Real physical roll ID contains stable reusable colour-operator evidence | BlueNeg 8-bit preview/pseudo-GT, metadata matched | correct roll versus random/same-film/date-location-scene-matched wrong roll | correct-roll hidden-frame advantage survives all controls | advantage disappears or is driven by deterioration/scanner/exposure |
| EXP-CT-03 | Support shrinkage prevents unsupported strong-colour extrapolation without washing out the look | strongest CT-01 explicit operator | support-aware prior/identity shrinkage | fewer hidden severe chromatic failures at matched style strength | gain comes only from weakening the transfer |
| EXP-VIS-00 | A normalized deterministic bridge can preserve #56 style with #09 containment | #09 and #56 on one manifest | strength/luma + source/chroma gamut mode | Higher style/appeal than #09; no severe gold failure | red-speckle/gradient artifact or no style gain |
| EXP-FC-00 | A stock-associated unpaired signal survives source/scene controls | current random file split | group split + identifiability controls | Signal exceeds permutation/source/grayscale controls | Close reference-derived lane |
| EXP-FC-01 | Multiple stable bounded case transforms exist | normalized #09/#56 + global champion | case construction/capacity ladder | Nontrivial stable modes after saturation/contrast matching | Keep one global transform |
| EXP-FC-02 | Per-scene Oracle selection has real value | best global expert | hard Evaluator Oracle | Meets preregistered win/CI/mode gates with zero severe | Stop FilmCase routing |
| EXP-FC-03 | Simple context retrieval predicts useful cases | Oracle/global/random | handcrafted/generic retrieval feature | Closes ≥80% Oracle gap; stable routes | If Oracle strong and gap remains, open FC5 |
| EXP-FC-04 | Asymmetric transform applicability beats generic similarity | best EXP-FC-03 | pair/listwise reranker | Beats generic and closes ≥80% Oracle gap | Keep generic/global winner |
| EXP-FC-05 | Hard sparse routing avoids style averaging | matched soft/dense router | hard Top-1/medoid selection | Better style frontier with no artifact increase | Use empirically simpler winner |
| EXP-FC-06 | Confidence/OOD fallback contains routing risk | router without fallback | calibrated reject/fallback | High-risk OOD miss ≤5%; unknown state 100% fail closed | Do not productize router |
| EXP-FC-07 | Bounded local residual fixes a systematic local class | global FilmCase winner | bilateral-grid residual | Independent local gain; no halo/seam/speckle | Delete local branch |
| EXP-FC-08 | Every retained component has independent value | simplest global champion | frozen ablation | Full system passes rights/safety/style/OOD/product gates | Remove unsupported complexity |
| EXP-COLOR-01 | 1D + 3D LUT can move beyond the preferred deterministic anchors | `53/55/56/09/01` on one frozen set | global transform | Higher style/preference; zero severe gold-set artifacts | Severe artifact or no style gain |
| EXP-COLOR-02 | SepLUT can increase style without instability | EXP-COLOR-01 | representation | Higher style frontier; no severe gold-set artifacts | Same/poorer, bland or artifact-prone result |
| EXP-LOCAL-01 | Bilateral grid adds scene-aware style safely | current global winner | local grid | Local style/preference improves; no severe halo/tile/color artifacts | Severe artifact, blandness or no independent gain |
| EXP-FX-01 | Exposure-domain halation matches real radial behavior | current display-level halation | composition domain | Held-out point/edge profile improves | Lens/scanner confound unresolved |
| EXP-FX-02 | Density-aware grain matches real NPS | current procedural grain | density-conditioned params | Held-out NPS/ACF within repeatability | Scanner noise not separated |
| EXP-GEN-01 | Research-only Creative comparison outside FilmCase | IP2P fixed grid | FLUX.2 FP8 | Only after a future explicit instruction/approval; never FilmCase evidence | Not scheduled by this plan |
| EXP-RUNTIME-01 | FP16/CoreML/ONNX is color-noninferior | FP32 reference | runtime/precision | ΔE/golden parity + speed gain | Neutral/skin drift or instability |

Each experiment produces: contract, config, input manifest hash, environment, raw per-image metrics, failure gallery, aggregate with CI, decision and scoped commit. No result may update a profile merely because one contact sheet looks good.

---

## 7. Multi-agent / multi-chat ownership protocol

Default execution mode is **Mode A** (one writer). **Mode B** may be used for
independent read-only literature, data and hostile-review leaves; the root
integration owner alone edits shared authority files and commits the integrated
evidence bundle. Mode C requires an explicit same-project coordination artifact
before any write.

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

After propagation, re-integrate bottom-up: verify changed leaves first, then
their U5.CT parent, sibling evaluation/product contracts, `ULT`, active docs and
the final evidence bundle. Record propagation evidence in the project agent log
before closure.

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
- Current data and algorithm-paper authority: `docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`, `docs/DATA_REPRODUCTION_MANIFEST.json`
- Historical FiveK deletion/freeze state: `docs/FIVEK_AUTO_BASE_MODEL_TRACKER.md`; do not treat it as current-Mac availability
- Halation evidence limits: `docs/HALATION_SYSTEM_SPEC.md`
- Data rights boundary: `docs/data/DATA_LICENSE_BOUNDARIES.md`
- Product research synthesis and earlier external links: `docs/planning/ULTIMATE_ROADMAP_2026.md`
- Supporting artifact/system-risk research: `docs/planning/FARO_RESEARCH_PROGRAM_2026.md`
- Autonomous unpaired FilmCase scientific DAG: `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`
- Initial visual audit and deterministic bridge experiment: `docs/VISUAL_STYLE_AUDIT_2026-07-11.md`

---

*Tracker initialized: 2026-07-10; FilmCase subtree frozen: 2026-07-11; Roll2Film subtree opened: 2026-07-12 and re-gated after external audit on 2026-07-15. U0.3 remains a fail-closed 4,210-row legacy audit while current Windows `film_domain` contains 4,212 JPEGs of not-yet-propagated eligibility. FilmSet is local and ready for manifest/lockbox freeze; only BlueNeg acquisition remains download-gated. Integration owner: repository owner or explicitly assigned Codex root agent.*
