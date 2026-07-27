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

**Conditional latent-mode authority, 2026-07-16:**
`docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md` and
`docs/data/LATENT_STOCK_MODE_EVIDENCE_REGISTRY.md` define `H-LSM-1`, the
formal `K=1` branch and the separation between observed evidence and latent
inference. No current stock has proved multiple transferable modes; unpaired
digital-to-film operator identification remains unresolved.

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
| A stock may conditionally contain multiple latent modes | research hypothesis | Averaging could hide stable appearance/operator structure, but current data prove no stock with `K>1` | Test only after stock/connectivity/identifiability/rights gates; `K=1` is a formal result |
| Physical interpretation of latent modes is unknown by default | accepted epistemic boundary | Current sources lack trustworthy exposure/EI/illuminant/push-pull/process/scanner labels | Upgrade only from independent structured evidence; appearance never backfills truth |
| Mode and content spaces remain separate | accepted research contract | Current 4x4 scene colour and geometry/source shortcuts can manufacture clusters and routing | Revisit only if a preregistered nuisance-controlled study demonstrates a safer representation |

---

## 3. DRPT node tree

Stock-first RF overlay (authoritative over the older U5-oriented tree below):

```text
ULT
`- RF  Stock-first real-film evidence and expert mainline
   |- RF0.3 / RF1.3  Historical/unknown FSA/OWI auxiliary lane
   |- RF0.4           Authoritative stock registry and first-pilot audit
   |- RF1.4           Per-stock label/content/nuisance identifiability
   |- LSM             Conditional within-stock latent-mode hypothesis
   |  |- LSM0         Ontology/epistemic contract and formal K=1 branch
   |  |- LSM1         Data/connectivity feasibility; no fitting
   |  `- LSM2-LSM8    Residual/operator identifiability through product validation, gated
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
| RF2.C0 | complete: one external control only | Isolated external spektrafilm spectral prior/control on the provisional nine-image gold set | RF2.S0 negative gates + current external revision/licence audit | Ektar100/fixed-e0: style 8.034, non-basic residual 7.343, zero new clipping and no severe visual failure across 3 blind rounds/9 gold; auto policies are exposure-dominated negative diagnostics; no tracked code/profile/LUT, training, fitting, stock truth or integration; `docs/REAL_FILM_SPEKTRAFILM_EXTERNAL_CONTROL_RESULTS.md` |
| SF0.4 | complete: metadata pass | Audit Commons exact Ektar100/Superia X-TRA400/Gold200 plus Velvia-family control | RF2.S0 close | 658 rows; 3 exact stocks pass source/rights/uploader gates; zero pixels; snapshot `f7ee9db...`; `docs/REAL_FILM_COMMONS_STOCK_SOURCE_AUDIT_RESULTS.md` |
| SF0.5 | complete: one-stock pass, learning closed | Audit derivative-only Commons pixels and true source groups | SF0.4 | 36 files/27.81MB clean; Ektar 26/8 authors passes, Superia 2/2 and Gold 8/4 fail; `docs/REAL_FILM_COMMONS_STOCK_PIXEL_PILOT_RESULTS.md` |
| SF0.6A | complete: one new metadata pass | Audit ten exact Commons stock categories for true-author and derivative-rights support | SF0.5 | UltraMax400 passes 84/15 authors/51 strict rows; 9 stop; zero pixels; `docs/REAL_FILM_COMMONS_STOCK_EXPANSION_RESULTS.md` |
| SF0.6B | complete: Kodachrome64 pass | Audit exact Kodachrome25/64, Ektachrome Elite100/200 and Vision3 50D/250D | SF0.6A | Kodachrome64 51/22 authors/15 strict rows; five stop; zero pixels; `docs/REAL_FILM_COMMONS_THIRD_STOCK_RESULTS.md` |
| SF0.7 | complete: UltraMax pass, Kodachrome stop | Audit derivative-only UltraMax400/Kodachrome64 pixels | SF0.6B | 51 files/42.26MB clean; UltraMax 37/8 authors passes, Kodachrome 14/2 fails; `docs/REAL_FILM_COMMONS_MULTI_STOCK_PIXEL_RESULTS.md` |
| SF0.8A | complete: YFCC15M source pass | Find a third-stock source without weakening Commons gates | SF0.7 | 7.35M rows/1.738GB, repeat-identical audit; Velvia50 51 rows/26 UIDs, prospective contamination filter leaves 45/23; `docs/REAL_FILM_YFCC_STOCK_METADATA_RESULTS.md` |
| SF0.8B | complete: third provisional S0 pixel stock | Reverify and audit at most 32 YFCC Velvia50 pixels, max four per UID | SF0.8A | 25 files/4.58MB/14 UIDs/16%; live rights, decode, zero duplicate, content and vision pass; `docs/REAL_FILM_YFCC_VELVIA50_PIXEL_RESULTS.md` |
| SF1.0A | complete: Ektar source bridge candidate | Search frozen YFCC15M for Ektar100 and UltraMax400 under exact CC-BY text/UID gates | SF0.8B | Ektar 26/10/23.08% passes; UltraMax 16/5/68.75% stops; repeat SHA `966032...10b5`; `docs/REAL_FILM_YFCC_MATCHED_STOCK_RESULTS.md` |
| SF1.0A2 | complete: small diagnostic bridge | Reverify/audit at most 20 YFCC Ektar100 pixels, max four per UID | SF1.0A | 16 files/3.63MB/5 UIDs/25%; live rights, duplicates, source and vision pass; `docs/REAL_FILM_YFCC_EKTAR100_BRIDGE_RESULTS.md` |
| SF1.0B | complete: current pools closed | Falsify Ektar100/UltraMax400/Velvia50 stock signal against source/content/colour shortcuts | SF1.0A2 | both RGB edges fail; scene colour/content dominate; Ektar source geometry 92.86%/p=.016; `docs/REAL_FILM_CONNECTED_STOCK_IDENTIFIABILITY_RESULTS.md` |
| SF1.1 | complete: metadata pass | Acquire/filter public 65,644,027,904-byte YFCC100M SQLite for exact stocks and shared authors | SF1.0B | SHA/S3 pass; two byte-identical scans; Ektar/Velvia 780/240 rows, 122/62 UIDs, 16 shared; `docs/REAL_FILM_YFCC_FULL_INDEX_RESULTS.md` |
| SF1.2 | complete: rights pass | Verify at most 128 bounded Flickr pages for live CC BY support across the 16 Ektar/Velvia shared UIDs | SF1.1 pass | 61 HTML requests; eight bilateral live-rights authors; no image request; `docs/REAL_FILM_YFCC_SHARED_AUTHOR_RIGHTS_RESULTS.md` |
| SF1.3A | complete: pixel/integrity pass | Acquire and integrity-audit at most 38 derivatives/512MiB across the eight SF1.2 UIDs and two stocks | SF1.2 pass | 37 files; 21/16 by stock; eight bilateral UIDs; zero exact/dHash<=4 pairs; no confirmed severe artifact |
| SF1.3B | complete: closed | Leave-one-UID-out Ektar/Velvia identifiability against luma/HOG/4x4 colour/standardized RGB/geometry controls | SF1.3A pass | RGB 56.25%/p=.464; best nuisance 68.75%; delta -12.5 points, CI [-31.25,0]; pool closed for learning |
| SF2.0A | complete: content-confounded close | Audit 63 deterministic Apollo 7 NASA/JSC HTML pages for SO-368/SO-121 stock-magazine-filter-content metadata connectivity | SF1.3B close + Apollo source/rights reconnaissance | 63/63 metadata pages pass ID/stock/support/filter bridge; only one of two required shared content tags passes; no images/fitting/training/LSM; `docs/REAL_FILM_APOLLO7_METADATA_FEASIBILITY_RESULTS.md` |
| SF2.0B0 | complete: insufficient-roll close | Snapshot keyless NASA/JSC STS098 metadata for exact VELVI/5775/5776 queries | SF2.0A close + broader database reconnaissance | rows pass at 168/329/10, but Velvia has 2 rolls below frozen 4-roll minimum; zero overlap/errors/images; no B1/fitting/training/LSM; `docs/REAL_FILM_NASA_STS098_STOCK_SNAPSHOT_RESULTS.md` |
| SF2.0C0 | complete: no-candidate close | Cross-mission NASA/JSC exact-code mission-by-stock-by-roll connectivity census | SF2.0B0 close + reusable keyless source client | 13,255/397/122 all-mission rows across 25 missions; only STS098 joins both primaries and Velvia still has two supported rolls below four; no photos/pixels/fitting/training/LSM; `docs/REAL_FILM_NASA_CROSS_MISSION_CONNECTIVITY_RESULTS.md` |
| SF2.1A | complete: contract-mismatch close | Openverse exact-stock strict-licence shared-creator metadata graph | SF2.0C0 close + bounded official API reconnaissance | 48 metadata requests/960 rows; Ektar pagination repeats 19 identities and only UltraMax passes per-stock gate; no live-page/pixel/fitting/training/LSM; `docs/REAL_FILM_OPENVERSE_SHARED_CREATOR_RESULTS.md` |
| SF2.2R | complete: no-DoR close | Bounded Smithsonian Open Access exact-stock/source-density reconnaissance | SF2.1A close + official S3 unit indexes | fixed 8/256-shard probes across seven units expose disconnected family-level Kodachrome/Ektachrome evidence, not exact connected stock variants; avoid 6.64GB full metadata transfer; no pixels/fitting/training/LSM; `docs/REAL_FILM_INSTITUTIONAL_SOURCE_RECONNAISSANCE_RESULTS.md` |
| SF2.3 | complete: insufficient-connectivity close | Union the three immutable Commons snapshots into a conservative exact-stock/shared-author connectivity graph | SF0.4/SF0.6A/SF0.6B snapshots + SF1.3B negative evidence | 294 strict rows/19 stocks/30 authors; only Ektar100 and UltraMax400 eligible, one shared author and zero retained two-author edges; no live preflight/pixels/fitting/training/LSM; `docs/REAL_FILM_COMMONS_UNION_CONNECTIVITY_RESULTS.md` |
| SF2.4R | complete: terms-blocked no-DoR | Bounded Newgrain exact-stock/shared-photographer source reconnaissance | SF2.3 close + public product/terms surfaces | public catalogue exposes 385 approved stocks and rich stock/process/group fields, but Terms expressly prohibit automated queries/scraping/mining; stopped after two catalogue queries and one two-document schema probe; no retained snapshot/images/fitting/training/LSM; `docs/REAL_FILM_NEWGRAIN_SOURCE_RECONNAISSANCE_RESULTS.md` |
| SF2.5R | complete: register physical-only, source closed | PROV VPRS 17684 digitised negatives ↔ VPRS 17690 film-stock register feasibility | official public API + catalogue descriptions | two byte-identical passes find 6,716 digital collection items but 30/30 register items physical with zero digital/IIIF fields; no pixels requested; reopen only on transcription/digitisation/export; `docs/REAL_FILM_PROV_NEGATIVE_REGISTER_RECONNAISSANCE_RESULTS.md` |
| SF2.6R | complete: public-data unavailable | Current official-release audit for SillyStill CineStill pairs and Emulating Emulsion Velvia chart correspondences | primary papers, project pages and pinned public repositories | SillyStill still exposes only one illustrative pair, placeholder dataset links and no root licence; Emulating Emulsion exposes the method and figures but no measurements, parameters, code or reusable data licence; no download/fitting/training; `docs/REAL_FILM_PAIRED_SOURCE_AVAILABILITY_RECONNAISSANCE_RESULTS.md` |
| SF2.7R | complete: nuisance source pass | ColorReference same-physical Velvia 100F slide scanner/software nuisance corpus | official test-data page + exact server headers | 11 assets/71,068,957 bytes; four complete pipelines × five same Set 3 slides; CRC/decode clean, zero cross-pipeline exact duplicates; opens SF2.7A only; `docs/REAL_FILM_COLORREFERENCE_SCANNER_NUISANCE_RESULTS.md` |
| SF2.7A | complete: partial control pass | Same-slide scanner/software nuisance separability and canonicalization audit | SF2.7R | 20/20 alignments pass; raw held-out device-RGB median .06055, bounded 3x3+bias .01446 (76.11% reduction); every-pair material gate fails on the .01464 same-software/device-model pair, while the largest pair is .09865; no stock fit/training/LSM; `docs/REAL_FILM_SCANNER_NUISANCE_QUANTIFICATION_RESULTS.md` |
| SF2.8R | complete: public data unavailable | NTNU controlled Ektachrome E100/Velvia 50 reversal-film source reconnaissance | CC-BY article + exact public NVA thesis record | controlled two-stock/two-scene/two-illuminant/multi-exposure hyperspectral/MSI design, but public files contain no raw data, measurements, code, manifest or dataset licence; six analysed frames lack same-illumination cross-stock control; no fitting/training/LSM; `docs/REAL_FILM_NTNU_CONTROLLED_REVERSAL_SOURCE_RESULTS.md` |
| SF2.9R | complete: limited measurement pass | ColorReference multi-family IT8/ISO 12641 direct-measurement audit | five exact public reference archives + frozen contract | 1,468,016 bytes, CRC/parse clean, 288 common patches and 41-point 380--780nm spectra; pairwise target medians 1.52--4.22 Delta E76, but calibrated-target manufacture and missing common recorder input make stock operators unidentified; no fitting/training/LSM/redistribution; `docs/REAL_FILM_COLORREFERENCE_MULTIFAMILY_IT8_RESULTS.md` |
| SF2.10R | complete: promising topology / pixel DoR blocked | Audit Color Precision comparison topology, scanner nuisance support and pixel-rights DoR | SF2.9R limitation + eight retained public HTML responses | exact repeat: 927 URLs, 20 weak stock-looking labels and 456 apparent scanner pairs, but no explicit pixel reuse grant, verified pair manifest or independent roll/process replication. Zero image requests; no fitting/training/LSM; `docs/REAL_FILM_COLOR_PRECISION_METADATA_PREFLIGHT_RESULTS.md` |
| LSM0 | complete: hypothesis contract | Ontology, observed/latent/physical interpretation split, legal naming and formal K=1 branch | stock-first authorities + frozen negative evidence | `docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md`; no claim that any stock has multiple modes |
| LSM1 | data-gated: no eligible stock | Per-stock support/connectivity/rights/identifiability feasibility matrix | a stock's RF/SF evidence | SF1.3A passes rights/pixel/connectivity but SF1.3B fails stock identifiability; no training/clustering |
| LSM2 | data-gated | Basic/strength-normalised residual appearance identifiability | LSM1 full pass | stable residual evidence across groups or K=1/unidentified stop; no operator claim |
| LSM3 | data-gated | Compare appearance-only and assumption-sensitive explicit operator signatures | LSM2 pass + separately eligible neutral controls | film-inspired hypothesis only; disagreement across matcher/canonicalizer/pool closes unidentified |
| LSM4 | data-gated | K=1/HDBSCAN/GMM/factor-mixture group-aware mode existence audit | LSM3 pass + frozen development policy | stable cross-group modes after nuisance/negative controls, or accept K=1/close |
| LSM5 | data-gated | Fixed simplest explicit mode bank | LSM4 K>1 support | correct/wrong/shuffled/global/basic/style-matched/OOD/severe comparison |
| LSM6 | data-gated | Evaluator Oracle over stock global champion | frozen LSM5 bank | significant group-aware value or product routing closes |
| LSM7 | data-gated | Hard medoid, then sparse retrieval, then bounded router if necessary | LSM6 pass | stable routing with confidence/OOD fallback; no dense averaging/direct RGB |
| LSM8 | data-gated | Full-resolution unseen-group/product validation | frozen LSM7 policy | severe veto, stress CI, deterministic replay and target-hardware evidence |
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
| U1.1 | complete: single ingress | `render_film` consumes `WorkingImage` | U0.4 | JPEG/PNG/TIFF8, PNG/TIFF16 ingress and real ARW script E2Es pass with explicit provenance; the renderer main path no longer uses the legacy sRGB8 adapter |
| U1.2 | in progress: approximation policy pass | `scene/display/unknown` state and Reference/Approximation policy | U1.1 | Source/post-decode states round-trip separately. Current renderer always labels `film-inspired/look-approximation`; unknown fails closed explicitly and calibrated Reference is false. A future evidence-gated Reference mode remains unimplemented. |
| U1.2A | complete: fail-closed pass | Reject embedded ICC conversion failure instead of discarding the profile and continuing | U1.2 | PNG/JPEG malformed ICC and renderer no-output fixtures; ordinary SDR/profile regressions and 233-test suite pass |
| U1.2B | complete: fail-closed pass | Reject real raster transparency instead of falsely claiming discarded alpha is preserved | U1.2A | RGBA/LA/palette transparency rejects before RGB working pixels; opaque alpha strips with warning and exact RGB equivalence; renderer no-output; 563 tests; `docs/U1_2B_ALPHA_FAIL_CLOSED_RESULTS.md` |
| U1.2C | complete: fail-closed pass | Reject multi-frame/page raster inputs instead of silently taking frame zero | U1.2B | two-frame GIF/two-page TIFF reject before pixel decode; renderer no-output and single-frame regressions pass; 566 tests; `docs/U1_2C_MULTIFRAME_FAIL_CLOSED_RESULTS.md` |
| U1.3 | in progress: float default pass | Correct TIFF/PNG/JPEG encoding, 8/16-bit and ICC | U1.1 | one float32 safe-Lab/effect core feeds default sRGB8 and opt-in PNG/TIFF16 with embedded ICC; calibrated scene-to-display, HDR and wide gamut remain pending |
| U1.3A | complete: opt-in float path | Opt-in float safe-Lab path to true PNG/TIFF16 without changing default 8-bit compatibility | U1.1/U1.3 primitives | uint16 decode/ICC/>256 levels and real ARW vision smoke pass; default remains 8-bit; no calibrated claim |
| U1.3B | complete: float default | Remove the default renderer's early sRGB8 quantization while preserving 8-bit export compatibility | U1.3A | sRGB8 colour exact; deterministic effects <=1 code; 223 tests and full-resolution RAW colour/B&W audit pass; pre-existing B&W chroma defect remains separate |
| U1.4 | in progress: primitives retained, first operator closed | ACEScg or validated wide-gamut working contract | U1.2 | U1.4A/B retain explicit Rec.2020 math/file I/O; C1 isolated math passes but C2 real-image OOD automatic boundary gate fails, so no operator integration; OCIO/ACES and user-facing support remain open |
| U1.4A | complete: primitive pass | Dependency-free D65 linear-sRGB/linear-Rec.2020 conversion primitive | U1.2B/C | matrix/config error 0, identity `2.36e-16`, extended roundtrip `2.38e-7`, neutral 0, no clamp; WorkingImage provenance preserved; 576 tests; `docs/U1_4A_LINEAR_REC2020_PRIMITIVE_RESULTS.md` |
| U1.4B | complete: file-boundary pass | BT.2020 SDR 16-bit RGB PNG cICP ingress/egress | U1.4A | exact CICP/sample/determinism gates, `1.44e-5` linear roundtrip and fail-closed cases pass; 584 tests; no renderer/HDR/ACES/arbitrary-profile claim; `docs/U1_4B_REC2020_SDR_PNG_CICP_RESULTS.md` |
| U1.4C0 | complete: compatibility audit | Determine whether current colour/effect operators can consume Rec.2020 truthfully | U1.4A/B | current safe-Lab boundary and all FilmFX are sRGB-bound; direct integration and sRGB-clip roundtrip claims forbidden; U1.4C1 pure-Lab/backend split authorized; `docs/U1_4C0_WIDE_GAMUT_OPERATOR_COMPATIBILITY_AUDIT.md` |
| U1.4C1 | complete: isolated research pass | Extract working-space-aware D65 Lab colour kernel with legacy parity | U1.4C0 | C1A-C pass conversion, pure-kernel, destination-gamut and wide-colour witness gates with frozen legacy parity; visual/OOD and any integration remain separate; six colour styles only, no effects/renderer/schema claim |
| U1.4C1A | complete: math pass | Add shared-PCS D65 Lab conversion for linear sRGB/Rec.2020 | U1.4C1 | legacy/skimage error 0, Rec.2020 roundtrip `4.77e-7`, cross-space same-colour Lab `6.10e-5`; 593 tests; `docs/U1_4C1A_WORKING_SPACE_LAB_PRIMITIVE_RESULTS.md` |
| U1.4C1B | complete: extraction pass | Move the existing safe-Lab transform behind a pure reusable Lab API | U1.4C1A | both frozen seeded hashes exact, eight-style full/tiled gate unchanged, deterministic/nonmutating/fail-closed kernel tests and 600-test suite pass; RGB/gamut/effects remain outside; `docs/U1_4C1B_PURE_SAFE_LAB_KERNEL_RESULTS.md` |
| U1.4C1C | complete: gamut/adapter pass | Add destination gamut policies and isolated six-style Rec.2020 WorkingImage adapter | U1.4C1B | source/chroma 12/12 pass; preregistered Velvia witness has `0.4797` linear-sRGB excursion; deterministic/provenance/fail-closed gates and 625 tests pass; visual/OOD/integration remain closed; `docs/U1_4C1C_REC2020_SAFE_LAB_ADAPTER_RESULTS.md` |
| U1.4C2 | complete: closed automatic fail | Bounded real-image severe-artifact/OOD audit before any C1C integration | U1.4C1C | 96x2 full-resolution renders repeat identically, but 66/96 exceed frozen 0.5% new Rec.2020 boundary gate; worst 6.7808%; visual stage forbidden/not opened; no confirmed visual-severe verdict or integration; `docs/U1_4C2_REC2020_VISUAL_OOD_RESULTS.md` |
| U1.5 | in progress: explicit rejection strengthened | HEIF/HDR/gain-map detect/preserve or explicit rejection | U1.2 | U1.5A rejects unsupported containers/signals; U1.5B closes the first/last-4-MiB known-marker blind spot; U1.5C pins two real libultrahdr MPO references that fail closed through U1.2C; complete ISO 21496-1 detection and real format support remain pending |
| U1.5A | complete: fail-closed pass | Fail closed on HEIF/AVIF and recognized HDR/gain-map signals before pixel conversion | U1.5 | AVIF, Adobe/Android, Apple and PNG metadata fixtures reject before output; SDR regressions and 230-test suite pass |
| U1.5B | complete: fail-closed pass | Structurally traverse JPEG APP and PNG text metadata for recognized gain-map signals outside the U1.5A edge sample | U1.5A | middle-of-file JPEG/PNG, compressed-text, bounded-expansion, ordinary-large and no-output gates pass; 48 focused/639 full tests; detection only, no HDR/ISO support claim; `docs/U1_5B_STRUCTURED_GAINMAP_SIGNAL_SCAN_RESULTS.md` |
| U1.5C | complete: real-reference pass | Bind exact rights-cleared libultrahdr real gain-map references to the existing fail-closed ingress boundary | U1.2C/U1.5B + pinned upstream fixtures | two exact CC-BY-4.0 MPO/two-frame fixtures reject before working pixels and renderer output; 53 focused/652 full tests; no source change or general MPF/ISO/HDR claim; `docs/U1_5C_LIBULTRAHDR_REFERENCE_REGRESSION_RESULTS.md` |
| U1.6 | in progress: percentile/DAG or orchestration next | Halo-aware tile/cache renderer | U1.3 | U1.6A-D/F/G1 pass and U1.6E/G0 close direct shortcuts; physical/density halation still needs staged percentile, scratch-DAG and separate family integration; orchestration, streaming/cache and bounded total memory remain pending |
| U1.6A | complete: pass | Deterministic finite-support halo-aware tiled-execution primitive | U1.3B | 21 focused/269 full tests; exact coverage; committed Gaussian full/seam error 0.0; maximum expanded window equals frozen bound; no renderer integration or 100MP claim; `docs/U1_6A_HALO_AWARE_TILING_RESULTS.md` |
| U1.6B | complete: pass | Experimental two-pass safe-Lab source context and coordinate-exact dither | U1.6A | all eight safe-rich styles and fixed real-raster smoke are byte-identical with max/seam error 0.0; nonzero colour-core grain rejects; 15 focused/284 full tests; default CLI unchanged; `docs/U1_6B_SAFE_LAB_GLOBAL_CONTEXT_RESULTS.md` |
| U1.6C | complete: pass | Experimental finite-support simple-halation tiled composite | U1.6A/U1.6B | 21 focused/305 full tests; active-effect real-raster max/seam `5.96e-08`, sRGB8 byte parity and bounded halo; physical halation and CLI integration forbidden; `docs/U1_6C_SIMPLE_HALATION_TILING_RESULTS.md` |
| U1.6D | complete: numerical pass | Exact sparse-event dust/scratch tiled composite | U1.6A/U1.6B | 106 targeted/328 full tests; 3,660-byte context and zero-halo real-raster float/sRGB8 byte parity; inherited stress visuals remain heuristic, no default/realism claim; `docs/U1_6D_SPARSE_DUST_TILING_RESULTS.md` |
| U1.6E | complete: closed | One-pass ordered simple-halation then sparse-dust composite | U1.6A/U1.6C/U1.6D | reversed order differs only `1.19e-07`, inside frozen `1e-6`; screen/white-alpha operators commute algebraically, so observable-order gate fails and no adapter remains; `docs/U1_6E_ORDERED_SUPPORTED_EFFECTS_RESULTS.md` |
| U1.6F | complete: numerical/resource pass | Exact legacy grain with bounded-RAM temporary-disk staging | U1.6A/U1.6E | 21 focused/127 adjacent/349 full tests; colour/B&W residual/float/sRGB8 byte parity, two-field scratch budget and zero residue; 0.35 stress visually rejected; `docs/U1_6F_GRAIN_STAGED_CONTEXT_RESULTS.md` |
| U1.6G0 | complete: direct route closed | Physical/density-halation tiled dataflow audit | U1.6A/U1.6C/U1.6F | sigma-52 full-grid blur differs by max/seam `5.81e-4` with nominal halo 156 because downsample grids depend on full shape; require percentile/global-resample/field-DAG primitives; `docs/U1_6G0_PHYSICAL_HALATION_DATAFLOW_AUDIT.md` |
| U1.6G1 | complete: pass | Shape-stable global-resample primitive | U1.6G0 | sigma-52 counterexample now has byte-identical full/tiled output and zero max/seam error at tile 37/64; 45 focused and 420 full CPU tests pass; no effect integration or legacy/physical claim; `docs/U1_6G1_SHAPE_STABLE_GLOBAL_RESAMPLE_RESULTS.md` |
| U1.6G2 | complete: pass | Exact two-pass float32 streaming percentile | U1.6G0/U1.6G1 | 1,048,613-value formal stream and 49,793 random finite bit patterns match NumPy-linear bytes; 1.5MiB histograms, 11 focused/431 full tests pass; no effect integration or 100MP claim; `docs/U1_6G2_EXACT_STREAMING_PERCENTILE_RESULTS.md` |
| U1.6G3 | complete: static-plan pass / prerequisites ready, executor absent | Physical/density field DAG, blur classification and lifetime/resource plan | U1.6G0/G1/G2 | AST-backed 41/31-node graphs preserve 8/2 and 6/1 inventories, transitive lifetimes/resources; G4D now binds passed G4A/G4C providers and removes static missing capabilities without graph/resource drift; no effect executor or integration exists; `docs/U1_6G3_HALATION_FIELD_DAG_LIFETIME_RESULTS.md` |
| U1.6G4A | complete: pass | Coordinate-exact scalar gradient windows | U1.6G3 | tile 37/64 and row 1/19 assemblies reproduce full `gy/gx` bytes with zero error from one-pixel original-coordinate windows; 34 focused/493 full tests pass; only row-chunked global staging remains before integration; `docs/U1_6G4A_COORDINATE_EXACT_GRADIENT_RESULTS.md` |
| U1.6G4B | complete: closed | Row-reader construction of exact G1-v1 global coarse stages | U1.6G1/G3/G4A | bounded reads pass, but scalar chunk-1 cases differ by up to `4.19e-9` in coarse data and `4.66e-9` after reconstruction because `tensordot` reduction order depends on batch height; preserve G1-v1 and do not select lucky chunks; `docs/U1_6G4B_ROW_STAGE_INVARIANCE_RESULTS.md` |
| U1.6G4C | complete: numerical/field pass | `shape-stable-global-resample-v2-explicit-f32` chunk-invariant area reducer | U1.6G4B | five frozen fields pass all full/row/tiled/repeat byte gates; v1 hash is unchanged, real drift passes and v2 closes the `900x1600` last-cell failure; two-field autonomous review finds zero severe field artifact; 25 focused/505 full tests; no effect integration; `docs/U1_6G4C_CHUNK_INVARIANT_GLOBAL_RESAMPLE_RESULTS.md` |
| U1.6G4D | complete: static binding pass | Bind executable G4A/G4C providers to G3 capability vocabulary | U1.6G3/G4A/G4C | exact provider identity resolves both required names; two families at 1.5MP/24MP become statically ready with unchanged nodes/lifetimes/resources; 80 focused/511 full tests; no graph execution; `docs/U1_6G4D_HALATION_CAPABILITY_BINDING_RESULTS.md` |
| U1.6G4E | complete: staged-v2 pass / legacy-compatible promotion closed | `staged-density-halation-v1-defaults` executor | U1.6G3/G4D | all ten policies meet staged-v2 (`<=1.02e-7` alpha, `<=5.96e-8` composite, zero seams, sRGB8 parity), repeat/resource and zero-severe visual gates; `u41-14` legacy alpha max `5.338e-4` exceeds frozen `5e-4`, so no drop-in/default promotion or integration; 107 focused-adjacent/527 full tests; `docs/U1_6G4E_STAGED_DENSITY_EXECUTOR_RESULTS.md` |
| U1.6G4F | complete: opt-in research retention pass | Explicit new-operator value/safety audit on fresh confirmatory cases | U1.6G4E | all automatic gates pass on `07/12/16`; 2/3 plausible unamplified localized effects and zero severe; deterministic payload/sheet repeat; 532 tests; no integration/default/legacy claim; `docs/U1_6G4F_EXPLICIT_NEW_OPERATOR_AUDIT_RESULTS.md` |
| U1.6G4G | complete: local 24MP measured pass | Measured 24MP memory/runtime and orchestration-readiness audit | U1.6G4F | corrected process-tree RSS 0.963-0.983GiB, worker 28.6-29.7s, repeat hashes and atomic failure cleanup pass; rejected launcher-only RSS report retained as invalid; 536 tests; no integration/100MP claim; `docs/U1_6G4G_24MP_RESOURCE_AUDIT_RESULTS.md` |
| U1.6G4H | complete: isolated adapter pass | Research adapter/orchestration parity and lifetime contract | U1.6G4G | 36/36 float/sRGB8 parity policies pass; one returned ndarray, ndarray-free metadata, input preservation, repeat and failure propagation pass; zero production import/schema drift; 553 tests; no CLI/default/100MP claim; `docs/U1_6G4H_RESEARCH_ADAPTER_RESULTS.md` |
| U1.6G4I | complete: local 100MP adapter pass | Isolated-adapter 100MP process-tree resource/determinism audit | U1.6G4H | two runs pass at 4.066-4.072GB peak and 123.8-131.8s worker time; source/output/metadata hashes repeat, preflight/failure cleanup pass; 558 tests; no complete renderer or production integration; `docs/U1_6G4I_100MP_ADAPTER_RESOURCE_RESULTS.md` |

Do not claim camera-accurate RAW solely from generic rawpy. Reference-grade camera paths require a known DNG/IDT/profile; generic development remains labeled.

### U2 — Style-safe renderer and profile system

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
| U2.1 | in progress: v1 contract pass | Versioned profile + recipe JSON schema | U1.2 | U2.1A supplies strict v1 schemas, exact safe-rich migration and replay verifier; broader API/evolution/operator profiles remain pending |
| U2.1A | complete: pass | Strict v1 profile/recipe identity, safe-rich migration and optional provenance writer | U1.2 approximation policy + U1.3B float core | 30 focused/248 full tests pass; committed output byte parity and all hashes verify; current profile remains heuristic look approximation; `docs/U2_1A_VERSIONED_RENDER_CONTRACT_RESULTS.md` |
| U2.2 | complete: numerical representation pass | Monotone exposure/sensitometry curves | U1.4, U2.1 | U2.2A exposure/density and U2.2B non-duplicative print composition pass: `[0,1]`, endpoint/partition/replay exact, min Jacobian .0470, identity .1596 and affine residual .1015. Clean-room witness only; no stock/integration claim; `docs/U2_2B_SENSITOMETRY_PRINT_COMPOSITION_RESULTS.md` |
| U2.3 | complete: identity composition pass / nontrivial witness closed | Tetrahedral 3D LUT/global residual | U2.2 | U2.3A preserves U2.2B exactly through identity LUT and independent tetrahedral parity is exact; the frozen cyclic witness exceeds first/second residual smoothness gates and is rejected without retuning; generic fail-closed wrapper retained, no visual/stock/integration claim; `docs/U2_3A_SMOOTH_RESIDUAL_COMPOSITION_RESULTS.md` |
| U2.4 | pending: software boundary pass, real operators absent | Negative/slide/B&W interpretation plugin boundary | U2.1A | U2.4A establishes a synthetic-test-only fail-closed extension boundary; real operators/profiles remain evidence-gated and no integration opens |
| U2.4A | complete: boundary pass | Pure interpretation request/result/registry boundary with three synthetic witnesses | U2.1A/U2.5/U2.6 | 3/3 test-only witnesses and exhaustive domain/numeric/metadata/eligibility controls pass; 23 boundary/698 full tests; zero renderer/schema/profile integration; `docs/U2_4A_INTERPRETATION_PLUGIN_BOUNDARY_RESULTS.md` |
| U2.5 | complete: current-safe-rich compatibility pass | Legacy `safe_lab` adapter | U2.1A, U1.3 | explicit profile-driven path exactly matches all eight current styles at sRGB8 and three at sRGB16; invalid profile/recipe provenance fails before output; arbitrary future profiles remain outside the claim |
| U2.5A | complete: invariant pass | Enforce exact neutral-axis RGB for existing HP5/Tri-X safe-Lab profiles | U1.3B defect evidence | B&W float/8/16-bit achromatic invariant, frozen Velvia hash, 225 tests and full-resolution RAW vision pass |
| U2.5B | complete: pass | Explicit profile-driven safe-Lab colour adapter with pre-output profile validation | U2.1A/U2.6A | 8/8 sRGB8 byte and 3/3 sRGB16 sample parity; invalid profile/asset/style zero outputs; verified recipe, explicit metrics and 675 tests; default/schemas/operators unchanged; `docs/U2_5B_PROFILE_DRIVEN_SAFE_LAB_ADAPTER_RESULTS.md` |
| U2.6 | complete: declared-evidence visibility pass | Profile evidence labels | U2.1A | validated API/CLI expose `heuristic/measured/paired/held-out` and identity/grade/claim fields without inference or schema change; current safe-rich truth remains none/none/heuristic/non-calibrated |
| U2.6A | complete: pass | Deterministic validated read-only profile-evidence summary API and CLI | U2.1A | exact declared fields, two-run byte identity and fail-closed escalation/tamper gates pass; 9 focused/656 full tests; no schema, renderer or evidence-grade change; `docs/U2_6A_PROFILE_EVIDENCE_INSPECTION_RESULTS.md` |

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
| U5.CT1 | in progress; analytic L0/gauge/shaper pass, HDR bake parity closed | Versioned invertible explicit-operator contract plus known-operator pseudo-roll simulator | U1/U2 interfaces; no external data | L0/gauge/log1p shaper pass near machine precision. `[0,16]` shaped 33/65 cubes converge but fail frozen max-error gates at .0351/.00899, so no HDR LUT parity/integration claim; analytic L1/L2 remains authoritative; `docs/ROLL2FILM_CT1_L0_GAUGE_SHAPER_RESULTS.md` |
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
| U5.R1A | complete: strict local tooling | ChromaticTail/FilmStyleSafe ontology, annotation schema, split and sampling/preregistration worksheet | U4 foundation | 7 focused/363 full tests; VLM/human evidence isolation, conservative 3+3 labels, A0/A1/B0-B4 leakage audit and non-binding power worksheet; `docs/FILMSTYLESAFE_R1A_RESULTS.md` |
| U5.R1B | complete: design/membership contract | Stress generator plus real cross-algorithm failure suite design | U5.R1A | Frozen transform/failure families, ID11 regression, 53/55/56 strength controls, suite leakage audit; no participants/training; `docs/FILMSTYLESAFE_R1B_DESIGN.md` |
| U5.R1B2 | complete: provisional inventory scaffold | Bounded A0 inventory and synthetic operator prototype cards | U5.R1B | 7 validated cards; placeholder hashes; no pixels; `docs/FILMSTYLESAFE_R1B2_INVENTORY.md` |
| U5.R1B3 | complete: A0 identity binding | Bind real A0 hashes/IDs from U4.1/U4.2/ID11 local artifacts | U5.R1B2 | 5/7 members bound with verified SHA-256; synth/hardneg still placeholders; `docs/FILMSTYLESAFE_R1B3_BINDING.md` |
| U5.R1B4 | complete: synthetic prototype | First explicit highlight-chroma-island failure operator | U5.R1B3 | Deterministic PNG on u41-01; output `c04a9237...`; large magenta disk noted by autonomous vision; `docs/FILMSTYLESAFE_R1B4_SYNTHETIC.md` |
| U5.R1B5 | complete: HF-speckle + RF2.C0 bind | Speckle-v1 synthetic and Ektar/fixed-e0 A0 external control | U5.R1B4 | HF dots `73af7d4e...`; Ektar `d23ddd75...`; hardneg still unbound; `docs/FILMSTYLESAFE_R1B5_RESULTS.md` |
| GH0 | complete | Cursor Goal harness: rule, skill, state, stop hook, protocol, tests | none | `docs/drpt/CURSOR_GOAL_PROTOCOL.md`; Goal state ACTIVE |
| GH1 | complete: acceptance repair | Validate complete Goal state before follow-up, audit live Git conflicts and define checkpoint-HEAD semantics | GH0 + Cursor handoff audit | 19 focused tests; malformed state and live conflict fail closed; loop limit/protocol agree |
| U5.R1B6 | complete: hard-negative bound | Bind legitimate-local bloom/halation A0 member | U5.R1B5 | output `87f111a2...`; 9/9 A0 members bound; opens R1C planning only; `docs/FILMSTYLESAFE_R1B6_HARDNEG.md` |
| U5.R1C | complete: A0 pilot inconclusive | Conventional-metric failure study plus prospective SCIS v0 | U5.R1B6 | Conv gap confirmed (sens 0.33); SCIS v0 0.67 vs hardneg/external; candidate not promoted; `docs/FILMSTYLESAFE_R1C_A0_PILOT.md` |
| U5.R1C2 | complete: SCIS v0.1 improved | Refine SCIS for sparse HF / style-robust residual | U5.R1C | v0.1 perfect vs hardneg/external; still style-contaminated by 53/55/56; `docs/FILMSTYLESAFE_R1C2_SCIS_V01.md` |
| U5.R1C3 | complete: weak-pass, no selection | Conditioned style-control calibration for SCIS without excluding 53/55/56 | U5.R1C2 | affine 2/3, quadratic 1/3 at zero FPR across all five negatives; sparse HF absorbed and 53 remains max negative; no candidate/gate; one minimal cross-fit mechanism diagnostic opens; `docs/FILMSTYLESAFE_R1C3_CONDITIONED_STYLE_CONTROL_RESULTS.md` |
| U5.R1C3D | complete: fail, route closed | Test whether spatial cross-fitting prevents same-pair sparse-failure absorption | U5.R1C3 weak-pass | fixed 4x4 neighbour-exclusion remains 2/3; sparse HF 1.2444 and RF2.C0 max negative 63.9936; no more conditioned-SCIS candidates/tuning; metrics remain descriptive; `docs/FILMSTYLESAFE_R1C3D_SPATIAL_CROSSFIT_RESULTS.md` |
| U5.R2A | complete: numerical pass, semantic-safety counterexample retained | Versioned numerically constrained operator contract | U1/U2 renderer foundation | Affine + monotone spline + exact tetrahedral LUT is deterministic/replayable; 704 tests and byte-identical audits pass. A smooth in-gamut positive-Jacobian blue-to-purple transform passes every numerical gate, proving numerical regularity is not semantic safety; no fitting/integration; `docs/U5_R2A_CONSTRAINED_GLOBAL_OPERATOR_RESULTS.md` |
| U5.R2B | complete: one B0 challenger retained | Strong global operator frontier | U5.R2A, U5.R1 | Margin-4 anchor56 passes style/non-basic/clipping and all-nine full-resolution severe veto; anchor09 fails ID11 red-speckle/posterization; raw 01/53/55/56 fail clipping and safe-rich is bland. No production/stock/cross-rater claim; `docs/U5_R2B_GLOBAL_OPERATOR_FRONTIER_RESULTS.md` |
| U5.R2C | complete: local tooling pass; actual B1 gated | B1 complete fixed-bank cross-rater empirical-ceiling policy and annotation budget | U5.R2B | K=7/panel/lineage/policy validators and byte-identical 54N--78N workload pass; binding N null; no B1 pixels/recruitment/outcomes and R3 remains closed; `docs/U5_R2C_EMPIRICAL_CEILING_TOOLING_RESULTS.md` |
| U5.R2D | complete: properties pass; shortcut rejected | StatLUT-style statistics-to-explicit-LUT shortcut and representation audit | U5.R2B, U5.R2C local tooling | Permutation error 4.26e-14, but red/blue palette L2 is 111.69; two LUTs give identical reference pixels/descriptors yet differ 1.0 on an absent-blue probe. 726 tests; no operator/stock claim; `docs/U5_R2D_STATISTICS_TO_LUT_SHORTCUT_AUDIT_RESULTS.md` |
| U5.R2D1 | complete: canonical information only | CPU known-operator recovery under operator-family and palette-support shift | U5.R2D | Exact reference delta improves 54.3% vs target-only, but unpaired interaction is 125% worse than global and captures negative style. 384 operators, zero leakage, 737 tests; larger model forbidden; `docs/U5_R2D1_SYNTHETIC_OPERATOR_RECOVERY_RESULTS.md` |
| U5.R2D2 | complete: canonicalizer-sensitive/unidentified | Multiple imperfect canonicalizers and hard operator retrieval | U5.R2D1 | Zero of four practical hypotheses pass; pairwise rendered disagreement 0.1567 vs 0.03; hard Top-1/Top-3 fail. Palette oracle is informative but not a complete policy; 743 tests; `docs/U5_R2D2_CANONICALIZER_RETRIEVAL_SENSITIVITY_RESULTS.md` |
| U5.R2E0 | complete: numerical/diversity primitive pass | Clean-room density/sensitometry-domain explicit film-inspired operator primitive | U5.R2D2 | Five bounded non-affine witnesses pass exact partition/replay, positive sampled Jacobian and pairwise diversity gates; 748 tests. No visual/stock/product claim; `docs/U5_R2E0_DENSITY_DOMAIN_OPERATOR_RESULTS.md` |
| U5.R2E1 | complete: one B0 density challenger retained | Bounded-strength density-witness gold/stress frontier | U5.R2E0 | Cyan-shadow/warm-highlight s0.50 blind-wins 2/3; style 13.49, non-basic 7.39, zero new clipping; no severe failure in 27 full-res shortlist renders or ID11; no fitting/stock/production claim; `docs/U5_R2E1_DENSITY_WITNESS_FRONTIER_RESULTS.md` |
| U5.R2F0 | complete: external E2E feasible / SSL unavailable | CanonCGT canonical-pivot explicit-LUT source, licence and smoke audit | U5.R2D2 | Apache commit/checkpoint fixed; 5.06M-param E2E loads and is deterministic, but SSL is a 2-byte placeholder and predicted LUTs are unbounded. Generic grading control only; `docs/CANONCGT_EXTERNAL_ML_SOURCE_AUDIT.md` |
| U5.R2F1 | complete: reference-sensitive, raw policies safety-fail | Immutable CanonCGT E2E reference-condition and explicit-LUT safety pilot | U5.R2F0, U5.R2E1 | Pairwise reference DE 8.53 and 6/9 pass style+non-basic, but 0/9 pass clipping/range; 81x2 exact, LUT replay error 0, no visual shortlist; `docs/U5_R2F1_CANONCGT_REFERENCE_CONDITION_RESULTS.md` |
| U5.R2F2 | complete: no automatic survivor | Data-independent bounded projection frontier over immutable predicted LUTs | U5.R2F1 | Node clipping fails structure/safety; two safe contractions pass structure/range 9/9 but style/non-basic 0/9; exact 243x2 and empty visual shortlist; `docs/U5_R2F2_PROJECTED_LUT_FRONTIER_RESULTS.md` |
| U5.R2G0 | complete: style retained, boundary safety fail | Constrained explicit distillation of immutable ML LUT controls | U5.R2F2 | Style 9/9, non-basic 6/9, sensitivity 10.07 and +4.77 style vs F2; clipping 0/9, no survivor or visual review; exact 81x2; `docs/U5_R2G0_CONSTRAINED_EXPLICIT_DISTILLATION_RESULTS.md` |
| U5.R2G1 | complete: no survivor, CanonCGT branch closed | Fixed output-quantization headroom over immutable G0 operators | U5.R2G0 | Structure/range/style 9/9 and retention pass; clipping 0/9 because codes 1/254 remain within frozen epsilon; exact 81x2, no visual review; `docs/U5_R2G1_QUANTIZATION_HEADROOM_RESULTS.md` |
| U5.R2H0 | complete: Velvia-only feasibility | Film-specific spectral/sensitometric prior and obtainable-data feasibility audit | U5.R2G1 | Velvia 50 alone exposes characteristic, sensitivity and separated CMY dye-density graphs; Portra/Ektar and public print-chain data are incomplete. Opens synthetic-only H0A; no fitting/calibration claim; `docs/U5_R2H0_SPECTRAL_SENSITOMETRIC_SOURCE_AUDIT.md` |
| U5.R2H0A | complete: numerically invalid/unidentified | Velvia 50 datasheet spectral witness numerical identifiability pilot | U5.R2H0 | Strong base effect 20.91 median DE, but neutral chroma 6.22 fails and D65-equal metamers differ 82.70 median / 144.35 p95, dominating the effect; no visual review or rescue; `docs/U5_R2H0A_VELVIA_DATASHEET_WITNESS_RESULTS.md` |
| U5.R2H0C | complete: CAVE research-only source pass | Measured-natural-reflectance conditional spectral variability source audit | U5.R2H0A negative evidence | Official 405,988,465-byte CAVE reflectance archive is eligible for internal research with no redistribution; opens exact acquisition/integrity and preregistration only; `docs/U5_R2H0C_MEASURED_REFLECTANCE_SOURCE_AUDIT.md` |
| U5.R2H0C1 | complete: bounded empirical-prior candidate | CAVE integrity and conditional-support feasibility | U5.R2H0C | 992/992 official path/size/CRC matches; 263 cross-scene DE<=1 pairs give output DE 1.38 median / 5.47 p95 and pass scene-bootstrap gates. Opens simple hard canonicalizer comparison only; `docs/U5_R2H0C1_CAVE_CONDITIONAL_VARIABILITY_RESULTS.md` |
| U5.R2H0C2 | complete: hard T=1 candidate | Group-held-out simple measured-spectrum canonicalizer comparison | U5.R2H0C1 | Cross-scene hard Top-1 at DE<=1 covers 36.13%, wins 76.60%, and reduces selected median/p95 error to .339/2.975 vs smooth .894/3.939; wider thresholds fail stability; `docs/U5_R2H0C2_HARD_SPECTRUM_CANONICALIZER_RESULTS.md` |
| U5.R2H0C3 | complete: external replication failed, branch closed | External measured-spectrum replication | U5.R2H0C2 | Fixed CAVE-bank T=1 selects 184/1,732 USGS queries but wins only 6.52%; smooth .271/4.331 beats hard 1.561/5.690 selected median/p95, 0/4 evaluable chapter directions pass and full-policy p95 worsens. No retuning/same-source bank/neural rescue; `docs/U5_R2H0C3_EXTERNAL_SPECTRUM_REPLICATION_RESULTS.md` |
| U5.R2H1 | complete: retain hard 1-NN for A0 visual diagnostic | Hard content routing between anchor56 and density-cyan safe operators | U5.R2B, U5.R2E1 | Composite BA 70.67%, p=.010 and 45.61% regret closure; style view passes, non-basic view fails; simple 1-NN retained over logistic; no stock/preference/production claim; `docs/U5_R2H1_SAFE_BANK_CONTENT_ROUTING_RESULTS.md` |
| U5.R2H2 | complete: exact router closed | Exact hard-1NN full-resolution and blind visual diagnostic | U5.R2H1 | 0/41 routed severe and 0/10 new severe; all three blind rounds are identically 6 routed wins / 4 global wins, below frozen 7/10; no retuning; `docs/U5_R2H2_HARD_ROUTING_VISUAL_AUDIT_RESULTS.md` |
| U5.R2I0 | complete: automatic pass / visual value closed | Real-image frontier for fixed U2.2B sensitometry-print operator | U2.2B, U2.3A | all five strengths pass style/non-basic/gold clipping; s1.0 reaches 19.28/14.40 and 0.214% gold clipping, 18/18 full-res severe-clean, but stable green cast loses all 3 blind rounds to safe-rich; no challenger retained; `docs/U5_R2I0_SENSITOMETRY_PRINT_FRONTIER_RESULTS.md` |
| U5.R2I1 | complete: numerical pass | Data-independent inverse-neutral gauge for U2.2B | U5.R2I0 | neutral spread .1630 -> 6.41e-6, endpoint exact, min Jacobian .0148, affine residual .0549 and exact replay; 811 tests; real-image frontier opens separately, no stock/integration claim; `docs/U5_R2I1_NEUTRAL_AXIS_GAUGE_RESULTS.md` |
| U5.R2I1B | complete: no survivor, route closed | Real-image frontier for fixed I1 neutral gauge | U5.R2I1 | strongest s1.0 style 5.66/non-basic .72, all strengths miss both frozen floors with zero new gold clipping; no visual candidates; no retuning/capacity rescue; `docs/U5_R2I1B_NEUTRAL_GAUGE_FRONTIER_RESULTS.md` |
| U5.R2J0 | complete: representation pass | Data-independent two-matrix positive-film-response architecture witness | SF2.6R method precedent + I1B close | five original witnesses pass every boundedness/Jacobian/diversity/replay gate; minimum affine residual .09596; 818 tests; opens J1 only; `docs/U5_R2J0_POSITIVE_FILM_RESPONSE_RESULTS.md` |
| U5.R2J1 | complete: no visual gain | Bounded-strength real-image frontier for the fixed J0 bank | U5.R2J0 | seven automatic survivors and 27/27 shortlist gold renders severe-clean, but every blind round selects retained R2E1 first and safe-rich second; no J1 candidate enters the top two; representation retained as negative control, no retune/integration; `docs/U5_R2J1_POSITIVE_FILM_FRONTIER_RESULTS.md` |
| U5.R2K0 | complete: fidelity pass / regularity fail | Clean-room fixed-geometry bounded Gaussian residual representation | U5.R2J1 close + GLUT primary-source audit | N27 fits frozen nonlinear controls at .00990/.00344 RMSE with 85.98%/87.65% affine-relative gains, but min det(J) is -3.31/-1.87, diagonal derivatives turn negative and one coefficient exceeds the 4.0 bound; exact range/replay do not rescue folds; 837 tests; `docs/U5_R2K0_BOUNDED_GAUSSIAN_RESIDUAL_RESULTS.md` |
| U5.R2K1 | complete: structural pass / fidelity fail | Bounded invertible triangular monotone coupling representation | U5.R2K0 regularity fail | positive orientation and analytic inverse pass (min det .168/.359; inverse <5.6e-16), but nonlinear RMSE .0623/.0282 misses .015, gains are 11.17%/-2.34% vs affine, and exact partition parity misses by 1.11e-16; no capacity/optimizer rescue; 841 tests; `docs/U5_R2K1_TRIANGULAR_MONOTONE_COUPLING_RESULTS.md` |
| U5.R2K2 | complete: structural pass / confirmation and norm fail | Finite monotonic rational-quadratic spline colour coupling | U5.R2K1 fidelity fail + Neural Spline Flows formula audit | exact boundedness/inverse/partition and positive orientation pass; density confirms .01435 but norm 9.457 exceeds 8, positive confirms .01965 above .015 despite .00067 fit; no bin/stage/optimizer rescue or real-image frontier; 849 tests; `docs/U5_R2K2_RATIONAL_QUADRATIC_COUPLING_RESULTS.md` |
| U5.R2K3 | complete: representation/diversity pass, norm fail | Analytic RGB-cube gamut-polar tone/hue/chroma palette factorization | U5.R2K2 close + ACES invertible-polar design principles | exact bounds/neutral/endpoints/inverse/replay and positive orientation; non-affine residual .0521--.0741 and minimum pairwise RMSE .0450, but cyan-shadow/warm-highlight norm 11.198 exceeds 8; fixed bank closes before rendering; 853 tests; `docs/U5_R2K3_GAMUT_POLAR_PALETTE_RESULTS.md` |
| U5.R2L0 | complete: material Oracle gap / severe pass | Post-E1 per-image hard strength Oracle for the retained density-cyan operator | U5.R2E1 | s0.65 selected 36/41; mean/median style gain 3.07/3.56, gold mean 2.67, worst selected clipping .4925%; 41/41 full-resolution severe-clean including ID11/face; post-output Oracle only, opens simplest-policy contract, not deployable routing; 861 tests; `docs/U5_R2L0_DENSITY_STRENGTH_ORACLE_RESULTS.md` |
| U5.R2L1 | complete: exact policy gate failed | Deterministic full-frame clipping preflight for hard s0.65/s0.50 selection | U5.R2L0 | both strength banks replay RGB8-exactly 41/41, but full-frame assignment/output matches are 40/41: stress20 .50015% exceeds the .5% gate while sampled Oracle was .49247%; close without threshold/epsilon/sampling rescue; 868 tests; `docs/U5_R2L1_FULL_FRAME_STRENGTH_PREFLIGHT_RESULTS.md` |
| U5.R2M0 | complete: structural/positive pass, density fidelity + identity fail | Bounded interval-Möbius triangular coupling representation | U5.R2K1/K3 closes + ENNeLUT invertibility prior | positive RMSE .01170 passes and max Jacobian norm is 2.842, but density RMSE .03089 misses .015 and fitted identity max error 5.97e-5 misses exactness; no render/rescue; 873 tests; `docs/U5_R2M0_INTERVAL_MOBIUS_COUPLING_RESULTS.md` |
| U5.R2N0 | complete: external control feasible / clean-room only | ModFlows palette-embedding and invertible-flow source/method audit | U5.R2M0 close + current primary sources | pinned 18.97MB MIT-labelled B0 weight is obtainable; official code has no licence and training lineage is incomplete; opens synthetic checkpoint/shortcut audit only, no stock pixels/fitting/integration; `docs/U5_R2N0_MODFLOWS_SOURCE_AUDIT.md` |
| U5.R2N1 | complete: structure pass / shortcut and raw range fail | Clean-room ModFlows B0 checkpoint and synthetic palette audit | U5.R2N0 | exact architecture/repeat, identity `5.48e-5`, min det `.6368`, norm `1.607` pass; permutation cosine `.7255`, palette/geometry ratio `.803` and raw range `[-.073,1.119]` fail; no real images/rescue; 886 tests; `docs/U5_R2N1_MODFLOWS_B0_SYNTHETIC_RESULTS.md` |
| U5.R2O0 | complete: full synthetic representation pass | Cube-preserving diffeomorphic explicit colour flow | K0-K3/N1 negative representation evidence | fixed 192-parameter stationary flow confirms density/positive controls at `.01091/.00544` RMSE, improves 84.45%/80.27% over affine, min det `.117/.275`, max norm `3.993/2.781`, exact range/endpoints and byte-identical reports; no redundant image frontier, fitting or stock claim; 891 tests; `docs/U5_R2O0_CUBE_DIFFEO_COLOUR_FLOW_RESULTS.md` |
| U5.R2P0 | complete: same-scene pair prior / source unavailable | ChameleonTuner WACV 2026 source, supervision and project-fit audit | R2O0 pass + current primary sources | method uses LSC+LoFTR on same-scene pairs then NSGA-II over a local 17-cube LUT; it does not support unrelated similar-photo supervision. Official repo is one 131-byte README with no code/licence; retain for future misaligned controlled pairs only; `docs/U5_R2P0_CHAMELEON_TUNER_SOURCE_AUDIT.md` |
| U5.R2Q0 | complete: explicit inference prior / official training ineligible | SA-LUT ICCV 2025 source, training-lineage and project-boundary audit | R2P0 close + current primary sources | inference predicts a simplex of 64 bounded 4D-LUT bases plus a context map and renders explicitly, but official training uses a direct Style2Log image generator, GAN supervision and undeclared image/LUT roots. Do not run the checkpoint or download PST50; retain clean-room non-generative architecture prior only; `docs/U5_R2Q0_SA_LUT_SOURCE_TRAINING_AUDIT.md` |
| U5.R2R0 | complete: fixed-asset audit feasible | D-LUT WACV 2025 source, method and generative-boundary audit | R2Q0 close + current primary sources | per-style 93K score MLP moves identity-LUT nodes toward one image palette; final RGB is explicit, but scene-palette proportions are the supervision and the paper/demo epsilon differs. Open only a disclosed post-exploratory structural audit of 41 official LUTs; no training/images/film claim; `docs/U5_R2R0_D_LUT_SOURCE_METHOD_AUDIT.md` |
| U5.R2R1 | complete: no structural survivor | Published D-LUT trajectory range/orientation audit | U5.R2R0 | identity/repeat pass, but nonidentity range and orientation are each 0/40 while norm is 40/40; step 40 is strong/non-affine yet has 48.02% sampled trilinear and 48.85% tetrahedral nonpositive determinants. Close before images without rescue; `docs/U5_R2R1_D_LUT_PUBLISHED_ASSET_RESULTS.md` |
| U5.R2S0 | complete: full analytic mechanism pass | Analytic palette-score forcing through cube-preserving diffeomorphic flow | U5.R2O0 + U5.R2R1 | all three palettes pass attraction/style/non-affine/reference/range/Jacobian/inverse/repeat; identity RMSE .132-.153, affine residual .053-.066, minimum pairwise RMSE .1078, min determinant .0142, max norm 4.162; opens only a separately frozen synthetic histogram-score recovery leaf; 901 tests; `docs/U5_R2S0_PALETTE_SCORE_DIFFEO_ORACLE_RESULTS.md` |
| U5.R2S1D | complete: KDE selected / raw-histogram retrieval loses | Canonical-histogram hard case retrieval and query-density controls | U5.R2S0 | repeated development selects query KDE .12 at median/p90 oracle-output RMSE .0327/.0508 versus hard Hellinger .0873/.1283 and global .1128/.1359; KDE retains .984 style, .909 non-affine and .935 separation. Seed 27012 remains untouched; open one frozen confirmation; `docs/U5_R2S1_HISTOGRAM_CASE_RETRIEVAL_DEVELOPMENT_RESULTS.md` |
| U5.R2S1C | complete: all synthetic recovery gates pass | Confirm query KDE .12 against hard Top-1 and global mean | U5.R2S1D | untouched median/p90 error .03046/.05029, .959 direction, .985 style, .909 non-affine, .934 separation; 63.46%/71.88% median error reduction vs hard/global; min det .0150, max norm 3.993, inverse 7.35e-7, exact permutation/repeat. Retain mechanism only; content-palette nuisance blocks images; `docs/U5_R2S1_HISTOGRAM_SCORE_CONFIRMATION_RESULTS.md` |
| U5.R2S2D | complete: no residual winner / no confirmation | Recover a shared style flow across independent content scenes with unpaired matched-neutral controls | U5.R2S1C + real scene-colour shortcut evidence | raw styled KDE has lowest error .0639 but strong identity-negative .1049 and norm 8.228, confirming content response; best ridge is worse at .0726 with identity-negative .0913/norm 10.702; best density ratio is .1053, only 4.25% above global. Seed 28105 untouched; close without capacity rescue; `docs/U5_R2S2_CONTENT_PALETTE_NUISANCE_DEVELOPMENT_RESULTS.md` |
| U5.R2S3D | complete: repeat-exact `distribution_fail` | Fit safe flow directly by independent distribution matching and separate fit from operator recovery | U5.R2S2D + U5.R2O0 | RFF/SW reduce fit loss but reach only 11.80%/3.36% held-out gain; oracle `.11591/.15423` and `.11444/.16447`, A/B `.07902/.13138`, all fail while structure/style pass. Close objectives without rescue/visual; seed 28203 untouched; `docs/U5_R2S3_UNPAIRED_DISTRIBUTION_OPERATOR_PILOT_RESULTS.md` |
| U5.R2S4D | complete: conditional match/operator fail | Test one shared explicit flow across correctly corresponding synthetic content-condition distributions | U5.R2S3 repeated `distribution_fail` | correct conditions reach 71.36% held-out distribution improvement and separate shuffled, but hidden-operator `.10751/.11900` RMSE and 8.45%/24.98% pooled/shuffled improvements miss frozen gates; all structure/A-B gates and exact repeat pass. No rescue, confirmation or visual candidate; `docs/U5_R2S4_DIVERSIFIED_DISTRIBUTION_OPERATOR_DEVELOPMENT_RESULTS.md` |
| U5.R2T0 | complete: architecture prior / execution closed | Audit StatLUT statistical reference conditioning, supervision, topology and asset lineage | U5.R2S2D + 2026-07-09 source | image branch predicts a global LUT but is trained from known random-LUT same-image pairs; patch shuffle does not test content/style identifiability, 4,000 professional LUTs lack a rights manifest, code/checkpoints are absent and the text diffusion branch is forbidden. Retain Lab-stat descriptor prior only; no capacity rescue/download/training; `docs/U5_R2T0_STATLUT_SOURCE_METHOD_AUDIT.md` |
| U5.R2U0 | complete: clean-room synthetic control only | Audit ColorFM hierarchical coupling, explicit flow, pseudo-supervision and source availability | U5.R2S3/S4 + ECCV 2026 source | HCC is a useful semantic/distribution coupling prior but creates pseudo-pairs rather than identifying transport. Official commit `153798a...` has README/static assets only, no code/checkpoint/licence; paper MLP/Euler path lacks project bounds. A future synthetic HCC-to-O0 control may open after S3/S4; no current pixels/training/stock claim; `docs/U5_R2U0_COLORFM_SOURCE_METHOD_AUDIT.md` |
| U5.R2U1D | complete: pair fit/operator fail | Test clean-room HCC pseudo-pairs while fitting only the bounded O0 flow | U5.R2U0 + repeated eligible U5.R2S4 branch | exact repeat; correct HCC fits pseudo-pairs by 84.72% but `.08791/.09699` oracle and `.04176` A/B miss gates, losing to random and pooled controls. Structure is safe; no rescue, confirmation or visual candidate; `docs/U5_R2U1_HIERARCHICAL_COLOUR_COUPLING_DEVELOPMENT_RESULTS.md` |
| U5.R2V0 | complete: published path excluded | Audit cmKAN hypernetwork/KAN colour matching, unpaired training, topology, source and data boundary | U5.R2U0 + ICCV 2025 source | pinned source is complete but research-only; its unpaired lane is CycleGAN and predicts spatially varying KAN parameters per pixel before directly returning RGB. No global cube/Jacobian/inverse/spatial guarantee and camera-pair data are not film evidence. Retain only a future bounded global-spline prior; no checkpoint/data download or training; `docs/U5_R2V0_CMKAN_SOURCE_METHOD_AUDIT.md` |
| U5.R2W0 | complete: retain data-gated CFSM question / reject final-algorithm claim | Audit submitted NFRM/CFSM reference-look architecture against current evidence and primary methods | U5.R2O0 + S1-S3 negatives + submitted audit | same-known-look/different-content supervision is a distinct useful experiment, but single final references entangle content/capture/nuisance and do not identify film. Keep look/operator and content/applicability spaces separate; reuse the bounded O0 renderer; `docs/U5_R2W0_REFERENCE_LOOK_MATCHING_SOURCE_METHOD_AUDIT.md` |
| U5.R2W1D | complete: paired upper bound only / output-only fails | Test output-only single/multi-reference look recovery across unrelated generated content | U5.R2W0 + U5.R2O0 | exact repeat: paired `.02498/.07282` passes, but output-only single/four lose identity/global; unseen four ridge is `.06631/.08191` yet content BA is 98.44% and identity-reference false RMSE `.06783`. `53/55/56` stays one strength path. No rescue/visual; `docs/U5_R2W1_REFERENCE_LOOK_IDENTIFIABILITY_DEVELOPMENT_RESULTS.md` |
| U5.R2W2F0 | complete: no coherent global recipe champion | Determine whether three local FilmSet recipe domains are coherent bounded global operators | repeated paired-only U5.R2W1 + frozen FilmSet source/target/internal-dev manifests | exact repeat over 40 IDs/160 payloads: Cinema is basic-only; ClassNeg/Velvia are adaptive-or-spatial. All O0 structure gates pass but every domain misses global coherence, chiefly grid dispersion `.164/.163/.208` vs `.08`. W2F1/local-capacity rescue stays closed; final 628 untouched; `docs/U5_R2W2F0_FILMSET_RECIPE_GLOBAL_EXPLAINABILITY_RESULTS.md` |
| U5.R2Z0 | complete: ClassNeg Oracle only / Velvia no Oracle | Test a fixed development case bank and input-only hard retrieval inside known FilmSet recipes | W2F0 adaptive/spatial branches + frozen paired partition | exact repeat: ClassNeg Oracle improves shared 16.72%, wins 75% and has positive CI, but spatial photometric NN is worse than shared and closes `-208.62%` of the gap. Velvia misses Oracle gates at 9.78%/68.75%. Wrong-recipe banks lose every query; all structure gates pass. No selector/router/visual/final-628/film claim; `docs/U5_R2Z0_FILMSET_CASE_RETRIEVAL_RESULTS.md` |
| U5.R2Z1 | complete: off-diagonal Oracle and applicability fail | Test whether asymmetric query/operator applicability can recover the ClassNeg evaluator gap without raw-content similarity | Z0 `case_bank_oracle_only` + failed global/spatial photometric selectors | exact repeat: excluding each query's own operator leaves only 7.32% Oracle gain/62.5% wins with non-positive CI; rank-2/rank-4 policies are 28.54%/34.70% worse than shared. All structure gates pass, fresh targets remain unread and no capacity/visual/final-628/film rescue opens; `docs/U5_R2Z1_ASYMMETRIC_APPLICABILITY_RESULTS.md` |
| U5.R2AA0 | complete: bounded nuisance pilot feasible / operator unidentified | Audit official Kodak VISION3 250D and VISION 2383 negative-to-print source completeness | Z1 close + H0 source methodology + official 2026 sheets | both sheets expose characteristic, sensitivity and separated CMY dye graphs; 2383 LAD supplies neutral aims. Digital spectrum, absolute dye mapping, printer SPDs, timing and xenon/view response remain missing. Open AA1 synthetic nuisance audit only; no real-image/stock/calibration claim; `docs/U5_R2AA0_KODAK_NEGATIVE_PRINT_SOURCE_AUDIT.md` |
| U5.R2AA1 | frozen: curve binding ready | Test whether a 250D-to-2383 datasheet chain retains non-basic effect across missing-variable hypotheses | AA0 bounded nuisance feasibility | 729-colour synthetic grid; 162-member spectrum/placement/dye/printer/view ensemble; source-ink overlays, basic/affine controls, nuisance/effect ratios and neutral/Jacobian/range gates frozen before curve annotation. No real images, stock pixels, LUT/profile or visual stage; `docs/planning/U5_R2AA1_KODAK_NEGATIVE_PRINT_NUISANCE_CONTRACT.md` |
| U5.R2W2R | source preflight complete; pixel leaf not ready | Audit INRetouch RTD as a real-raster same-preset bridge | U5.R2W0 | official topology is highly relevant: 167 presets x 569 FiveK contents, paired natural/preset rasters, held-out contents and presets. A preset is a shared recipe, not presumed global-operator truth; global-explainability must pass first. Repository is gated, about 18 GB and CC BY-NC-SA 4.0; no gated file was accessed. Frozen DoR requires W1 evidence, human click-through/contact disclosure, exact lineage and non-commercial isolation; not film/stock truth; `docs/planning/U5_R2W2R_INRETOUCH_RTD_SOURCE_PREFLIGHT.md`, `configs/u5_r2w2r_inretouch_rtd_source_preflight_decision_v1.json` |
| U5.R2X0 | complete: analytic affine-only close | Determine whether the 2025 automatic affine-barycentric palette-transfer path is more than one affine RGB operator | W2F0 close + official JCST equations | exact repeat: direct path equals one affine operator to `9.99e-16`; valid transports admit determinant `-1` and `5p-2` cube range `[-2,3]`. No image/CPLEX, clipping, capacity or local-path rescue; `docs/U5_R2X0_PALETTE_TRANSFER_AFFINE_COLLAPSE_RESULTS.md` |
| U5.R2Y0 | complete: content/histogram/texture shortcut close | Audit whether NegClone 0.2.0 stock fingerprints isolate stock rather than scene colour, exposure and texture | X0 close + exact MIT PyPI sdist | exact repeat: film-free scene colour moves bias by `.8485` L2, exposure distribution moves tone by `254.4/255`, checker texture becomes `.4` grain versus flat zero. Retain only as a negative control; no more-photo/neural/preset rescue or stock claim; `docs/U5_R2Y0_NEGCLONE_CONTENT_SHORTCUT_RESULTS.md` |
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
| U6.2A | complete: synthetic representation pass | Clean-room Boolean/Poisson grain representation witness | U6.0 + U1.6F + IPOL model equations | max mean error .0133; midtone variance ratios 2.15/2.25; large-small lag-1 delta .1416; repeat/partition/replay exact; 878 tests; opens crop visual frontier only; `docs/U6_2A_BOOLEAN_GRAIN_REPRESENTATION_RESULTS.md` |
| U6.2B | complete: automatic pass / visual severe close | Fixed Boolean-grain existing-image crop frontier | U6.2A | both policies pass luma/low-pass/endpoint/repeat gates, but small fails severe 4/5 and large 5/5 with salt-like bright speckles; legacy `.018` is 0/5; 882 tests; no rescue/integration; `docs/U6_2B_BOOLEAN_GRAIN_CROP_FRONTIER_RESULTS.md` |
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
| EXP-LSM-00 | One stock may retain stable residual modes after all prerequisite gates | K=1 stock global; 53/55/56 strength-path negative control | group-aware residual signature and mixture candidate | preregistered cross-group stability beyond content/source/basic/strength controls | accept K=1 or close as nuisance/unidentified; never add capacity to rescue |
| EXP-LSM-01 | A statistically stable mode bank may have product routing value | stock global champion | frozen Evaluator Oracle over fixed modes | significant Oracle gain with no severe artifact | keep descriptive modes only; product remains global |
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
- Isolated external spectral-prior control: `docs/REAL_FILM_SPEKTRAFILM_EXTERNAL_CONTROL_RESULTS.md`; external outputs are ignored comparison evidence, never stock truth or teachers

---

*Tracker initialized: 2026-07-10; FilmCase subtree frozen: 2026-07-11; Roll2Film subtree opened: 2026-07-12 and re-gated after external audit on 2026-07-15. U0.3 remains a fail-closed 4,210-row legacy audit while current Windows `film_domain` contains 4,212 JPEGs of not-yet-propagated eligibility. FilmSet is local and ready for manifest/lockbox freeze; only BlueNeg acquisition remains download-gated. Integration owner: repository owner or explicitly assigned Codex root agent.*
