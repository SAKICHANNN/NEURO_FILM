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
| Physical image formation | P3Q synthetic mechanism retained; P3R photographic route closed | The no-fit annular kernel passes synthetic thickness/index/attenuation gates, but on 12 independent photographs its P95 difference from the equal-moment Gaussian is only `.000594 < .001` and it creates 15,356 frozen isolated excursions. Automatic failure forbids visual review and closes runtime use without retuning. |
| Physical-arm routing value | P8BQ censored / no router | The exact P7F-to-native arm has a fresh winner-only Oracle hint (9/27 versus AO6 7/27), but B0 censors 15/27 pairwise comparisons. No imputation, extra P8BP rounds, selector training or routing opens; AO6 remains global. |
| Complete fixed-bank Oracle | BH0 router closed; BH1 confirms AO6 incumbent | BH1's two exact 24-output runs pass automatic and severe gates. B0 wins all three anonymous rounds 7-5 but reaches 21/36 choices, one below the frozen 22-choice gate. No extra rounds or retuning: retain fixed AO6 t15/c35 and continue a distinct algorithm leaf. |
| Public paired-film refresh | BI0 closed current public surface | SillyStill still exposes no data tree/root licence; Emulating Emulsion exposes no public data repository/link or dataset licence. Two official-source audits are exact; no pixels, contact, fitting or training. |
| Scanner spectral boundary | P6D reference retained; P6E nonlinear RGB route closed | A scanner-A metamer pair gives a `.12618` RGB-only lower bound. Across disjoint synthetic groups the fixed quadratic is 101.66% worse than bounded 3x3 on confirmation mean and fails both tail gates. Keep bounded 3x3 + offline spectral oracle; no capacity rescue or calibration claim. |
| Measured scanner chain | P6J bundle retained; P6K temporal nuisance confirmed; calibrated route data-blocked | P6K finds two repeat-stable TG13/Epson acquisition-response regimes across 14 public FITS wedges, with all six unseen confirmations passing. Treat them only as scanner/wedge nuisance controls. No scanner repeats across manufactured measured targets, so calibrated work remains closed; resume distinct film-image-formation/stock-look algorithms. |
| Controlled film wedge sources | P2K/P2L1/P5E closed | Apollo 7's exact 1.756GB SO-368 wedge archive passes CRC/layout and visibly contains the intended horizontal chart, but the frozen endpoint statistic includes its black bottom film border and fails (`.00635 < .8`); no post-result crop rescue or curve fit. Apollo 16 retains only two endpoint-like levels and a 127px broad-edge stress. |
| Compact explicit colour capacity | AS1 keeps trilinear baseline; AT0 adaptive dose closed | Gaussian loses 3/4 analytic targets to equal-parameter trilinear. AO6 per-image dose normalization cuts residual MAD to `.37284x` but loses all three blind rounds to fixed AO6 while B0 ranks first. Seek a genuinely new controlled direction, not same-population strength retuning. |
| Controlled Velvia display-chain residual | BR0 chroma closed; BR1 chart tone passes; BR2 photo transfer closed | BR1's six-patch monotone residual is real controlled evidence, but on 12 independent photographs its strong effect requires up to 54.41% material gamut attenuation, creates 18.93% new boundary and 14.90% local L-star order inversion. No visual or policy rescue; retain only as display-chain observation. |
| Public paired colour source | AU0 data-blocked | The claimed 41-pair CineStill-800T source is methodologically strong, but its official repository publishes no dataset, reusable image licence or held-group lineage. One example pair is not used. Continue source search or a distinct generic physical mechanism. |
| Independent physical-inspired colour control | AV1 strong style / clipping close | Pinned Filmr is deterministic after its four stochastic grain amplitudes are disabled and reaches style/non-basic `15.94/5.17`, but worst gold clipping is `2.059%` against the `0.5%` veto. No visual rescue or parameter search; seek a different bounded operator/data source. |
| Controlled paired chart source | AW0-AW9 complete / router closed | FilmMatch's exact 140-file source passes integrity and 68-pair mapping. Global and factorized explicit operators fail grouped capacity. A sparse known-high-exposure Oracle is real (12/12 wins, median +16.11%), but raw median-code routing is content-confounded on the fit-forbidden scene; the clean render remains too flat/muted versus E100. Retain only paired code-domain development evidence; next use content-normalized/matched-case explicit operators with global fallback. |
| Optical-density residual colour | AZ0 development pass / AZ1 preference close | Fixed `.15/.35` neutral/opponent density residual repeats all automatic and severe gates on 16 fresh photos, but receives only `5/5/6` candidate preferences; 1/3 rounds passes versus 2 required. Retain research evidence only; no rescue or product promotion. |
| Perceptual residual colour | BA0 automatic close | Fixed D65 CIELAB L*/a*b* factorization is clipping-free and distinct from AZ0, but misses frozen style, real-film-displacement, AO6-difference and AZ0-strength gates. No visual review or strength rescue. |
| Hue/value residual colour | BA1 automatic pass / preference close | Fixed HSV value then hue/saturation factorization passes all automatic and severe gates with zero new clipping, but blind preference versus AZ0 is `5/4/3`; 0/3 rounds pass. No coordinate/strength rescue or product promotion. |
| Public same-subject pair search | BB0 current visible pool closed | Flickr Analog + Digital pages 1-2 expose 136 records and six weak explicit pair candidates, but none is derivative-rights eligible on both sides. Metadata-only repeat audit passes; no pixels, fitting or training. |
| AO6 procedural FilmFX value | BC0 artifact close; BC1 visual-value close | Shared linear optical-density grain removes BC0 coloured speckles, preserves chromaticity to `2.22e-16` and adds zero clipping. Per-sample randomized masked preference is dispersed (`3/2/3` grain; `3/2/4` grain+halation), so neither arm wins a round. Retain research-only; no strength/round rescue or integration. |
| Controlled multi-stock chart source | BD0 structure pass; BD1 stock identification closed | Exact 2.410GB Color Precision Charts archive has 24 same-setup pages, nine named stocks, 12 stock/process variants and paired Frontier/Noritsu views. BD1's two exact exposure-unassigned page-bag audits reach only 20.83% raw and 33.33% basic-normalized symmetric top-1; both miss frozen rank/distance gates. Retain as internal scanner-nuisance stress only; no exposure assignment, fitting, training, LSM or product claim. |
| Registered FilmMatch case Oracle | BE0 evaluator value / bank closed | The fit-forbidden same scene registers at 706/779 inliers. Positive-regime and EV+2 experts improve registered RMSE 11.15%/10.09% over global, but the combined bank fails diversity: zero-regime=EV0 and EV-5/EV-4 collapse. No target-blind retrieval or promotion opens; seek independent paired validation. |
| FilmMatch root-polynomial baseline | BF0 closed | Scene-linear signed RPCC satisfies scale equivariance but loses badly to affine in held-illuminant and held-EV folds and is structurally unsafe. Ordinary cubic has higher capacity but folds/leaves gamut; prior bounded FilmMatch global families already lost final preference. No validation or safety rescue. |
| FilmMatch / explicit reference-look challengers | BL8-BL14 closed; AO6 retained | BL5 is materially non-basic but fails fresh preference. BL13 safely amplifies AO6 chroma but wins only 1/3 rounds. BL14 hard fallback gains 9.50% over global yet transfers one unsafe activation (`1.18075x` worst group) and narrowly fails shuffled control. Close source-conditioned routing on this session; next algorithm must be mechanism-distinct or use independent paired evidence. |
| Fixed global LUT distillation | BM0 context-averaging close | A repeat-exact monotone 1D+17-cube distillation captures 55.97% median AO6 error reduction but retains only 82.77% median style, misses AO6 by 8.52 Delta E76 and regresses the worst camera to 1.430x identity. The safe 3D residual cannot activate. Keep contextual AO6 and test hard case/operator-signature selection, not a larger dense LUT. |
| Hard case/operator Oracle | BM1 evaluator Oracle passes; BM2 ready | Held-camera hard Top-1 case selection improves median error 28.85% over the operator medoid and 19.03% over an oracle scalar-strength path; 9/17 materially win strength and eight case IDs are used with zero new boundary. This proves potential value only. Next test a frozen colour-independent content-space selector; no mode or product claim. |
| Content hard retrieval | BM2 semantic selector closed | Fixed grayscale DINO hard Top-1 matches only 2/17 Oracle cases, improves 4/17 sources, regresses median error 12.98% and reaches an 18.18x worst tail. Keep BM1 as Oracle-only evidence; test one classical tone/layout nearest-case control, then close simple retrieval if it also fails. |
| Case compatibility | BM3 closes fixed nearest-neighbour rules; learned pair ranker ready | Classical grayscale tone/layout retrieval also matches only 2/17 Oracle cases and worsens median error 45.72%. Stop descriptor sweeps. BM1's Oracle value permits one group-crossfit pairwise compatibility model whose only output is a hard explicit case ID. |
| Learned case compatibility | BM4 current router closed | Nested pairwise ridge recovers zero median Oracle gain, improves 1/17 and zero folds, with 2.292x worst tail. Preserve BM1 as evidence that larger grouped case data could matter; do not increase router capacity on 17 sources. |
| Parallel projection-curve operator | BN2 confirmation pass; BN3 visual promotion closed | Corrected source-resolution BN3 is repeat-exact and severe-clean, but adaptive receives `20/36` choices against the frozen `22/36` gate despite winning 2/3 rounds and seven source majorities. Retain BN1/BN2 as explicit-representation evidence only; AO6 stays incumbent and the next algorithm must be mechanism-distinct. |
| Triangular logit transport | BN4-BN6 method evidence retained; BN7 promotion closed | BN7 is repeat-exact, visible and severe-clean, but adaptive loses fixed AO6 in all three rounds (`2/3/4`), reaches only `9/30` choices and three source majorities. No retuning/router rescue; AO6 remains incumbent and the next algorithm must be mechanism-distinct. |
| Historical preference learning | BN8 connectivity pass; BN9 selector closed | 108 sealed source-experiment units are connected, but the fixed interpretable evaluator does not transfer across experiments/cohorts (gains `1.70/.77` points, permutation p=`.614`). Close larger-model and feature rescue; keep AO6 and invest in a distinct renderer mechanism. |
| Emerging controlled bracket source | BG0 current surface closed | 27 Talkative Photographer stock pages yield only two complete five-frame groups (APX400/CineStill50D), below five required; CineStill alone has two scanner variants. Explicit pixel licence is also absent. Metadata only; no image access or contact. |
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
| 1q13 | SF2.6R | Refresh controlled paired-film source availability | complete: public-data unavailable | SillyStill still has one example plus placeholder dataset links/no root licence; Emulating Emulsion has method/figures but no measurements, parameters, code or reusable data licence; no fitting/training |
| 1q14 | SF2.7R | Acquire same-slide multi-scanner nuisance control | complete: source pass | 11 assets/71,068,957 bytes; 4 pipelines × 5 slides complete; CRC/decode clean and zero cross-pipeline duplicates; opens SF2.7A only |
| 1q15 | SF2.7A | Quantify scanner/software nuisance on same physical slides | complete: partial control pass | raw held-out median .06055; bounded 3x3+bias .01446/76.11% lower; .01464-.09865 pair spread rejects universal scanner modes; no stock fit/training/LSM |
| 1q16 | SF2.8R | Audit NTNU controlled Ektachrome/Velvia hyperspectral source | complete: public-data unavailable | strong two-stock/two-scene/two-light/multi-exposure design, but exact public files contain no raw pixels/measurements/code/licence and analysed frames are cross-stock illuminant-confounded; no fitting/training/LSM |
| 1q17 | SF2.9R | Audit direct multi-family IT8 transmission measurements | complete: physical/nuisance evidence only | 5 archives/1,468,016 bytes, 288 common patches and 41-point spectra pass; calibrated-target manufacture and absent common recorder input close stock operator fitting/training/LSM |
| 1q17b | SF2.9B | Test Ektachrome-family vs Velvia-50-family stability across target charges | complete: held-charge measurement pass only | 100 archives/29,856,706 bytes; frozen 13-charge confirmation gives spectral/Lab BA=AUC=1.0, exact p=.000583 vs year-only .536 and batch-error .357; no stock operator or appearance claim |
| 1q18 | U6.P6L | Audit same-plate three-scan residuals as scanner nuisance | complete: source correlation close | exact 88.48MB and decode pass, but repeat 1/3 correlation `.99310 < .995`; no normalization, crop or gate rescue and no residual analysis |
| 1q18 | SF2.10R | Audit Color Precision comparison metadata, scanner topology and pixel rights | complete: promising topology / pixel DoR blocked | exact repeat; 927 URLs, 20 weak labels, 456 apparent scanner pairs, but rights, verified pairing and independent replication fail; zero image requests |
| 1q12 | RF2.C0 | Isolated spektrafilm external spectral control | complete: one external control only | Ektar100/fixed-e0 passes style/non-basic/clipping and 3-round visual gates; auto exposure is unstable nuisance evidence; no integration, teacher, stock or calibration claim |
| 1r | LSM0 | Freeze within-stock latent-mode ontology and epistemic contract | complete after propagation | hypothesis only; observed/latent ledgers separate; no stock has proved multiple modes |
| 1s | U5.R2AZ1 | Confirm the fixed AZ0 optical-density operator on an independent image population | complete: preference closed | automatic/severe gates pass; blind `5/5/6`, only 1/3 rounds; no retuning/product claim |
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
| 1be | U2.2B | Non-duplicative sensitometry-to-print composition | complete: numerical pass | one sensitometry stage, exact endpoints/range/parity, min Jacobian .0470, affine residual .1015; U2.3 opens, no stock/integration claim |
| 1bf | U2.3A | Compose U2.2B with bounded tetrahedral residual | complete: identity pass / witness closed | identity composition and scalar parity exact; cyclic residual fails frozen first/second smoothness gates; generic fail-closed wrapper retained; evaluate U2.2B base visually only under a new contract |
| 1bg | U5.R2I0 | Evaluate fixed U2.2B on frozen real-image frontier | complete: metrics pass / visual value closed | all strengths pass, s1.0 style/non-basic 19.28/14.40; 18/18 severe-clean but green cast loses blind 0-3 to safe-rich; no challenger; neutral-axis gauge hypothesis next |
| 1bh | U5.R2I1 | Canonicalize U2.2B through its own neutral response | complete: numerical pass | neutral spread .1630 -> 6.41e-6 with positive Jacobian and non-affine residual retained; freeze real-image frontier next |
| 1bi | U5.R2I1B | Test neutral-gauged path on frozen real images | complete: no survivor, route closed | s1.0 only style/non-basic 5.66/.72; all strengths fail both floors; no visual candidates or rescue |
| 1bj | U5.R2J0 | Audit compact positive-film response representation | complete: representation pass | five synthetic witnesses pass all regularity/diversity/replay gates; min affine residual .09596; 818 tests; opens J1 only |
| 1bk | U5.R2J1 | Run fixed positive-film bank on frozen real images | complete: no visual gain | 7 automatic survivors and 27/27 shortlist renders severe-clean, but R2E1 wins all three blind rounds and safe-rich is runner-up; J1 bank closes without retune |
| 1bl | U5.R2K0 | Audit bounded fixed-Gaussian residual colour representation | complete: fidelity pass / regularity fail | N27 gains 85.98%/87.65% over affine at .00990/.00344 RMSE, but folds colour space (min det -3.31/-1.87), negative diagonal derivatives and coefficient 7.09 fail; no rescue |
| 1bm | U5.R2K1 | Audit structurally invertible triangular colour coupling | complete: structural pass / fidelity fail | min det .168/.359 and inverse <5.6e-16 pass, but RMSE .0623/.0282 and gains 11.17%/-2.34% fail; exact partition differs 1.11e-16; no capacity chase |
| 1bn | U5.R2K2 | Audit finite rational-quadratic spline colour coupling | complete: structural pass / confirmation and norm fail | density RMSE .01435 passes but norm 9.457 fails; positive fit .00067 generalizes to .01965 above .015; exact inverse/partition and positive orientation pass; no retune/render |
| 1bo | U5.R2K3 | Audit analytic gamut-polar film-palette factorization | complete: representation/diversity pass, norm fail | exact gamut/neutral/inverse and strong non-affine diversity pass; cyan-shadow/warm-highlight norm 11.198 > 8 closes full bank before real images; no retune |
| 1bp | U5.R2L0 | Measure a post-E1 per-image hard-strength Oracle | complete: material gap / severe pass | s0.65 selected 36/41; mean/median style gain 3.07/3.56, worst selected clipping .4925%; 41/41 severe-clean including ID11/face; opens only simplest inference-time hard-policy contract |
| 1bq | U5.R2L1 | Replay the Oracle with a full-frame inference-time hard preflight | complete: exact policy gate failed | both strengths replay RGB8-exactly 41/41, but stress20 shifts .49247% sampled to .50015% full-frame; assignment/output 40/41 closes without threshold/epsilon/sampling rescue |
| 1br | U5.R2M0 | Audit a bounded interval-Möbius explicit coupling flow | complete: structural/positive pass, density+identity fail | positive RMSE .01170 passes; density .03089 misses .015 and fitted identity max error 5.97e-5 misses exactness; max norm 2.842, inverse/partition clean; no render/rescue |
| 1bs | U6.2A | Audit clean-room Boolean/Poisson physical-inspired grain representation | complete: synthetic representation pass | mean error .0133; variance ratios 2.15/2.25; radius-correlation delta .1416; exact repeat/partition/replay; opens fixed crop visual frontier only |
| 1bt | U6.2B | Test fixed Boolean-grain policies on difficult existing-image crops | complete: automatic pass / visual severe close | small/large pass aggregate gates but fail severe 4/5 and 5/5 with salt-like bright points; legacy `.018` 0/5; no rescue/integration |
| 1bu | U5.R2N0 | Audit ModFlows palette embedding/invertible colour-flow source | complete: B0 external control feasible | 18.97MB MIT-labelled weight pinned; code unlicensed/training lineage incomplete; clean-room synthetic audit only, no stock pixels/fitting |
| 1bv | U5.R2N1 | Audit pinned ModFlows B0 checkpoint and palette/content separation | complete: structure pass / shortcut+range fail | identity/det/norm pass; permutation cosine .7255, palette/geometry .803 and raw range [-.073,1.119] fail; no images/clamp/fine-tune/B6 |
| 1bw | U5.R2O0 | Audit a cube-preserving diffeomorphic explicit colour flow | complete: full synthetic representation pass | both nonlinear controls pass fidelity/gain/range/Jacobian/norm/inverse/repeat; compact bounded-parameter representation retained, but no redundant image frontier, fitting or stock claim |
| 1bx | U5.R2P0 | Audit ChameleonTuner region-paired LUT optimization | complete: future paired prior only | requires same-scene pairs; official repo has README only/no licence; no unrelated-photo supervision, current film fitting or implementation |
| 1by | U5.R2Q0 | Audit SA-LUT reference-conditioned spatial 4D LUT | complete: clean-room architecture prior only | explicit 4D-LUT inference boundary passes conceptually, but official training uses a direct pseudo-Log image generator, GAN loss and incomplete asset lineage; no checkpoint/PST50 run, stock evidence or implementation |
| 1bz | U5.R2R0 | Audit D-LUT score/Langevin reference-conditioned explicit LUT | complete: fixed official-asset audit feasible | not a direct RGB diffusion generator, but one-image scene-palette density is the only supervision; paper/demo epsilon contradiction retained; no stock or unpaired-operator claim |
| 1ca | U5.R2R1 | Characterize all 41 published D-LUT trajectory steps | complete: 0/40 nonidentity survivors | identity/repeat and norm pass; range 0/40 and orientation 0/40; step40 is strong/non-affine but about 48% locally reversed; no images or rescue |
| 1cb | U5.R2S0 | Combine analytic reference-palette score with cube-preserving flow | complete: all analytic mechanism gates pass | three palettes are strong/non-affine/distinct and preserve range/orientation/norm/inverse/repeat; synthetic histogram-score recovery contract may open, no image/film claim |
| 1cc | U5.R2S1D | Test canonical-histogram hard case retrieval against KDE/blend/global controls | complete: query KDE .12 selected | repeated dev: KDE median error .0327 vs hard Top-1 .0873/global .1128; raw-hist hard retrieval loses this question, seed 27012 untouched |
| 1cd | U5.R2S1C | Confirm fixed query KDE .12 on untouched synthetic palettes | complete: all gates pass | median/p90 .0305/.0503; 63.5%/71.9% better than hard/global; exact permutation/repeat and safe structure; next must isolate content nuisance before images |
| 1ce | U5.R2S2D | Separate shared style flow from independently varying content palettes | complete: no residual winner | raw KDE wins error but fails identity nuisance; ridge worse/unstable, density ratio near global; no confirm/capacity rescue, seed 28105 untouched |
| 1cf | U5.R2S3D | Fit explicit flow from independent neutral/styled distributions | complete: repeated `distribution_fail` | RFF/SW held-out gain 11.80%/3.36%; oracle and A/B stability fail while every structure gate passes; objective non-identification, no rescue/visual |
| 1cg | U5.R2S4D | Test diversified conditional distribution matching with one shared explicit flow | complete: conditional match/operator fail | correct conditions reach 71.36% distribution improvement but `.10751/.11900` oracle RMSE and only 8.45%/24.98% improvement over pooled/shuffled; exact repeat; no rescue or visuals |
| 1ch | U5.R2T0 | Audit StatLUT statistical-to-LUT learning and source lineage | complete: retain prior / close execution | image branch is explicit but learns known random-LUT same-image supervision; Lab statistics remain content-sensitive, 4,000 professional LUTs lack rights lineage, code/checkpoints absent, diffusion text branch excluded |
| 1ci | U5.R2U0 | Audit ColorFM semantic HCC and flow-parameter prediction | complete: clean-room synthetic control only | HCC constructs plausible pseudo-pairs, not identified transport; official repository has no code/checkpoint/licence and paper MLP/Euler path lacks project bounds |
| 1cj | U5.R2U1D | Compare clean-room HCC pseudo-pairs using the bounded O0 flow | complete: pair fit/operator fail | exact repeat; correct HCC fits pseudo-pairs 84.72% but `.08791/.09699` oracle and `.04176` A/B fail; it loses to random and pooled controls; no rescue/visual |
| 1ck | U5.R2V0 | Audit cmKAN unpaired colour matching and source boundary | complete: exclude published path | unpaired training is CycleGAN; a content encoder predicts spatially varying per-pixel KAN parameters and directly returns RGB, without cube/Jacobian/spatial guarantees; retain bounded global-spline prior only |
| 1cl | U5.R2W0 | Audit submitted NFRM/CFSM reference-look direction | complete: retain CFSM question, reject final-algorithm/film claim | same-look/different-content supervision is useful; single final references remain content/capture/nuisance confounded; keep operator and content spaces separate and reuse O0 |
| 1cm | U5.R2W1D | Test fixed reference-look retrieval/regression across generated content | complete: paired upper bound only | exact repeat; paired `.02498/.07282` passes, but output-only methods lose identity/global, retain 98.44% content signal and fail identity-reference control; no rescue/visual |
| 1cn | U5.R2W2F0 | Test paired/global explainability of local FilmSet recipes | complete: no coherent global recipe champion | exact repeat over 40 IDs/160 payloads: Cinema is basic-only; ClassNeg/Velvia are adaptive-or-spatial. All structure gates pass, but grid dispersion fails all three; no W2F1, local rescue, final-628 or film/stock claim |
| 1cv | U5.R2BK21 | Test canonicalizer-consensus hard appearance matching | complete: content-confound control fails | material appearance match and weak-reference fallback work, but two objectives jointly authorize a shifted-content reference; consensus is not a sufficient content-safety gate |
| 1cw | U5.R2BK22 | Test colour-independent content-cell worst-case gate | complete: synthetic mechanism pass | identical pooled target sets are separated only by external cell correspondence; valid authorizes, shuffled hard-falls back; real connectivity remains unproved |
| 1cw1 | U5.R2BK23 | Test the frozen content-cell gate on real FiveK Expert C rasters | complete: K=1/identity | pooled appearance improves 92.89%, but only 64.06% of observed photo cells pass and worst improvement is -669.18%; exact repeat, structure passes, no shared operator or film claim |
| 1cw2 | U5.R2BK24 | Test colour-blind semantic pseudo-pairs as unpaired supervision for one explicit operator | complete: K=1/identity | DINOv2 retrieval beats shuffled control but regresses 15/16 held-out photos; exact repeat, no same-identity leakage, structure safe; close without rescue |
| 1cr | U5.R2Z0 | Test a fixed FilmSet case bank and input-only within-recipe hard retrieval | complete: ClassNeg Oracle only / Velvia no Oracle | exact repeat; ClassNeg Oracle gains 16.72% and wins 75%, but spatial photometric NN is worse than shared with `-208.62%` gap closure. Velvia misses Oracle gates; no selector/router/visual/final-628/film claim |
| 1cs | U5.R2Z1 | Test asymmetric query/operator applicability for the ClassNeg Oracle gap | complete: off-diagonal bank/router route closed | exact repeat; excluding self-operator leaves 7.32% Oracle gain/62.5% wins and non-positive CI; rank2/rank4 are 28.54%/34.70% worse than shared. Fresh targets stay sealed; no capacity/visual/final-628/film rescue |
| 1ct | U5.R2AA0 | Audit official Kodak 250D-to-2383 datasheet-chain inputs | complete: bounded nuisance pilot feasible / operator unidentified | both stocks publish characteristic, sensitivity and separated CMY graphs; 2383 LAD supplies neutral aims. Scene spectra, absolute dye mapping, printer SPDs/timing and xenon/view response remain missing; AA1 synthetic audit only |
| 1cu | U5.R2AA1 | Test Kodak datasheet-chain nuisance identifiability on a synthetic RGB grid | complete: strong/non-basic but nuisance-unidentified | style/basic/affine pass; spectrum, placement, dye-map, ensemble and neutral gates fail. Exact repeat; no AA2, visual, fitting, clamp/LUT/neural rescue or physical Kodak claim |
| 1cv | U5.R2AB0 | Audit CAVE DoRF named-film response curves and rights | complete: limited internal-research source pass | exact 201 records/1,024 samples/46 strict RGB triplets; page/runtime count contradiction and source duplicates preserved; licence unknown, so AB1 synthetic-only and no images/fitting/training/product |
| 1cw | U5.R2AB1 | Test DoRF strict RGB response diversity and safety | complete: insufficient diversity / bank closed | exact repeat; 27/46 individual survivors but closest pair `.001016 < .02`; strong response curves retained only as descriptive synthetic prior, no AB2/images/bank/router/rescue |
| 1cx | U5.R2AC0 | Audit Filmulator shared-developer spatial mechanism | complete: mechanism prior only / direct reuse blocked | technically distinct local tone mechanism, but GPLv3+ source and undecided project licence forbid copy/port/link; synthetic first-principles contract only |
| 1cy | U5.R2AC1 | Test independent bounded shared-resource diffusion representation | complete: effect too weak / route closed | exact repeat and all structure gates pass; local-vs-global `.0019468 < .002`; no retune/AC2/photos/integration |
| 1cz | U5.R2AD0 | Audit spektrafilm spatial DIR coupling and RF2.C0 overlap | complete: isolated external ablation feasible | RF2.C0 forced inhibitor diffusion to zero but retained non-spatial DIR; GPL/CC-BY-SA source may be executed only as an ignored external comparator |
| 1da | U5.R2AD1 | Test spatial DIR on/off with the retained Ektar100 fixed-e0 control | complete: replay + red-speckle failure / closed | nine off/on PNGs repeat, but all on-float hashes differ microscopically and worst new isolated red speckles `.1481% > .1%`; no vision, parameter rescue or AD2 |
| 1db | U5.R2AE0 | Audit spectral_film_lut source, profile lineage and headless operator path | complete: synthetic audit feasible / profile truth blocked | MIT code executes headlessly, but 86 unique profiles lack per-row source/page/uncertainty; 31 historical PDFs were removed and cannot be treated as redistributable truth |
| 1dc | U5.R2AE1 | Source-bound synthetic operator-bank structural diversity audit | complete: strong/non-basic but folding failure / closed | style 19.07–23.96 and all families distinct, but every chain has 1.61–9.50% negative-Jacobian cells vs `.5%`; no visual/AE2 or rescue |
| 1dd | U5.R2AF0 | Audit AceTone source, supervision, rights and topology contract | complete: generative selector excluded / tokenizer stress only | exact Apache checkpoint may enter one analytic safe-LUT stress; Qwen/GRPO, photos, benchmark, training and film claims remain closed |
| 1de | U5.R2AF1 | Exact-checkpoint safe-LUT topology stress | complete: topology + fidelity failure / closed | 8/8 outputs fold at 7.54–10.60%; zero survivors despite exact controls/replay; no decoder or projection rescue |
| 1df | U5.R2AG0 | Audit bounded convex-gradient palette map | complete: compact clean-room pilot feasible | 65-scalar in-cube SPD-Jacobian construction is distinct from O0; remains canonical palette mapping, not film operator identification |
| 1dg | U5.R2AG1 | Test fixed 16-anchor convex-gradient representation | complete: safe structure / insufficient capacity / closed | exact repeat and structural gates pass; density RMSE .04047 fails .015, positive control passes .00945; no capacity or tolerance rescue |
| 1dh | U5.R2AH0 | Audit learned group-invariant reference-only operator prediction | complete: development pilot feasible | new crossed synthetic groups, hierarchical Deep Sets and explicit content controls may predict only bounded O0 parameters; W1 confirmation and all real pixels stay sealed |
| 1di | U5.R2AH1D | Freeze and test one group-invariant O0-parameter predictor | complete: repeat-exact recovery/shortcut/strength failure | `.04001/.09723` median/p90; only 7.25%/4.89% vs identity/global, content BA 27.10%, fixed-content look accuracy 12.5%, strength Spearman -1. Structure passes; no AH1C or rescue |
| 1dj | U5.R2AI0 | Test fixed composition of the anchor56 margin-4 and density-cyan B0 champions | complete: stronger B0 composition retained | density-then-anchor s.50 reaches style/non-basic 15.0576/11.3299, zero new clip, exact repeats, 3/3 blind parent wins and 0/9 severe; no production/stock claim |
| 1dk | U5.R2AI1S | Preflight an independent digital-photo/OOD source population | complete: 17-row source pass | 18/18 exact CC0 RAWs acquired and repeat-identical; one visible Kodak target excluded by frozen rule with no replacement; 17 eligible/nine makes/11 buckets, zero overlap or severe source failures |
| 1dl | U5.R2AI1 | Confirm the frozen composition on independent digital-photo/OOD inputs | complete: automatic style-advantage failure / closed | exact repeat and zero clipping; candidate style/non-basic 13.1756/12.6753, but style gain -.5261, only 7/17 style wins and max 32.2442 exceeds 31.4746; no blind/full-res review or rescue |
| 1dm | U5.R2AJ0A | Audit official RawTherapee HaldCLUT source, rights and semantics | complete: licensed acquisition feasible / no stock truth | exact 421,602,289-byte CC BY-SA 4.0 archive and 194 non-Creative colour entries frozen by range-only metadata/README audit; no LUT decoded |
| 1dn | U5.R2AJ0B | Acquire and integrity-audit the exact HaldCLUT archive | complete: exact archive / frozen mode-contract failure | byte/MD5 pass; first audit rejects B&W auxiliary mode `L` under unchanged RGB/RGBA-only v1; no repeat evidence or structural-ready state |
| 1do | U5.R2AJ0B2 | Confirm exact lane-aware Hald decode profiles | complete: two-process integrity pass | commit 9bd4e68; all SHA/profile/path/image-record and strict canonical repeat gates pass; no conversion, photograph render or aesthetic inspection |
| 1dp | U5.R2AJ0C0 | Review first Hald structural controls contract before execution | complete: pre-execution contract close / zero access | invalid probe/native geometry and incomplete evidence/negative/novelty/selection rules close v1 before code or primary metrics |
| 1dq | U5.R2AJ0C0B | Run corrected controls-only Hald evaluator | complete: exact two-process conformance pass / zero access | commit d8fdcb1; legal N=4/36/144/256 controls, independent reconstruction and all gates pass; archive, primary and photograph access are zero |
| 1dr | U5.R2AJ0C1 | Run the frozen per-Hald structural frontier | complete: 194/194 structural veto / route closed | both child byte streams and parent reconstruction are exact; all 194 pass style/non-basic but all 194 fail minimum Jacobian determinant, zero survivor, zero photograph render; no rescue |
| 1ds | U5.R2AK0 | Audit NCT learnable Bezier flows as a distinct ML explicit-operator route | complete: trajectory prior only / direct execution closed | explicit ODE parameters fit the renderer boundary, but per-image distribution endpoints remain content-confounded; official supplement has no code/checkpoint/licence and no cube/Jacobian/severe gates |
| 1du | U5.R2AK1 | Test compact time-dependent cube flow on noncommuting staged colour reactions | complete: safe absolute fit / relative capacity failure / closed | exact repeat; `.00235/.00259` RMSE and all structural gates pass, but candidate is 73.94%/87.72% worse than K4 and 2.56x/2.86x K5; no capacity rescue or photographs |
| 1dv | U5.R2AL0 | Audit clean-room analytic named-colour monotone curves | complete: clean-room analytic prior only | external learned pipeline/code/checkpoint/colour-naming asset/data closed; retain continuous analytic chroma-sector partition plus constrained curves for synthetic testing |
| 1dw | U5.R2AL1 | Test analytic chroma-sector curve capacity | complete: relative capacity pass / absolute+inverse failure / closed | exact repeat; `.006254/.006222` beats global by 74.41%/74.89% and K3 narrowly, but exceeds `.006` and inverse fails both orders; all other structural gates pass, no rescue/photos |
| 1dx | U5.R2AM0 | Audit invertible image-adaptive coordinate curves | complete: stricter clean-room prior only | IAC random rank repair and missing monotone/condition/whole-inverse/cube/Jacobian gates close direct use; retain deterministic SO(3)+full-cube normalization+analytic inverse curves |
| 1dy | U5.R2AM1 | Test bounded SO(3)-coordinate curve capacity | complete: inverse/Jacobian pass; capacity/cube/neutral failure / closed | exact repeat; both targets miss `.015`, leave cube and violate neutral spread. Density loses positive-matrix/K3 controls; positive is relatively competitive but `.02124`; no rescue/photos |
| 1dz | U5.R2AN0 | Recover the existing positive-film operator from paired observations | complete: exact held-group recovery / advance | exact repeat; candidate `~1e-16` versus one-matrix `.00982`--`.01252`, all structure gates pass; next test noise/outlier robustness |
| 1ea | U5.R2AN1 | Test robust recovery with noisy/corrupted chart pairs | complete: absolute pass, one relative failure / fixed soft-L1 closed | exact repeat; one `61.44% < 65%` outlier-gain failure despite `.000437` RMSE; test separate two-stage rejection |
| 1eb | U5.R2AO0 | Acquire real Velvia 50/reference chart display-proxy pairs | complete: exact bounded source pass | two 291,706-byte downloads and 24-pair extractions exact; opens AO1 only |
| 1ec | U5.R2AO1 | Explain real Velvia chart display proxy with explicit operators | complete: one-matrix champion | exact repeat; `.04404` mean held-row RMSE, 72.51%/46.62% gains over identity/full affine; two-matrix slightly worse |
| 1ed | U5.R2AO2 | Test full-chart one-matrix Look Approximation on real photos | complete: single-strength transfer closed | s0.50 lacks non-basic style; s0.75 is strong but 1.336% gold clipping; no visual candidate |
| 1ee | U5.R2AO3/AO3V | Test factorized tone/chroma boundary guard | complete: automatic pass, aesthetic champion closed | `.50/1.00` is zero-clipping and severe-clean, but loses 0/3 blind rounds to retained B0 comparator |
| 1ef | U5.R2AO4S/AO4P | Find and extract independent paired positive-film evidence | complete: Figure 11 proxy passes | 47 Velvia/HSI + 56 Ektachrome/HSI uniform palette pairs; Figure 6 retained only as illuminant/exposure nuisance |
| 1eg | U5.R2AO4C | Test Velvia chart↔pigment cross-domain transfer | complete: pass | one-matrix beats identity/full-affine by 62.16%/40.53% and 63.79%/47.20%; stock specificity remains content-confounded |
| 1eh | U5.R2AO5/AO5F/AO5V | Fit combined 71-pair Velvia operator and challenge photo frontier | complete: direction improves, champion closed | exact `.02941` combined fit; severe-clean survivor beats chart proxy 3/3 but loses vivid B0 0/3 |
| 1ei | U5.R2AO6/AO6V | Compose bounded combined-film residual with vivid B0 density path | complete: new B0 development champion | exact zero-clipping `t15/c35` reaches `13.24/9.87/2.05`; beats B0 2/3 and stronger residual 3/3; 0/18 severe |
| 1ej | U5.R2AO7S/AO7 | Confirm the fixed AO6 champion on a genuinely fresh raw.pixls.us population | complete: automatic pass, preference promotion closed | 16 rows/nine makes after one frozen 404 and one chart exclusion; exact repeats, zero clipping and material residual, but candidate wins only 1/3 and identical layouts disagree |
| 1ek | U5.R2AO8-AO9 | Test proxy-operator diversity, mature case routing and published-equation capacity | complete: diversity real, routing/capacity closed | directions are not a strength path, but kNN is split-unstable; clean-room 30p baseline regresses RGB/DeltaE and produces 7.04% raw OOG. Retain bounded global AO6 |
| 1el | U5.R2AP0-AP2 | Fit the independent Ektachrome palette proxy and test bounded photo execution | complete: direction/guard retained, direct route closed | proxy direction is distinct and explainable; direct transfer clips or loses blind comparisons, while the analytical residual-direction guard remains reusable |
| 1em | U5.R2AP3-AP4 | Compose and independently confirm the fixed Ektachrome residual with B0 | complete: automatic/severe pass, fresh preference closed | t10/c25 is material and 0/16 severe on fresh photos, but fixed B0 wins 3/3 frozen blind rounds; no retuning/capacity rescue |
| 1dt | GOV.GCP1 | Freeze autonomous GCP ownership/cost/cleanup governance | complete: v2 USD 2,500 hard cap / zero cloud mutation | config/prefix `nf-019f4b76-agent` / `nf-019f4b76-`; USD 2,250 new-start stop, ledger-only ownership, paid idle forbidden |
| 1co | U5.R2W2R | Evaluate INRetouch RTD real-raster same-preset bridge | source preflight complete; pixel leaf not ready | frozen bounded-acquisition/global-explainability DoR; 167 preset recipes x 569 FiveK contents, but gated 18 GB CC BY-NC-SA source needs human click-through/contact disclosure, lineage reconciliation and W1 evidence; no data accessed |
| 1cp | U5.R2X0 | Collapse the 2025 automatic palette-transfer equations analytically | complete: affine-only close | exact repeat; direct path equals affine to `9.99e-16`, while valid transports admit determinant `-1` and `[-2,3]` cube output; no image/CPLEX/local-path or film claim |
| 1cq | U5.R2Y0 | Audit NegClone scan-averaged stock fingerprints for content shortcuts | complete: content/histogram/texture shortcut close | exact repeat; film-free colour bias L2 `.8485`, tone difference `254.4/255`, checker grain `.4` versus flat zero; no more-photo/neural/preset rescue or stock claim |
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

Current physical-runtime leaf: U6.P8BP is closed after a fresh nine-camera
CC0 RAW confirmation. Full native Standard is repeat-exact, boundary-clean and
has zero confirmed new severe artifacts, but wins only one of three blind
rounds against AO6 colour-only and 5/12 pairwise choices. The frozen
preference gate fails, so AO6 remains the simpler colour champion and no
router/default/product promotion opens.

Current physical-algorithm leaf: U6.P3O confirms P3N numerically and closes its
product value. Two exact runs over 17 disjoint CC0 RAW photos/nine makes retain
the `.009` response bound, zero isolated excursions/new boundaries and zero
confirmed severe artifacts. At the frozen direct diagnostic scale, all nine
rows are ties in all three blind rounds, so 0/3 rounds pass the 2-round value
gate. Retain the explicit numerical primitive only; parameter strengthening,
same-population rescue and runtime/product integration are closed.

Independent grain confirmation U6.P4Z-P4Z2 is also closed. Three new exact
CC0 16-bit B&W uniform scans pass source integrity, but the frozen P4T model
improves radial NPS while worsening short-lag ACF, and P4X's colour-derived
amplitude scale does not transfer. No T-MAX/Tri-X profile, retune or product
reopen is allowed.

U6.4A now passes one separate generic creative-diffusion representation:
positive six-scale linear-light scatter preserves constants, neutral channels,
energy and component bounds in two exact synthetic audits. It is explicitly
not film halation or a measured filter profile. U6.4B then closes the fixed
strong photographic route automatically: flat-region and isolated-excursion
gates fail, so visual review, retuning and integration remain forbidden.

Current real-grain leaf: U6.P4R closes stock association while retaining one
generic scanner-convolved signature candidate. Eight exact CC0 16-bit RGBA
TIFFs pass acquisition and visual gates; RGB NPS is highly repeatable, but
within-minus-cross similarity is only .00364 against the frozen .03 effect
gate. U6.P4S's label-blind shared anisotropic Poisson candidate (`.90/.65px`)
then improves held median NPS distance by 85.45% over P4D and passes exact
repeat/partition gates. U6.P4T also improves held ACF median error by 17.66%,
preserves density/transmittance semantics and has zero confirmed severe
failures on six fixed synthetic diagnostics. U6.P4U then closes the direct
photographic integration: two exact 16-image runs reproduce the no-material
P7F control and add zero boundary, but P4T's worst median luma drift is
`.05322 > .03`; visual review and retuning are forbidden. U6.P4V remains
formally unidentified: drift/local-residual correlation is `.850`, but the
frozen large-residual gate misses (`.04645 < .05`). P4W then retains one
repeat-stable same-scanner amplitude target and P4X selects a fixed global
`.125` proxy scale. P4Y closes the resulting material candidate: physical,
repeat and partition gates plus worst held ACF improve, but held median ACF
improvement is only `4.85% < 10%`. Visual review, same-scan retuning and
photographic integration remain forbidden. The closed P8BP challenger,
stock-specific fitting and calibration remain forbidden.

Current explicit-algorithm leaf: U6.P2E is closed automatically. Its two
reference-gauge runs are exact, but neutral scan compresses three independent
photo luma ranges to 8.3%-12.4% of source range and violates the frozen
per-image range gate. Visual rescue and negative-route retuning are forbidden.
U6.P2F passes only as a separately typed generic reversal-development
primitive. U6.P2G then repeats exactly on 18 CC0 photographs and passes every
automatic gate plus fixed visual severe review with 0/9 confirmed structural
failures. The outputs remain uniformly dense and cyan-green, so style and
preference are unresolved and no product/stock promotion opens. U6.P2H is
ready to attribute that appearance across direct transmittance and scanner
stages without fitting.

U6.P2H now localizes the problem: most luma/range compression already exists
in raw transmittance, while spectral is only the largest small incremental
scanner shift. U6.P2I is ready to define an explicit endpoint-derived
scan-signal black/white normalization; it must use synthetic flat endpoints,
never photograph fitting or silent clipping.

U6.P2I now passes that synthetic primitive exactly: endpoint separation is
`.575-.698`, 0/1 mapping is exact, inverse error is `1.11e-16`, and outside
signals fail closed. U6.P2J is ready to challenge the unchanged 18-image
population with reversal + scanner + normalization and the same severe-first
discipline.

U6.P2J is now closed automatically. Endpoint normalization restores range and
creates zero endpoint escape or new boundaries, but two independent photos
exceed the frozen 10% near-black gate at 11.76%/11.60%. Visual rescue,
endpoint shrinkage and toe retuning are forbidden. AO4S1 then finds a strong
independent FILM2PAINT controlled-target design, but no public patch table,
raw scan, manifest or reusable data licence; figure extraction and fitting are
closed. U5.R2AO4M then passes all structure gates and transfers between whole
proxy domains, but misses the frozen held-row RGB gain, regresses the chart
domain and worsens mean Delta E76 by 40.60%; it closes without photo or metric
rescue. U5.R2AN2 then locates 26/27 corrupt rows for all three witnesses, but
hard deletion plus linear refitting still loses on the critical cyan-shadow
witness and on two clean-noise rows. U5.R2AN3 Cauchy then loses to soft-L1 on
all three contaminated witnesses and still misses the critical linear-gain
gate, closing this robust rescue family without a sweep. U5.R2AQ0 passes the
bounded source/semantics gate: two exact 3,147,395-byte audits cover five
film-recorder RGB TIFFs and six target sets x five measured Velvia 100F
slides, but all sets share production date 2005:05 and are not independent
rolls/processes. U5.R2AQ1 passes all frozen gates on the exact 8,640-row table:
median/p95/max repeated-set radius `.553/1.280/2.855` Delta E76 and held-set
consensus RMSE `.890`. U5.R2AQ2 is ready to freeze complete target-set holdout
baselines; the claim remains recorder-to-measured-slide proxy only. AQ2 then
closes on joint held-set/held-grid evidence: quadratic reaches `13.72/17.71`
mean/worst-fold Delta E76 but produces 3.29% negative XYZ, while the bounded
nonnegative affine remains `22.79/30.07`. No larger model or clipping rescue.
AQ2D attributes this to source support: four of five held recorder grids exceed
the frozen local-distance or convex-hull gate; slide 4 is 38.54% outside the
other grids' hull. Reopen only with additional controlled grids or a distinct
physical evidence source. U5.R2AR0 then tests a distinct safe explicit family
on the reused 71-pair Velvia proxy: the K3 cube flow is structurally healthy
and improves chart-to-palette by 34.23%, but regresses palette-to-chart by
26.64% and one combined fit by 42.66% versus one-matrix. It follows proxy
support rather than yielding a stronger shared operator, so no capacity,
domain-router or photo rescue opens; AO6 t15/c35 remains the colour
development champion. U5.R2AR1 then gives both proxy domains equal total
weight in one shared bounded operator. Worst-domain RMSE improves 9.20% on
the full rows and 4.17% in fixed five-fold cross-fit for only 2.33%/3.72%
pooled regression, but minimum Jacobian determinant is `.001516 < .01`.
The frozen structural gate closes the family without strength, weighting,
capacity or photograph rescue; AO6 remains the colour development champion.

Current algorithm state: BJ0-BJ2 adaptive neutral-base LUTs are closed. U6.P4AC
closes the no-fit 2.0D-peaked family; P4AD-P4AE instead pass an exposure-Poisson
variance mechanism propagated through characteristic-curve slope and unchanged
U2.2 sensitometry. P4AF is synthetic-severe-clean, but P4AG-P4AJ close its first
LOD family: flat-field covariance moments pass, scalar exposure fails by up to
`842.85x`, and exact subpixel density moments still reach `1.551x` versus the
frozen `1.3x` gate. Retain P4AF as a full-resolution synthetic reference only;
the next grain leaf needs independent evidence or a genuinely different model.

U6.P4AK/P4AL close two independent source routes under their frozen non-OCR
contracts. U6.P4AM then proves an exact Gamma-Laplace mean-transmittance
correction mathematically, but it reduces the fixed correlated-field bias by
only 1.20% and misses the empirical flat gate. It therefore closes before
photographs. Do not rescue the same marginal with sample centering; the next
physical leaf must change the stochastic structure or move to another
high-information mechanism.

U6.P4AN/P4AO then close balanced and zero-DC copula source changes; the latter
reduces low-frequency drift but improves the frozen flat error only 14.68%.
U6.P4AP tests the current standards-derived causal AR family: all lag 1/2/3
development fits exceed the frozen coefficient envelope, so the family closes
with zero confirmation reads and no parameter rescue. The next grain leaf must
use a stability-by-construction source family or move to another physical stage.

U6.P4AR executes that distinct nonnegative spectral-mixture source. Its shared
quantized bundle does not collapse, but development NPS/ACF regress 95.80% and
17.34% versus P4T; it closes with zero confirmation reads. Stop scanner-code
spatial-source fitting on this reused cohort and advance another physical stage.

BJ0 closed on the unchanged safety-limited tail gate despite strong mean,
win-rate and P95 error improvements. Next: freeze an intrinsically
cube-preserving residual basis; do not retune or confirm BJ0.

BJ1 closed: cube membership is exact, but strict near-boundary safety and AY3
P95 fail. Next legal leaf is one preregistered strict-epsilon-interior LUT
basis, not a BJ1 parameter rescue.

BJ2 closes the adaptive neutral-base LUT family: strict safety is solved, but
AY3 fresh-tail generalization misses the fixed P95 gate by 0.3719%. Return to
a distinct film-look or physical-image-formation algorithm leaf.

U5.R2BK19 now passes a synthetic known-operator prerequisite for hard latent
modes. Two genuine residual directions are recovered at 100% held-group
accuracy with `.8154` absolute K2 error gain, while one direction at three
strengths has `.999909` minimum cosine and only `2.94e-6` absolute K2 gain.
This is not real-film or unpaired evidence. BK20 is the next algorithm leaf:
freeze multiple unpaired canonicalizers/control pools and return
`unidentified` whenever their inferred directions disagree.

U5.R2BK20 now provides that strict nonidentifiability result. Identity and a
nonlinear, invertible, cube-safe z-conditioned twist have byte-identical
unpaired observed sets, yet differ by `.06360` paired RGB RMSE and `.05900`
after best affine basic removal. Two canonicalizers recover no hidden twist.
The next algorithm leaf must therefore optimize film-inspired appearance value
with an explicit claim ceiling, canonicalizer sensitivity and hard fallback;
it cannot call reference-only output the real digital-to-film operator.

U6.P2M then tests a distinct density-dependent interimage development
equation without touching the sealed AO9 proxy pairs. Recovery is numerically
strong on unseen synthetic exposure groups, but the six independently fitted
couplings break an exact symmetric neutral-axis gate under measurement noise.
Close the fitter without a symmetry rescue or photograph stage. AO6 remains
the incumbent; the next algorithm leaf must be genuinely distinct.

U6.P2N proves exact dye-plus-retained-silver density composition, but P2O1
closes its fixed photographic profile: the look is strong/non-basic yet too
large, shifts luma, increases rather than reduces chroma, and creates isolated
high-difference points. Visual review and lower-strength rescue are forbidden.

U6.P6M passes a bounded primary-paper Callier mechanism audit. The exact
6,128,203-byte source and two byte-identical reports support material, density,
wavelength and illumination-geometry constraints, but provide no numeric Q
table. P6N may design a generic synthetic bounded operator; digitized-figure
fitting, scanner calibration, stock claims and photographic promotion remain
closed.

U6.P6N now passes all frozen synthetic gates with exact dye/diffuse identity,
strictly increasing directed density, `1.78e-15` inverse error and exact row
partitioning. Retain it as dormant generic infrastructure. Do not invent a
silver component for ordinary chromogenic film or reuse P2O1 photographs;
continue only with independent component evidence or another mechanism.

U6.P6O reproduces four thesis measurements from an exact 35,683,532-byte PDF
and explicitly preserves the silverLD table-rounding mismatch (`Q=1.40` versus
raw density ratio `1.45`). U6.P6P then closes the unchanged P6N quantitative
amplitude route: both silver samples exceed its fixed RGB envelope plus `.05`,
while dye identity and silver-over-dye controls pass. Do not retune P6N from
these four samples; continue a distinct physical or colour leaf.

U5.R2AO4R closes the missing mature root-polynomial baseline on the 71-pair
Velvia proxy pool. It regresses combined RGB RMSE by 180.82% versus the bounded
one-matrix control, regresses both cross-domain directions by more than 131%,
and has negative cube Jacobians plus 13.33% held-row OOG. This separates
inverse colour correction from forward film-look generation; do not rescue it
with clipping or more capacity.

U5.R2BL0 closes dense source-conditioned prediction: aggregate and tail error
improve, but only 13/33 held groups win. The next colour leaf must use a frozen
hard fallback or independent content-diverse evidence; dense routing cannot
advance.

U5.R2BO0 finds 53 title-exact same-scene film/digital weak pairs in one
218-photo album. The three capture families are balanced at 18/17/18 and all
selected rows are CC BY-NC-SA 2.0 with 1024px derivatives. Open only bounded
integrity/registration feasibility; one uploader and unknown stock, process,
exposure and scanner state still forbid fitting, training or calibration.

U5.R2BO1 acquires and verifies all 106 bounded derivatives (23.96MB): 53/53
pairs, zero exact duplicates and zero cross-pair dHash<=4. Open only frozen
registration feasibility; fitting/training remain closed pending geometry.

U5.R2BO2 registers 52/53 pairs across all three families; two runs are exact
and visual overlay review finds no obvious accepted warp failure. Keep the one
2.021px reject and open only a scene-split bounded-operator development pilot.

U5.R2BO3 closes a single family-average operator: curve+matrix wins 67.5% but
misses the frozen 5% median-over-basic gate at 4.03%; triangular also fails.
Keep 12 confirmation pairs unread and test per-scene case-bank Oracle value.

U5.R2BO4 closes retrieval: Oracle mean gain is 13.08% and wins 85%, but median
gains over global/medoid (8.76%/6.73%) and p95 ratio .943 miss frozen gates.
Do not train a similarity selector; confirmation remains unread.

U5.R2BO5 also closes a paired texture generator. The film-minus-digital
high-frequency direction reverses across the three capture families
(`.763/.730/4.138` energy ratios), pooled positive PSD support is zero and
correct alignment barely differs from a shifted control. Treat this as
camera/scanner/sharpening nuisance, not emulsion structure; confirmation stays
unread.

U5.R2BO6 closes a smooth shadow/mid/high positive-matrix operator. It is
exactly cube-safe, but mean/median held-scene gains are negative, it wins only
17/40 scenes and worsens the worst tail. Do not add knots or matrices; the
weak-pair confirmation remains unread.

U5.R2BO7 supplies the mature fixed-global LUT baseline. It is active,
Jacobian-positive and wins 33/40 scenes, but mean/median gains are only
2.09%/1.65% and P95 narrowly misses. Do not increase LUT degree or retune;
the next candidate must change information structure rather than capacity.

U5.R2BO8 tests source-only logit canonicalization and a shared preset with a
wrong-adapter control. Correct adapters recover only .048% median over wrong,
ordinary-control gain is .28%, and worst error grows 1.147x. Close this
canonicalizer without relabeling it as exposure/WB; confirmation stays unread.

U5.R2BO9-BP2 adds independent one-author B&W weak-pair evidence. Seven CC BY
2.0 film-over-digital composites all register, and the basic-normalized film
halves retain a cross-scene positive high-frequency/PSD residual while shifted,
edge and JPEG controls pass. Open one LOO mean-preserving density-domain
structure challenger only; this is not stock, emulsion, scanner or product
evidence.

U5.R2BP3 closes that stochastic challenger. Its single global analytical
no-clipping scale falls to `.0070`; PSD error improves only `.118%` over basic
and `.067%` over equal-energy white noise, while reversed shape slightly wins.
No visual review or per-pixel safety rescue opens. A distinct deterministic
density-domain acutance/spatial-response test is the next permissible use of
the paired B&W evidence.

U5.R2BP4 closes the deterministic density-acutance follow-up. Although all
seven held scenes improve nominally, the median PSD gain is only `.948%`, every
fold selects the weakest candidate, the matched linear-luma control is better,
and the safe scale falls to `.0267`. Stop fitting structure on this source;
continue with independent physical evidence or an unrelated explicit mechanism.

U5.R2BQ0-BQ0S2 acquired 512 independent FiveK digital-retouch controls and
retained 509 across 28 camera models, but the frozen normalization gate closes:
one prior-pool and one internal `dHash<=4` candidate exist. Both are visibly
unrelated low-information scenes, so the next ready leaf is a separately
versioned source-only multi-evidence duplicate adjudicator. BQ1 v1 operator
fitting, row substitution and threshold weakening remain forbidden.

U5.R2BQ0S3 rejects both dHash candidates using frozen source-only pHash/layout
and ORB/RANSAC evidence. BQ1v2 then proves strong evaluator-Oracle value on
128 target-hidden whole-camera rows: 33.39%/38.66% gains over pooled global and
29.67%/33.43% beyond global strength for aligned/filtered targets, with 87
distinct cases. Open source-only hard retrieval on development cross-validation;
no selector or product claim exists yet.

U5.R2BQ2A closes simple source-only nearest-case retrieval without rescue.
Tone-layout improves aligned/filtered confirmation mean error by 5.06%/7.45%,
but aligned wins 76/128 rows and misses the frozen 60% gate by one row. Open
one group-aware pairwise compatibility learner; keep hard Top-1/global fallback.

U5.R2BQ2B closes that pairwise learner on disjoint development cameras before
confirmation. Filtered passes, but aligned wins 54.81% and is 0.60% worse than
nearest. Do not add ranker capacity. Next test a bounded source-conditioned
explicit operator and require it to preserve strong style magnitude.

U5.R2BQ3 repeat-exactly closes direct source-to-parameter regression before
confirmation. It improves mean error over pooled global by 4.94%/11.14% and
beats strength and shuffled controls, but retains only 32.63%/47.30% of target
style and creates 0.3906%/0.1953% new epsilon-boundary samples. Keep all gates
frozen and open one factorized direction/magnitude plus analytical safe-residual
leaf; no dense-RGB, film or product claim opens.

U5.R2BQ4 factorization is a strong near-miss: all performance/control/safety
gates pass, mean gain over safe global is 11.48%/14.68%, and analytical residual
execution yields zero new boundary. Absolute target-style retention is only
45.58%/69.05% versus 70%, so confirmation remains unread. Diagnose the fitted
operator/Oracle style ceiling next; change renderer capacity only if the ceiling
itself is inadequate.

U5.R2BQ4D shows the current self-fit retains only 43.84%/58.52% of target
style, while every held row has a safe strong-style case and a style-constrained
Oracle costs only 3.62%/1.97% error. Close triangular fitting/routing rescue.
Next replay the mature strict-interior projection-curve family as a capacity
baseline on this bank before training any new selector.

U5.R2BQ5 closes the unchanged mature projection-curve baseline. Its group-held
Oracle is strong (24.46%/31.44% over global), but Jacobian safety collapses
median dose to .247/.300 and self-fit style to 27.59%/32.14%. More curves are
not enough without intrinsic regularity. Test the mature strict-interior 4^3
residual LUT next; no projection-curve selector or confirmation opens.

U5.R2BQ6 passes the unchanged strict-interior 4^3 LUT case bank: self-fit gain
is 49.78%/52.26% with 78.19%/79.41% style, and group-held Oracle gain is
30.06%/34.25% with 76.45%/82.65% style. Boundary/cube rates are zero. Operator
capacity is sufficient; BQ7 now tests source-only factorized LUT inference
against global and hard-retrieval controls. Confirmation stays closed.

U5.R2BQ7 closes dense LUT prediction. It carries source signal (6.89%/8.98%
over global; 15.01%/18.31% over shuffled) but averages target style to
53.14%/57.91% and wins only 45.19% versus global strength selection. BQ8 keeps
one real BQ6 LUT intact and learns hard Top-1 source/case compatibility; no
dense blending or BQ7 rescue opens.

U5.R2BQ8 keeps strong style (121.08%/125.47%) and beats random/shuffled, but
regresses 27.14%/21.68% versus global with poor tails. Close direct hard Top-1.
BQ9 holds its selected case identities fixed and measures only a
style-constrained global-to-case path Oracle before any strength learner.

U5.R2BQ9 finds 11.82%/15.33% mean path-Oracle value with 79.86%/82.43% style,
but only 39.26%/49.63% of rows improve and style feasibility is 85.93%/88.15%.
Close confidence/strength rescue. BQ10 tests a fixed fit-only strong-style
medoid as the K=1 alternative to coefficient-median blandness.

U5.R2BQ10 fixed champions retain 135.42%/134.74% style but cost 22.78%/20.53%
mean error versus global. Close the FiveK adaptive route without confirmation.
BQ1-BQ10 establish strict-interior LUT capacity but not source routing; retain
AO6/global safe-rich and return to real-film/physical-image-formation work.

U5.R2BS0 retains one latest external colour-chemistry mechanism: an
amplitude-matched non-spatial Langmuir donor passes exact automatic and severe
visual gates, ranking over its linear donor on 6/9 blinded photographs. It is
not an integration or preference promotion. The next ready leaf is one
clean-room bounded donor primitive and synthetic identifiability pilot.

U5.R2BT0 closes the latest spatial explicit baseline without rescue. A
clean-room 4x4x8 spatial/luma-rank colour-gain field is repeat-exact and beats
its spatial-only control, but does not beat a parameter-matched global RGB LUT
on held blocks (`-2.82%/+0.56%` mean; `29.17%/39.58%` wins) and creates up to
`2.49%` new encoded boundaries. No predictor, confirmation or visual gate
opens; return to new real-film or physical evidence rather than more grid
capacity.

U5.R2BU0 establishes one stock-first physical source signal without changing
the image operator. Exact Kodak VISION3 50D/250D/500T high-frequency MTF traces
separate all three pairs in at least two layers (`fafed347...6020b` twice),
while the characteristic-shape bank remains closed. BU1 may compile only an
explicit source-domain prior bundle; no colour look, roll calibration or
product path opens.

U6.P2V passes that clean-room primitive with exact linear/reference limits,
bounded monotone Jacobian behavior, neutral-axis and partition invariance.
U6.P2W then produces a visible effect on all 16 photographs and passes every
non-artifact automatic gate, but one image has 283 isolated excursions against
the frozen zero gate. Visual review is forbidden and this realization closes
without rescue. U6.P2X tests a distinct intrinsically bounded sigmoid family;
two exact runs remain boundary-clean but fail value (`.16768` median; 9/16
visible) and isolated-excursion (126) gates. Stop heuristic photographic
interimage families. U6.P2Y finds a genuine measured mechanism but zero
published complete numeric rows. U6.P2Z then verifies the latest Emulating
Emulsion paper and capture design but finds no reproducible public code,
parameters or complete paired rows. Both close without fitting. U6.P6Q is the
next ready leaf: audit Kodak Digital LAD as an official machine-readable
recorder/print calibration anchor, not a stock-response target. P6Q confirms
the guide semantics but closes the archive-image route because a 51,019,264-byte
member exceeds the frozen 50,000,000-byte limit before decompression. P6R
passes the H-387 guide-table-only typed code-to-density primitive: all 1,024
codes roundtrip within `1.14e-13`, negative/IP domains remain separate, and no
RGB transform occurs. P6S then passes the existing generic print interpretation
without photographic tuning: 824 compatible codes per negative/IP branch,
exact neutral LAD input density, all 16 gates true and two exact reports
(`e57aa2b...d7eab`). P6T is ready: freeze and test a local LAD print-chain
sensitivity/Jacobian audit without converting this anchor into camera-film or
stock calibration. P6T passes two exact runs (`9bb08b2a...1bd57`): analytic
error `3.83e-10`, whole-domain code-tangent relative error `1.36e-5`, and LAD
local error `2.76e-4`. Retain the explicit Jacobian and continue to a distinct
print/profile-compiler leaf; do not spend another leaf on neutral LAD glue.
P6U analytically reverses the full explicit print chain on 343 interior density
points with `8.88e-16` maximum inverse error and condition number `37.82`; five
invalid/unreachable controls reject and all 12 gates pass in two exact reports
(`2575423d...50654`). The next leaf should compile this reversible pair into a
versioned generic profile or use it in a bounded fitting experiment. P6V closes
the fixed FP16 single-LUT compiler: 17-cube fails three accuracy gates; 33-cube
misses only frozen RMSE (`.00027461 > .00025`) despite positive topology, zero
range escape and zero inverse rejection. Do not add size/precision or relax the
gate; retain the exact analytic pair and move to a distinct factorization.
P6W instead streams the unchanged explicit equations in float32. Two 24MP runs
produce the same SHA with max/RMSE error `3.88e-7/5.47e-8`, peak `171.5–171.7MB`
RSS and `9.57–10.04s` wall; all 11 gates pass without full-frame output or
clipping. Retain this ordinary-CPU reference and open native CPU parity, not a
larger LUT. P6X then evaluates the Balica-Trumpy 2025 fixed multispectral
centres with a shared nonnegative XYZ compiler. Paper7/paper8 are strong in
absolute CIEDE2000 error, but the fresh CAVE median improvement over RGB3 is
only `64.74%`, below the frozen `75%` gate. Close this ideal-centre family; a
successor requires measured LED/sensor responses or an independent film MSI
cube, not band or threshold rescue.
P6Y audits the newest controlled Ektachrome E100/Velvia 50 study's public NTNU
deposit rather than inferring data availability from the paper. The exact 109-page
thesis confirms the two-stock/two-light/multi-exposure/120-film/MSI-HSI design,
but the record exposes only one thesis PDF and one administrative ZIP: zero
machine-readable pixel payloads, zero repository links and no explicit pixel reuse
rights. Two reports are byte-identical (`e66066f1...24fbd`). Close acquisition
before pixel decode or fitting; reopen only for an exact rights-clear data release.
P6Z then tests the newest measured scanner context-glare table with one bounded
same-hardware fraction transferred across laboratories. It wins 7/9 against
both no-context and global controls, but confirmation p95 black-context error is
`.03588 > .02`; two reports are exact (`f71f52ff...f85165`). Close this exact
transfer and retain scanner/lab/polarity processing as nuisance, not stock truth.

U6.P4AS passes the joint dye-cloud signature test. One positive Gaussian cloud
scale per layer is frozen from development MTF only, then predicts disjoint MTF
with zero max RMSE and independent Poisson NPS with `.00812` worst shape RMSE.
Median NPS gains over no diffusion/wrong-layer controls are `95.90%/91.64%`;
NPS-only versus MTF scale p95 disagreement is `3.68%`. Retain the shared-scale
mechanism, not a real-film Gaussian radius. The next data leaf requires
rights-cleared MTF and NPS measured on the same material/process group; absent
that evidence, continue a distinct mechanism without photographic rescue.

U6.P4AT passes the no-fit source audit. Two exact reports (`53748b3e...fd447b`)
bind 14 multi-aperture granularity rows and 243 numerical MTF values; the source
states same packages/process, composite x-ray/film/process/scanner MTF and
figure-only PSD. The printed table has 27 series while prose says 26; retain the
contradiction. U6.P4AU is next: fit cloud scale from RMS ratios only, freeze it,
then test whether it explains film-type MTF differences after a shared nuisance
factor better than no-diffusion and wrong-film controls.

U6.P4AU closes the measured Gaussian shared-scale family. RMS-only fits are
compact and accurate (`5.71-6.89um`, max log RMSE `.05884`), but Table 25
residual dispersion regresses `2.02%` versus no diffusion and improves only
`1.15%` versus wrong-film, below both frozen 5% gates. Two reports are exact
(`574f0c59...569c7a`). Retain P4AS synthetic mechanism only; do not rescue this
profile on the same tables. Continue a distinct physical algorithm leaf.

U6.P6AA passes the isolated circular-aperture reference. The 13x13 positive
kernel has `.00314` maximum analytic MTF error and reduces RMSE by
`95.78%/99.53%` versus equal-moment Gaussian/point controls; impulse, constants,
rotation, repeat and row partitions pass in two exact reports
(`e25be197...956891`). P6AB now tests the same fixed 12.5um compiler directly at
4000-dpi pitch; production profile/runtime imports remain closed.

U6.P6AB is frozen at exact 4000-dpi pitch (`6.35um`) with the unchanged
disk-area compiler. It must stay within `.05` analytic MTF error through
Nyquist, improve RMSE by `50%/80%` over Gaussian/point controls and preserve
the P6AA execution invariants. No oversampled fallback or fitted correction is
allowed in the target path.

U6.P6AB closes the direct 4000-dpi compiler. Its 3x3 positive kernel preserves
constant, impulse, rotation, repeat and row-partition invariants and beats a
point aperture by `82.17%`, but maximum analytic MTF error is `.12906` and its
RMSE is `60.80%` worse than the equal-second-moment Gaussian. Two reports are
byte-exact (`74d0a252...49da48`). Do not rescue this discretization with wider
support, negative taps or a fitted correction; retain P6AA only as an offline
reference and continue a mechanism-distinct physical imaging leaf.

U6.P6AC is frozen as that distinct leaf. It keeps colour dye clouds in developed
density, converts to transmittance before scanner optics, integrates the 12.5um
circular aperture on an 8x physical lattice and only then samples at 4000 dpi.
A separately rendered 16x lattice is the reference; wrong density ordering,
post-raster P6AB and equal-moment Gaussian are fixed controls. This is an
offline synthetic reference experiment, not a P6AB rescue or product profile.

U6.P6AC confirms the physical ordering signal but closes the 8x raster
approximation. It improves RMSE by `91.95%` versus the best post-raster control
and `89.10%` versus aperture-on-density, yet misses its independent 16x
reference gates at `.006914 > .005` RMSE and `.013875 > .0125` P95. Two reports
are exact (`203c0676...30f846`). Retain the deterministic 16x path only as an
offline reference; P6AD may test direct continuous-cloud aperture quadrature,
not another oversampling factor or post-raster kernel.

U6.P6AD is frozen with no intermediate raster. It evaluates 256 deterministic
golden-angle disk samples per target pixel directly against the exact marked
cloud centres/radii, with 64 samples and the frozen P6AC 8x raster as controls.
It must meet the unchanged `.005/.0125` reference errors and materially beat
both controls; the sample count, sequence, geometry and gates cannot be tuned
after scoring.

U6.P6AD closes unchanged. The 256-point direct method improves RMSE `44.47%`
over the 64-point control, but misses the frozen 16x reference at
`.007340/.014538` RMSE/P95 and is `6.17%` worse than the P6AC 8x raster. Two
reports are exact (`b6bca9b9...0c96cd5`). Do not increase samples. Before
testing another compiler, P6AE must establish whether the 16x raster target
itself is converged.

U6.P6AE revokes that 16x target. FFT and spatial 16x convolution agree within
`1.18e-14`, and P95 plus the 8x-to-16x convergence gain pass, but the frozen
16x-to-32x RMSE is `.0031875 > .003`. Two formal reports are byte-identical
(`ad1155d4...8a66ef`). Do not rescue P6AC/P6AD or fit a convergence correction.
U6.P6AF must test the provisional 32x render against a separately rendered 64x
reference before another scanner compiler is selected.

U6.P6AF closes the raster-reference family. The 32x output regenerates exactly;
P95 `.003033`, convergence gain `67.10%` and exact replay pass, but 32x-to-64x
RMSE `.0015179` narrowly exceeds the frozen `.0015` gate. Two reports are exact
(`6a048782...f27397b`). Do not continue to 128x. U6.P6AG must instead test an
adaptive continuous-cloud/circular-aperture integral with an explicit error
bound and analytic unit geometries.

U6.P6AG closes before a formal report: a cloud-dense fixture pixel causes
QUADPACK roundoff-limited nonconvergence at the frozen `1e-10` tolerance. No
metric or gate decision was emitted; do not loosen tolerance or suppress the
warning. U6.P6AH may instead enumerate exact circle-topology intervals and use
analytic circle-chord antiderivatives, with no adaptive quadrature or raster
truth.

U6.P6AH also closes unchanged. Five analytic geometries are exact; range,
repeat and 3/7/11-row partitions pass, but independently rebuilding the entire
integral after swapping x/y axes differs by `3.96e-10 > 1e-10`. Two reports are
byte-identical (`ea8bd2c6...ceccde`). Do not use higher precision or reordered
sums as a same-fixture rescue. Seek rights-cleared measured scanner PSF/MTF
evidence or advance another ready U6 physical mechanism.

U6.P6AI closes before confirmation. Synthetic Gaussian and row-order controls
pass, but the eight development scans have zero common eligible transitions
against eight required; 52/104 edge attempts fail frozen robust line support
and scan 34407 has 0/13 measurable edges. Two development locks and reports
are byte-exact (`65075697...4a11`), with zero confirmation reads. Do not alter
geometry or thresholds on this cohort and do not infer scanner or film MTF.

# 2026-08-02 - U5.R2BU1 one-Gaussian MTF compiler closes

- Three VISION3 stock-specific layer kernels beat shared/wrong/identity
  controls, but confirmation median RMSE `.05307` misses the frozen `.05` gate.
- Close without Gaussian-mixture or threshold rescue; pursue a fresh
  stock-identifying source observation.

# 2026-08-02 - U5.R2BU2 granularity diversity passes

- First-party 50D/250D/500T 48 µm Sigma-D curves pass all source and diversity
  gates; every pair is material in all three channels.
- Open only an identity-bound observed-source bundle. No NPS, microscopic
  structure, placement, photographic render or product mapping opens.

# 2026-08-02 - U5.R2BU3 observed source profile retained

- Two formal compiler runs are byte-identical (`49c75234...306f` report,
  `233decd3...5252` bundle) and retain exact BU0/BU2 curve identities.
- The bundle is explicitly non-renderable and contains no PSF, NPS, particle
  geometry, placement or RGB transform. Next seek independent measured
  real-film evidence rather than elaborating this serialization layer.

# 2026-08-02 - U5.R2BU4 source signature robustness passes

- Two exact reports (`5dc51d14...e2fc`) classify all 60 frozen perturbation
  and missing-layer cases correctly; the worst normalized joint margin is
  `.56943`, while cyclic relabeling scores zero.
- This is same-source digitization robustness, not another roll or scanner.
  Rendering remains closed until an independent measured real-film observation.
