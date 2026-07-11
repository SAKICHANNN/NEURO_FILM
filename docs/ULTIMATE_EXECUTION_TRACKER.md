# K-MCFM Ultimate Execution Tracker

> Active planning authority from 2026-07-10.
> Strategy and research basis: `docs/planning/ULTIMATE_ROADMAP_2026.md`.
> This tracker records intended work; only rows marked `complete` are implemented facts.

---

## 1. Parent goal and task contract

**Parent goal `ULT`**: deliver a local-first, color-managed film-imaging product that looks strongly stylized while producing no confirmed severe glitch/artifact on the frozen gold set. Calibrated stock/process reproduction is a deferred optional evidence lane with stricter claim requirements.

**Autonomy invariant**: the active Style-safe/FilmCase path cannot depend on the owner supplying images, film/digital pairs, per-image labels, new preference votes or manual annotation. The frozen preference set `53/55/56/33/09/03/02/01` is the only current owner-preference evidence. Research may proceed autonomously within existing rights/cost gates; external human validation is a later release activity, not a prerequisite for the research DAG.

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
| FilmCase is the primary bounded-ML research hypothesis | candidate | Learn which explicit transformation applies to a scene instead of regressing an average RGB target | Retire if identifiability or Oracle-routing gates fail |
| Unpaired evidence supports only film-inspired claims | accepted for plan | A film scan does not identify the stock/process/scanner transform without a matched input | Only a rights-cleared paired measurement lane can promote calibrated claims |
| Generative image models are excluded from FilmCase | accepted for plan | User explicitly excluded the class; FilmCase renderer and supervision remain non-generative | Only a new explicit user instruction |
| Generative editing is isolated | accepted for plan | User excluded it from FilmCase; current paths are also artifact-prone | Future Creative work requires a separate explicit instruction and approval |
| Stock profile includes process and interpretation | accepted for plan | Negative/slide/B&W do not have one intrinsic display RGB look | None; schema invariant |
| Portra 400 + Velvia 50 are pilot stocks | proposed | Orthogonal negative/slide behaviors and high user value | Availability, rights or lab feasibility fails |
| FiveK is optional neutral auto-base only | accepted for plan | Expert retouch is not film identity; full local sources were deleted | Sources/rights restored and product evidence supports it |
| Community/Flickr/FilmSet assets are research-only by default | accepted for plan | Missing or limited rights; FilmSet is Capture One recipe target | Per-asset legal clearance |

---

## 3. DRPT node tree

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
    └→ U4 artifact/style/evaluator foundation ─────┼→ U5 FilmCase + U6 effects → U7 → U8
                                                   └→ U3 remains a deferred optional evidence lane
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
| U0.3 | complete (reference lane blocked) | Isolated manifest-v2 lineage/eligibility audit; all uncertain rows quarantined | none | 4,210/4,210 classified; 0 source groups/0 eligible rows; 140 near pairs retained inside quarantine; evidence: `docs/FILMCASE_U03_LINEAGE_AUDIT.md` |
| U0.4 | complete | CPU-safe CI, checksum baseline, local environment capture and explicit non-gold benchmark registry | U0.1 | `configs/reproducibility_baseline.json`, verifier, CI workflow, 25 local tests; evidence: `docs/REPRODUCIBILITY_BASELINE.md` |
| U0.5 | complete | Correct stale FiveK/data reproduction state | U0.1 | Reproduction manifest now records the local legacy manifest and deleted full-source boundary; 2026-07-11 |

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
| U4.3 | pending | Graded content/effect quality diagnostics | U4.1 | face/text/edge/texture plus NPS/radial/MTF/tile/determinism report |
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

### U5 — Bounded-AI challenge

| ID | Status | Deliverable | Dependencies | Exit evidence |
|---|---|---|---|---|
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

Stop at the simplest passing model. A valid result is “the analytic/global renderer wins; delete the retrieval and neural dependency.” Identifiability failure closes the unpaired-reference lane; Oracle failure closes routing. A film scan is never treated as an input/output pair.

The detailed hypotheses, CaseRecord schema, evaluator protocol, experiment DAG, numeric preregistration defaults and failure branches live in `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`.

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
4. `U4.1/U4.2`: freeze severe-artifact examples and the autonomous style/appeal scorecard; `U4.6` is only for calibrated profile claims.
5. `U0.2`: ask owner to confirm intended code license and release posture.

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

### Week 4 — evaluation and FilmCase identifiability

1. `U4.1`: expand the frozen safety corpus with face/text/skin/sky/ramp/texture/highlight cases.
2. `U4.2/U5.FC1`: preregister the autonomous blind visual protocol, nuisance-matched controls and source-controlled identifiability test.
3. `U4.5`: benchmark current renderer on exact target hardware.
4. Keep `U3` deferred; it is not an owner-data request or dependency.

No film purchase, lab booking, model download or GPU training occurs without the corresponding approval.

---

## 6. Experiment queue

| Experiment | Hypothesis | Fixed baseline | One changed variable | Promotion gate | Stop condition |
|---|---|---|---|---|---|
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
- Autonomous unpaired FilmCase scientific DAG: `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`
- Initial visual audit and deterministic bridge experiment: `docs/VISUAL_STYLE_AUDIT_2026-07-11.md`

---

*Tracker initialized: 2026-07-10; FilmCase subtree frozen: 2026-07-11. U0.3/U0.4/U0.5 completed 2026-07-11; reference-derived FilmCase is blocked by missing provenance, while frozen-anchor, deterministic-renderer and U4 gold/stress evaluation leaves remain ready. Integration owner: repository owner or explicitly assigned Codex root agent.*
