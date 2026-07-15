# Project Agent Log

Durable handoff log for non-trivial DRPT-governed work. Keep implementation truth in the active project documents and use this file to record decisions, evidence, risks and the next safe handoff.

---

## 2026-07-10 — Ultimate research and roadmap reset

- **Parent goal / node:** `ULT` / `U0.0`
- **DRPT level / mode:** L2, Mode B; root integration owner with three read-only research leaves
- **User objective:** conduct a thorough study and produce a complete plan to make K-MCFM “ultimate”
- **Primary workflow:** `dev-research-reliability`
- **Routing / secondary skills:** `codex-super-router`, `codex-super-research-harness`, `scientific-research-harness`, `codex-super-aiml-harness`, `plan-tracker-discipline`, `exa-search-preference` (Exa unavailable; native web fallback), `drpt-bi-governance`, `project-agent-log-discipline`, `project-structure-steward`

### Decisions

1. Retired diffusion-first SDXL/IP2P/SDEdit as the default architecture. Local evidence shows content/detail rewriting or infeasible 12GB training behavior.
2. Selected a calibrated hybrid architecture: deterministic high-precision Reference core, optional bounded curve/LUT/bilateral-grid predictor, isolated Creative generative mode.
3. Defined named-film identity as `stock + exposure/process + scan/print interpretation`, not one prompt or LUT per stock.
4. Selected Portra 400 and Velvia 50 as the proposed two-stock calibration pilot; no capture/purchase was authorized or started.
5. Classified Flickr, FilmSet, FiveK, FilmGrainStyle and community LoRAs as research-only by default; production claims require owned/cleared paired data.
6. Replaced a single aggregate evaluator with an initial four-part design: safety, authenticity, physical effects, and human/product performance. This was superseded on 2026-07-11 by the five-scorecard design that adds film-style salience.
7. Kept FLUX.2 Klein 4B as an optional teacher/Creative challenger. Official sources conflict on roughly 8GB versus 13GB VRAM, so 12GB compatibility is not assumed.

### Files changed

- `docs/planning/ULTIMATE_ROADMAP_2026.md` — research synthesis, architecture, data/capture, evaluation, product, risk and source matrix
- `docs/ULTIMATE_EXECUTION_TRACKER.md` — active DRPT tree, DoR/DoD, dependencies, branch gates, first 30 days and experiment queue
- `AGENTS.md` — corrected current truth and project invariants
- `README.md` — replaced failed diffusion quick start with current renderer/status and plan links
- `IMPL_PLAN.md` — added active-plan supersession banner while preserving historical V3 content
- `TASK_BOARD.md` — replaced stale diffusion board with compact current queue
- `docs/CONTENT_PRESERVING_FILM_TASK_TRACKER.md` — marked historical for active planning
- `docs/WINDOWS_TASK_TRACKER.md` — marked historical machine snapshot
- `docs/planning/GAP_ANALYSIS.md` and `docs/ONLINE_DATA_AUDIT.md` — added supersession warnings
- `docs/planning/README.md` — indexed the new strategy and active tracker
- `docs/drpt/AGENT_LOG.md` — this entry

No production code, data, model, output or user-owned untracked file was modified.

### Evidence and verification

- Audited current renderer, preprocessing, effects, neural LUT, FiveK, data/license and experiment reports.
- Confirmed `scripts/render_film.py` accepts only `safe_lab`, uses PIL RGB input and writes 8-bit PNG.
- Confirmed full FiveK sources were deleted locally after a 6.98GB freeze pack was retained.
- Confirmed no GitHub Actions workflow and 5 test files / 18 test functions.
- Ran `.\.venv\Scripts\python.exe -m pytest -q` before and after documentation work; final pre-commit result: `18 passed in 13.18s` (an earlier post-edit run passed in 9.48s).
- Ran `.\.venv\Scripts\python.exe scripts\render_film.py --help` to verify the documented current CLI surface.
- Read-only leaf checks compiled 105 Python files in memory, imported core modules and reported `pip check` clean.
- External claims were checked against papers, official repositories/model cards, standards bodies, manufacturer data sheets and official product documentation.
- Checked all changed Markdown for trailing whitespace: 0 findings.
- Checked relative Markdown link targets in all changed documents: 0 missing targets.
- Ran `git diff --check`: clean after removing Markdown hard-break whitespace.

### Unresolved risks and approvals

- Root license is absent despite historical MIT claims; owner/legal decision required.
- Current branch has no matching remote branch and contains local commits; no push authorized.
- Existing untracked `halationguide.md` and `scripts/make_velvia50_scheme_comparison_sheets.py` remain protected user work.
- Capture/lab/scanner budget, participant releases, model/data downloads, paid GPU, external contact and public release all require explicit approval.
- Exact capture cost, lab/scanner repeatability, M5 thermal performance and final numerical gates remain pilot measurements, not settled facts.

### Handoff

`U0.1` was completed in this change set. Next ready leaf is `U0.3`: manifest/leakage repair, followed by `U0.4/U4` CI/evaluation foundation. Code implementation should not begin before refreshing Git state and claiming bounded files in `docs/ULTIMATE_EXECUTION_TRACKER.md`.

---

## 2026-07-11 — Velvia 50 user-preference anchor

- **Parent goal / node:** `ULT` / `U4.7`
- **DRPT level / mode:** L1, Mode A
- **Input:** user stated the preferred numbered schemes as `53, 55, 56, 33, 09, 03, 02, 01`
- **Resolution:** mapped the numbers through `outputs/contact_sheets/velvia50_all_schemes_numbered_20260616/NUMBER_MAP.csv`
- **Decision:** retain the set as qualitative preference evidence without assuming list order is a ranking; treat `53/55/56/09/01` as full 20-render anchors and `33/03/02` as smoke-only direction cues
- **Interpretation:** all five comparable choices are deterministic baseline/gamut-safe Lab variants, supporting the calibrated deterministic core; this is preference evidence, not film-authenticity evidence
- **Files changed:** `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/planning/ULTIMATE_ROADMAP_2026.md`, `docs/drpt/AGENT_LOG.md`
- **Verification:** exact scheme labels and render counts checked against `NUMBER_MAP.csv`; no output image, mapping file or user-owned untracked file modified
- **Handoff:** include the five full anchors in the frozen U4 blind-study set and rerender them on one common held-out image set before inferring strength/style parameters

---

## 2026-07-11 — Add minimum film-style salience gate

- **Node/parent goal:** `U4.8` / `ULT`
- **Trigger:** user reported that many theoretically stronger schemes look technically clean but have little film style
- **Skills used:** `plan-tracker-discipline` primary; `project-agent-log-discipline` secondary
- **Decision:** expand evaluation from four to five independent scorecards by adding film-style salience between safety and stock authenticity
- **Files changed:** `AGENTS.md`, `TASK_BOARD.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/planning/ULTIMATE_ROADMAP_2026.md`, `docs/drpt/AGENT_LOG.md`
- **Verification intent:** future U4 evaluation must run separate color-only and full-look blind tests against neutral input, the preferred deterministic anchors and real film references; global saturation alone cannot pass
- **Risk/unknown:** salience thresholds remain a preregistered pilot decision; this feedback establishes the gate but does not yet quantify it
- **Handoff state:** U4.6 remains ready for evaluator implementation; no render, model, data or user-owned untracked artifact was modified

---

## 2026-07-11 — Make strong style under severe-artifact constraint the product standard

- **Node/parent goal:** `ULT` architecture decision / U4 objective propagation
- **Trigger:** user explicitly defined the true standard as “looks highly stylized while showing no severe glitch/artifact”
- **Skills used:** `plan-tracker-discipline` primary; `project-agent-log-discipline` secondary
- **Decision:** product optimization is now `maximize(style strength × appeal) subject to no confirmed severe artifact on the frozen gold set`; report artifact rate/CI on the wider stress set
- **Claim boundary:** stock/process authenticity is conditional for profiles labeled calibrated; strong `film-inspired` styles can ship without paired-film evidence if artifact, rights and product gates pass
- **Propagation:** updated project identity/modes, critical path, U3 calibration dependency, U4 scorecards, U5 experiment gates, phase order, North Star and user-facing README/TASK_BOARD
- **Files changed:** `AGENTS.md`, `README.md`, `TASK_BOARD.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/planning/ULTIMATE_ROADMAP_2026.md`, `docs/drpt/AGENT_LOG.md`
- **Verification intent:** freeze severe/moderate/intended-style examples; rerender `53/55/56/09/01` on one gold/stress set; blind-rate style strength, appeal and artifact severity separately
- **Risk/unknown:** the severe rubric, gold/stress membership and quantitative preference threshold are not implemented yet; no universal zero-artifact claim is permitted
- **Handoff state:** U0.3/U0.4 remain the next ready prerequisites, followed by U4.1/U4.2; optional U3 paired calibration no longer blocks Style-safe product work

---

## 2026-07-11 — First vision-led Velvia style audit

- **Node/parent goal:** `U4.9` / `ULT`
- **Trigger:** user asked whether visual capability could actively advance the style/artifact objective
- **Skills used:** direct visual inspection; `plan-tracker-discipline` primary; `project-agent-log-discipline` secondary
- **Evidence inspected:** numbered 56-scheme overview; preferred #01/#09/#53/#55/#56 contact sheets; safe-rich, local-map, NILUT and context-4D comparators; union-40 metrics; full-resolution union IDs 09/11/29
- **Findings:** preferred family shows stable cyan/blue shadow versus warm red/orange/yellow separation and accepts medium-high luma movement; safe-rich/local maps are technically clean but bland; NILUT movement is less coherent; clipping/chroma/L-SSIM alone do not predict preference
- **New artifact evidence:** union ID 11 exposes neon-red highlight speckling in #09/#56 even when #09 reports zero new clipping, proving clipping is not a sufficient artifact gate
- **Files changed:** added `docs/VISUAL_STYLE_AUDIT_2026-07-11.md`; updated `docs/ULTIMATE_EXECUTION_TRACKER.md` and this log
- **Verification:** scheme recipes cross-checked against manifests; union summaries parsed for 01/09/53/55/56/10/local-map/NILUT; source and output images inspected at original resolution; no output image or user-owned untracked file modified
- **Handoff state:** after U0.3/U0.4 and U4 gold/stress freeze, run `EXP-VIS-00`: #56-like palette strength with #09-like containment, comparing source versus chroma gamut compression

---

## 2026-07-11 — Freeze the autonomous unpaired FilmCase research DAG

- **Parent goal / node:** `ULT` / `U5.FC0`
- **DRPT level / mode:** L2, Mode B; root integration owner with three read-only leaves for experiment design, literature review and authority/propagation audit
- **User objective:** create a complete stepwise research plan for a non-generative ML method that learns strong film-like color without film/digital pairs, new user data or further user labeling
- **Primary workflow:** `plan-tracker-discipline`
- **Routing / secondary skills:** `codex-super-router`, `drpt-bi-governance`, `project-agent-log-discipline`, `project-structure-steward`, `codex-super-research-harness`, `codex-super-aiml-harness`, `scientific-research-harness`, `dev-research-reliability`

### Decisions

1. Added FilmCase as the primary bounded-ML research hypothesis: source-controlled identifiability, explicit bounded case transforms, Evaluator Oracle, generic retrieval, optional transform-aware asymmetric ranking, hard sparse routing and OOD fallback.
2. Made the no-user-input constraint an architecture invariant. Images, film/digital pairs, new preference votes and manual labels are not active dependencies; `53/55/56/33/09/03/02/01` remains the only owner-preference evidence.
3. Defined a case as a replayable bounded transform plus applicability descriptor, unpaired/anchor evidence, artifact state, rights and lineage. An unpaired scan is never treated as an input/output transformation.
4. Added two hard scientific stop gates before router training: unpaired style identifiability after source/uploader/scanner/scene controls, and Oracle-over-global routing value.
5. Selected the capacity ladder `CCM + monotone curves → smooth 3D LUT → SepLUT → global + bounded case residual → optional bilateral grid`. This incorporates 2026 unpaired-ISP evidence that noisy pseudo-pairs can make a more expressive 3D LUT less stable than a constrained linear color head.
6. Selected cross-application applicability ranking rather than dense RGB reconstruction. Default inference is hard Top-1/medoid selection; low confidence, OOD or unknown color state falls back to the global deterministic champion.
7. Froze an autonomous visual protocol with shuffled repeated passes, nuisance-matched controls, full-resolution adjudication and preserved raw votes. It is explicitly not represented as population preference.
8. Limited the autonomous claim ceiling to `film-inspired/unpaired-evidence`; calibrated named-stock claims still require a separate paired measurement lane.
9. Marked U3 calibration and external human validation deferred. They are not repeatedly requested from the user and do not block FilmCase.
10. Excluded generative image models from FilmCase. Existing generative entries remain isolated historical/Creative work requiring a future explicit instruction.
11. Kept internal U2 schema/renderer research independent of the blocked repository-license decision; U0.2 remains a public-release gate.
12. Ordered the deterministic global frontier (`U5.1`) before case-bank construction (`U5.FC2`) and made legacy bilateral/mask nodes explicit FC7 implementation subleaves.
13. Defined Oracle-gap closure on one lexicographic scene-group pairwise endpoint, using cluster bootstrap/permutation; repeated vision shuffles measure consistency and do not inflate statistical N.
14. Made CaseRecord rights per-reference with a strict allowed-use intersection and prohibited gold/stress/final-test samples from active case memory.

### Files changed

- `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md` — complete scientific contract, architecture, data/evidence lanes, hypotheses, DRPT nodes, numeric preregistration defaults, experiment DAG, failure branches, risk register and source matrix
- `docs/ULTIMATE_EXECUTION_TRACKER.md` — U5.FC0–U5.FC8 subtree, autonomy/claim ADRs, deferred U3/U4.4, EXP-FC-00–08 and next-leaf handoff
- `docs/planning/ULTIMATE_ROADMAP_2026.md` — FilmCase-first rationale, unpaired data lane, Oracle tree, evaluation protocol, risk and primary sources
- `AGENTS.md`, `README.md`, `IMPL_PLAN.md`, `TASK_BOARD.md` — propagated current truth, critical path and no-user-input boundary
- `docs/planning/README.md`, `docs/PROJECT_STRUCTURE.md` — indexed the subplan without creating a second active tracker or new top-level structure
- `docs/drpt/AGENT_LOG.md` — this entry

No renderer code, data, model, output or user-owned untracked file was modified.

### Evidence and verification

- Rechecked local V1 shared-basis/L1 collapse, V2 pseudo-teacher evidence, local-map saturation shortcut, the preferred anchor mapping and the union ID 11 red-speckle failure.
- Read-only design audit confirmed current local query assets: union40 and the FiveK freeze pack; FiveK remains query diversity only, never film truth. Current Velvia Flickr rows lack sufficient uploader/roll/scanner lineage for immediate identifiability claims.
- Primary-source review covered context-based enhancement, retrieval-augmented retouching, semantic/OT unpaired pseudo-pairing, Neural Preset, Modulated Flows, SepLUT, HDRNet, SA-LUT and RSFNet. No paper was treated as an end-to-end solution to this project.
- Checked all changed-document local Markdown targets: `LOCAL_LINKS_OK`.
- Checked Markdown fence balance: `FENCES_OK`.
- Checked the new plan for trailing whitespace: none.
- Ran `git diff --check`: clean; only line-ending notices.
- Ran `.\.venv\Scripts\python.exe -m pytest -q`: `18 passed in 6.37s`.
- Final read-only propagation audit found all substantive P0/P1 authority, dependency, statistic and rights issues resolved; the scoped commit remained the only procedural completion condition.
- Protected untracked `halationguide.md` and `scripts/make_velvia50_scheme_comparison_sheets.py` remain untouched and unstaged.

### Risks, approvals and handoff

- Current Flickr/FilmSet evidence remains research-only and may fail source-controlled identifiability; that is an intended falsification outcome, not a reason to loosen splits.
- Autonomous visual evidence can optimize the frozen owner-style direction but cannot substitute for an external population study or calibrated stock truth.
- License, large downloads, paid GPU/cloud, external contact, push/merge and public release still require explicit approval.
- `U5.FC0` is complete. The next ready implementation leaf remains `U0.3`, beginning with the read-only/current-asset FilmCase eligibility, recoverable-lineage and group-split audit. No expert/router training is allowed before U0.3/U0.4/U4 and U5.FC1 pass.

---

## 2026-07-11 — U0.3 FilmCase lineage audit and isolated module

- **Parent/node:** `ULT > U0 > U0.3`, with stale reproduction correction `U0.5`
- **DRPT level/mode:** L2, Mode A; no shared legacy renderer, training path, manifest, or source asset was modified
- **Primary workflow:** `plan-tracker-discipline`
- **Secondary disciplines:** `codex-super-router`, `codex-super-aiml-harness`, `dev-research-reliability`, `drpt-bi-governance`, `project-agent-log-discipline`, `project-structure-steward`
- **Implementation boundary:** added only `src/filmcase/` (lineage contract), `scripts/audit_filmcase_manifest.py` (thin CLI) and dedicated tests. The legacy manifest is read-only; artifacts write only below ignored `outputs/filmcase/`.
- **Evidence:** audited all 4,210 local legacy rows. 2,320 rows match caption-only sidecars; 1,890 are unresolved; 0 have an auditable source/uploader/roll/scanner group. All 4,210 remain quarantined. Legacy SHA-256 has 0 exact duplicate groups; local dHash at distance <=4 found 140 near pairs. One over-limit raster is recorded as `perceptual_hash_failed`, never bypassed. With no eligible group split, 0 cross-split conflicts is a quarantine result, not a claim that the historical random split was safe.
- **Decision:** U0.3 is complete as a fail-closed classification/audit. The reference-derived FilmCase lane is blocked; only frozen-anchor/deterministic work may continue. U0.5 is complete because the reproduction manifest now states the existing legacy-manifest and deleted-full-source boundary.
- **Files changed:** `src/filmcase/__init__.py`, `src/filmcase/lineage.py`, `scripts/audit_filmcase_manifest.py`, `tests/test_filmcase_lineage.py`, `docs/FILMCASE_U03_LINEAGE_AUDIT.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/DATA_REPRODUCTION_MANIFEST.json`, `docs/data/DATA_LICENSE_BOUNDARIES.md`, `scripts/README.md`, this log.
- **Verification:** targeted tests, compile checks, two full local audits (final audit 4,210 rows), JSON parse and diff check; full project regression is required before commit.
- **Risk/handoff:** recovery needs durable per-row provenance and a new group-aware duplicate audit. Do not infer it, download new data, train an expert/router, publish data-derived assets, or weaken quarantine without a valid evidence source and applicable approval.

---

## 2026-07-11 — U0.4 local reproducibility baseline

- **Parent/node:** `ULT > U0 > U0.4`
- **Implementation:** added a checksum-pinned reproducibility configuration, a read-only verifier with optional ignored environment capture, focused tests, and a CPU-only GitHub Actions workflow. The existing benchmark/fixture registries remain explicitly non-gold until U4.1 freezes the severe-artifact corpus.
- **Evidence:** local verifier passed against the committed config hashes and wrote an ignored report for Python 3.12.10, Windows 11, Pillow 12.1.1, OmegaConf 2.3.0, pytest 9.0.3 and torch 2.11.0+cu128 at commit `87b5d69`.
- **Containment:** no renderer, training path, image, output (other than ignored environment report), data manifest, user file, remote run or dependency download was changed.
- **Handoff:** U0 reproducibility prerequisites are now complete except the owner-controlled license decision. U4.1/U4.2 are next safe evaluation leaves; no claim of zero severe artifacts is allowed yet.

---

## 2026-07-11 — U4.1 fail-closed evaluation foundation

- **Parent/node:** `ULT > U4 > U4.1` (in progress, not promoted)
- **Implementation boundary:** added the isolated `src/filmcase/evaluation.py` contract, a source-freeze CLI and unit tests. It reads only the pre-existing local union source index and writes its generated manifest only under ignored `outputs/filmcase/`.
- **Evidence:** 8 reproducibly hashed gold seeds and 32 stress rows froze from union-40. The seed covers seven required categories but has no face sample. Existing ID 11 remains included because it exposes the known red-highlight speckle issue.
- **Decision:** no candidate can receive a gold pass from this set. The contract requires `final` status, complete coverage, availability, one non-uncertain verdict for every gold sample, and zero severe verdicts; stress output is a Wilson interval only.
- **Handoff/risk:** acquire neither data nor votes automatically. Continue only with local deterministic/anchor work while a valid cleared/local source can close the category gap; U4.2 likewise cannot make a promotion claim before U4.1 finalizes.

---

## 2026-07-11 — U4.2 blind style/appeal protocol foundation

- **Parent/node:** `ULT > U4 > U4.2` (in progress, gated)
- **Implementation:** added isolated three-round candidate blinding and review aggregation. It requires separate severe, style and appeal fields; candidate names are excluded from the public sheet.
- **Pre-registered safety rule:** 2/3 severe votes veto; one severe, uncertain or missing record escalates to original-resolution adjudication; unresolved ambiguity cannot support promotion.
- **Evidence:** a dry run generated 24 blinded rows from eight provisional seed samples and the five owner anchors plus a bland control. It contains no review result and has no effect on U4.1 status.
- **Handoff:** normalize all anchor renders on the eventual final U4.1 source set and add matched nuisance controls before collecting autonomous vision reviews. No external validation, image acquisition, upload or generative method is authorized or required here.

---

## 2026-07-11 — Normalized owner-anchor color replay

- **Parent/node:** `ULT > U4 > U4.1/U4.2` (evidence preparation only)
- **Implementation:** added an isolated batch caller for the legacy deterministic color CLI. It rendered 54 outputs: five owner-anchor-inspired recipes plus safe-rich bland control over all nine frozen gold samples, with grain disabled and source/output hashes captured.
- **Recipe provenance:** values were recovered from the union-40 run IDs. The replay is explicitly not a pixel-identical historical anchor reproduction because original contact sheets mix source sets and some include grain.
- **Visual sanity evidence:** full-resolution face replay preserves geometry/content across all six candidates. The red bicycle/ColorChecker replay confirms the known red-highlight speckling remains visible in normalized 09/56/53, while bland safe-rich suppresses it with weaker style.
- **Gate:** these directed checks establish a usable failure gallery and normalized comparison inputs; blind three-pass adjudication remains required before any severe/style conclusion or promotion.

---

## 2026-07-11 — U4.1 face coverage from existing FilmSet input

- **Parent/node:** `ULT > U4 > U4.1` (still in progress)
- **Evidence:** scanned 5,285 local FilmSet input images using OpenCV only to locate candidates, then visually verified `DSCF01200 iso1600.png` as a child-face stress image. The local `kaggle_view.json` snapshot identifies FilmSet as MIT; its image and metadata hashes are recorded in the ignored frozen-set artifact.
- **Decision:** added it as `FS_FACE_01`, `research-only`, and explicitly `filmcase_reference_eligible=false`. It closes the face category gap for internal artifact evaluation without weakening U0.3 lineage quarantine or creating a film/authenticity claim.
- **Gate:** all required content categories are now present in the 9 gold / 32 stress seed, but U4.1 remains provisional until normalized candidate outputs have complete three-pass visual severe adjudications.

---

## 2026-07-11 — Executable anonymous review assets

- **Parent/node:** `ULT > U4 > U4.2` (evidence collection prepared, no rating asserted)
- **Implementation:** extended the blind-audit builder to consume the normalized replay manifest and materialize only anonymous review copies under `assets/round_N/sample_ID/A.png`. Candidate names remain absent from the public sheet and asset path; the mapping is separate and ignored.
- **Evidence:** generated 27 blinded rows and 162 anonymous assets for 9 samples × 6 candidates × 3 rounds. An initial PowerShell invocation coerced unquoted `01` to `1` and failed before a usable sheet; quoted arguments generated the final complete assets.
- **Handoff:** record raw review rows before unblinding, then apply the fail-closed aggregation rule. Asset preparation itself yields no score, safety pass or promotion.

---

## 2026-07-11 — Blind-audit aggregation boundary

- **Parent/node:** `ULT > U4 > U4.2` (tooling only)
- **Implementation:** added an ignored-report CLI that reads raw review JSONL and the separate private mapping, validates every record through the existing protocol, then writes a traceable aggregate. It cannot create a review value or infer a visual preference.
- **Verification:** parser rejects malformed/non-object records; all 37 local tests pass.
- **Handoff:** populate raw review JSONL only after actual anonymous visual passes. Any missing, duplicate, uncertain or contradictory review remains fail-closed under the protocol.

---

## 2026-07-11 — EXP-VIS-00 red-highlight counterfactual

- **Parent/node:** `ULT > U4 > EXP-VIS-00` (single-case evidence only)
- **Question:** can #56-like palette strength retain containment while removing the known red metal speckle?
- **Controlled result:** `gamut_mode=chroma` visibly removed the speckle but, without an output margin, created 26.06% new hard clipping. Adding only `output_margin=4` produced `[4,251]`, zero new clipping and visually smooth highlights on ID 11.
- **Trade-off:** the margin-bounded candidate reduces mean chroma substantially; it is eligible only for full nine-gold replay and blind style/severe review, not promotion.
- **Boundary:** no model was trained and no existing renderer code/profile was changed.

---

## 2026-07-11 — Full-gold margin-bounded chroma challenger replay

- **Parent/node:** `ULT > U4 > EXP-VIS-00/U4.2` (candidate evidence collection)
- **Implementation:** the anchor replay tool now accepts an opt-in, separately named `anchor56_chroma_margin4_challenger`; default historical-anchor runs remain unchanged. A relative-output-dir bug was caught after the first rendered file, fixed by resolving under repository root, and rerun with a complete manifest.
- **Evidence:** replayed all 9 gold inputs across 7 columns (63 render records) and materialized 189 anonymous assets for 3 rounds. The challenger remains unpromoted and is visually/safety unreviewed outside its ID 11 counterfactual.
- **Handoff:** execute blind reviews using only the new anonymous asset paths, then aggregate raw records before opening the private mapping. Any gold severe/uncertain outcome remains a hard stop.

---

## 2026-07-11 — U4.3 chroma-speckle diagnostic foundation

- **Parent/node:** `ULT > U4 > U4.3` (in progress; non-promotional)
- **Implementation:** added a separate full-resolution high-frequency chroma-island diagnostic and ignored JSON CLI, with synthetic-island and shape-validation tests.
- **Evidence:** on ID 11, #56 source reported 4,517 candidate pixels (0.314%) versus 3,206 (0.223%) for chroma+margin4. Island counts moved in the opposite direction, so no one-dimensional quality claim is allowed.
- **Decision:** diagnostics direct reviewers to potential speckle regions; they cannot veto, clear or rank any candidate without the visual protocol.

---

## 2026-07-11 — Anonymous contact-sheet review preparation

- **Parent/node:** `ULT > U4 > U4.2` (not a completed scorecard)
- **Implementation:** added a public-asset-only contact-sheet generator. It reads the anonymous review sheet and assets for one round, never the private mapping, and produced three local round sheets.
- **Visual preparation evidence:** all three anonymous round sheets were inspected. Eight non-red-highlight rows preserve geometry/text/texture and show only modest cross-column color/tone differences; ID 11 remains the discriminating red-speckle row.
- **Boundary:** no raw review JSONL has been written and no candidate was unblinded, scored or promoted. The sheets make later independent per-cell review practical; they do not substitute for it.
## 2026-07-11 - Start data-independent WorkingImage ingress

- Node/parent goal: `U1.1` under `ULT`; prioritize ready engineering that does not require new external data.
- Trigger: owner redirected work away from the provenance-blocked data lane.
- Skills used: `codex-super-router`, `codex-super-aiml-harness`, `dev-research-reliability`, `drpt-bi-governance`, `project-agent-log-discipline`.
- Decisions: stop the attempted Flickr provenance pilot before it produced data; keep FilmCase reference-derived work blocked; route `render_film` through `WorkingImage` while retaining an explicit, fail-closed legacy sRGB8 adapter until U1.3.
- Files changed: `src/preprocess/raster_decode.py`, `src/preprocess/__init__.py`, `scripts/render_film.py`, `tests/test_render_film_ingress.py`, and the U1 tracker row.
- Verification evidence: 6 targeted preprocessing/ingress tests passed under the existing local Python 3.9 environment; the Python 3.12 reproducibility verifier returned `BASELINE_OK`; `git diff --check` passed.
- Risks or unknowns: the renderer still becomes 8-bit at the named compatibility adapter; full pytest was not rerun because the checked-in `.venv` is Python 3.9 while the current baseline requires Python 3.12.
- Handoff state: U1.1 is in progress; next data-independent leaf is high-precision render/export integration and end-to-end raster/RAW fixtures.

## 2026-07-11 - Reframe publication research around FARO

- Node/parent goal: `ULT > U5.R0`; preserve the engineering renderer while reopening the assumed final research architecture.
- Trigger: owner explicitly requested unconstrained deep research, novelty pressure-testing, a publishable method and a separate best-engineering path.
- Skills used: `codex-super-router`, `codex-super-research-harness`, `scientific-research-harness`, `codex-super-aiml-harness`, `exa-search-preference` with native-search fallback, `plan-tracker-discipline`, `drpt-bi-governance`, `project-agent-log-discipline`.
- Evidence: parallel read-only reviews covered bounded colour operators, film/personalized/retrieval prior art and evaluation/risk. Direct novelty threats include CVPR 2016 content-aware style ranking, TSFlow, DiffRetouch, Emulating Emulsion, SA-LUT, InstantRetouch, PPSD, StatLUT, Learn then Test, two-stage risk control and the June 2026 joint risk–acceptance–utility certificate.
- Decision after two hostile-review cycles: FilmCase remains a baseline/ablation and ChromaticTail/FilmStyleSafe Paper A is prioritized. Complete fixed-`K` policy calibration is adopted existing reliability machinery, not a FARO theorem. FARO is a conditional CV system/application paper; a statistical method paper stays blocked until optional R0T/R4T proves a non-redundant clustered/multirater/correlated-policy result. Candidate-wise calibration, numerical operator constraints and preview monitoring do not imply semantic safety. Confirmatory gates cover overall and selected-output protocol-adjudicated risk, target-look adherence, style-qualified coverage and all-scene tie-score preference against an eligible global policy. Named-film calibration remains separate.
- Files changed: `docs/planning/FARO_RESEARCH_PROGRAM_2026.md`, `README.md`, planning index, `AGENTS.md`, `TASK_BOARD.md`, active tracker and this log.
- Verification evidence: three final hostile reviewers returned PASS after the second revision; DRPT plan lint and `git diff --check` passed; all 50 primary-source links returned HTTP 200 (two multi-megabyte PDFs timed out only while streaming after the 200 response). No model/data download, training, participant study or production change is part of this leaf.
- Handoff state: `U5.R1A` chromatic ontology, traceable look rubric, multirater rule, A0/A1+B0-B4 split and sampling/preregistration schema is the next primary research leaf; optional `U5.R0T` is the theorem-gap audit. Adaptive work is blocked until the eligible global frontier and B1 complete empirical-ceiling gate pass; confirmatory policy claims additionally require adopted complete-policy certification and participant approval.

## 2026-07-12 — Replace benchmark-first priority with Roll2Film colour transfer

- **Node/parent goal:** `ULT > U5.CT0`, with data-state propagation through `U0.5`; preserve the current deterministic product renderer while correcting the primary paper to an algorithm that directly performs colour transfer.
- **Trigger:** the owner rejected a standalone benchmark route and clarified that publicly downloadable online data counts as available, provided the bytes can actually be obtained rather than merely mentioned in a paper or placeholder page.
- **Skills used:** `codex-super-router`, `codex-super-research-harness`, `scientific-research-harness`, `codex-super-aiml-harness`, `exa-search-preference` with native-search fallback because Exa was unavailable, `plan-tracker-discipline`, `drpt-bi-governance`, `project-agent-log-discipline`, and `project-structure-steward`.
- **Execution mode:** Mode B for three disjoint read-only evidence leaves (local data eligibility, algorithm formulation, novelty threats); root remained the only writer and integration owner. Subagents changed no files. The integrated evidence bundle was reconciled against direct local counts, public APIs and primary papers.
- **Data evidence:** the current Mac has 3,896 JPEGs in `data/film_domain` (3,786 across eight named directories plus 110 generic), but the historical 4,210-row lineage audit remains 0 eligible/4,210 quarantined. The traceable Velvia lane has 28 manifest rows but 26 unique images, eight owners and 15 rows from one owner. The 100 IP2P pairs are synthetic WB/gamma inversions. The FiveK source/freeze, FilmSet and BlueNeg are absent locally.
- **Remote reachability evidence:** FilmSet API metadata reports 11,262,805,356 bytes and MIT; the official Kaggle CLI listed concrete members without credentials and actually downloaded `FilmSet/test/Cinema/DSCF7071.png` (SHA-256 `7c4b1b8f716448941135eef5dc7d4f9d0a0575d0a5d859ca33a7b3e1745fc9d2`) to temporary storage. BlueNeg exposes 491 metadata rows, 53 roll IDs and 13 film-type strings; its approximately 688MB preview and 268MB pseudo-GT lanes were enumerated and a 1,024-byte range read succeeded. DigitalFilm_dataset exposes about 3.35GB and also passed a ZIP range read, but remains quarantined because its Internet-collected samples lack per-image rights/roll/scanner lineage. PhotoGAN C200, SillyStill pairs and Emulating Emulsion data are not counted as available because no usable first-party corpus endpoint was verified.
- **Scientific decision:** the primary hypothesis is **Roll2Film**—jointly identify and apply a shared explicit roll-look colour operator from multiple content-diverse, unpaired frames grouped by physical roll. Roll grouping is the provisional novelty; LUTs, flow/OT, canonical pivots, semantic pseudo-pairs, constraints, best-of-K and benchmarks are occupied components/baselines. FilmSet is a paired-blind Level-A film-recipe experiment; BlueNeg is a Level-B grouped-real-film mechanism pilot; named-stock calibration remains Level C and requires controlled paired rolls/process/scan sessions. ChromaticTail/FilmStyleSafe becomes supporting evaluation and FARO a product/system wrapper. Algorithm failure closes the paper route rather than promoting a benchmark fallback.
- **Files changed:** `docs/planning/ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`, `docs/planning/FARO_RESEARCH_PROGRAM_2026.md`, `docs/planning/ULTIMATE_ROADMAP_2026.md`, `docs/planning/README.md`, `AGENTS.md`, `README.md`, `TASK_BOARD.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`, `docs/DATA_REPRODUCTION_MANIFEST.json`, and this log.
- **Verification evidence:** both the active tracker and new research plan pass the DRPT linter; the reproduction manifest passes `jq`; `git diff --check` passes; all 16 unique external links in the new plan returned HTTP 200 or 206; concrete FilmSet/BlueNeg/DigitalFilm byte-access probes succeeded. This is a documentation/research leaf, so model tests and training were not run.
- **Non-actions and risks:** no full dataset/model download, training, participant study, external contact, cloud/GPU work or push occurred. The existing user deletion of `data/raw/.gitkeep` remains untouched and unstaged. No `first` claim is made; a dedicated grouped-observation/system-identification audit and CT3 group-size falsification remain mandatory. FilmSet is recipe simulation, BlueNeg is restoration/archive data, and neither proves named-stock fidelity.
- **Handoff state:** `U5.CT1` explicit invertible-operator contract/known-operator simulator and `U5.CT2` metadata-only grouping/pair-blinding contracts are the next ready no-data leaves. Full FilmSet and BlueNeg lane downloads remain gated until CT3 passes and the corresponding acquisition is explicitly approved.

## 2026-07-13 - Execute CT1/E0 affine identifiability foundation

- Node/parent goal: `ULT > U5.CT1/U5.CT3`; run the cheapest known-truth falsification before any data acquisition or learned model.
- Skills used: `dev-research-reliability` as the only write workflow; `codex-super-router`, `codex-super-aiml-harness`, `plan-tracker-discipline`, `drpt-bi-governance`, `project-agent-log-discipline` and `project-structure-steward` as read-only governance/review layers.
- Decisions: isolate new work under `src/roll2film`; begin with an orientation-preserving affine v1 operator and SPD Gaussian-transport estimator that sees only an independent neutral prior plus unordered target frames. Keep CT1/CT3 open because curve/LUT, coverage, strong nuisance, flexible-nuisance and novelty gates are not yet tested.
- Files changed: `src/roll2film/`, `scripts/run_roll2film_e0.py`, `configs/roll2film_e0.json`, focused tests, script index, active tracker/task board, current-truth corrections in `AGENTS.md`, and `docs/ROLL2FILM_CT1_E0_FOUNDATION.md`.
- Verification evidence: 48-replicate E0 uses group sizes 1/2/4/8/16/32. Primary correct-group holdout RGB RMSE falls from 0.017172 to 0.004224 (75.4%); the 32-frame mixed-operator control remains 0.064019. Mild exposure/scene/noise stress falls from 0.021609 to 0.005464 versus 0.070120 mixed. Focused tests pass and the raw ignored report records config hash/environment.
- Risks or unknowns: the pass is conditional on a known SPD affine truth and matched neutral distribution. It is not real-film, named-stock, general unpaired-transfer or product evidence; it does not authorize FilmSet/BlueNeg downloads.
- Handoff state: extend CT1/E0 to monotone curves/smooth LUT residuals, support breadth and hostile nuisance absorption. Only a complete CT3 pass can reopen the acquisition gate.

## 2026-07-15 - Package visual evidence for external deep research

- Node/parent goal: `ULT > U5.CT` research handoff support; make the external review prompt evidence-bearing instead of referring to opaque historical scheme numbers.
- Skills used: `dev-research-reliability` as the write workflow, with plan/DRPT/log/structure disciplines as governance reviewers.
- Decisions: package the 3900x9480 historical 56-scheme sheet and mapping, all 54 same-input normalized anchor renders, the four-file ID 11 red-highlight counterfactual, CT1/E0 raw evidence and current project authority. The prompt states that 02/03/33 are smoke cues and requires the reviewer to stop visual inference if attachments cannot be read.
- Current-data correction: live Windows inspection found the complete FilmSet tree locally (21,140 files / 11,262,805,356 bytes), with 4,657 files per train branch and 628 per test branch. This conflicts with older active text saying 638 test images and is surfaced as an explicit primary-source audit question. The 903-file FiveK freeze pack is also present, while BlueNeg remains absent.
- Files changed: `docs/planning/ROLL2FILM_WEB_DEEP_RESEARCH_PROMPT_CN.md`, planning index, package/counterfactual readmes and this log. Generated ignored artifact: `outputs/research_handoff/ROLL2FILM_RESEARCH_HANDOFF_20260715.zip` plus its unpacked directory and checksum manifest.
- Verification evidence: package contains 73 files / 119,556,049 unpacked bytes; ZIP is 119,258,248 bytes with SHA-256 `A452473C69FFCA9BE5D0528FCCCC9D04CBAC5D6822BA8D53500451E7FE34C88D`. `CHECKSUMS.csv` records every non-manifest file.
- Risks or unknowns: uploading the ZIP to an external web researcher is a user action; archive readability and source licenses do not imply public-release clearance. FilmSet is recipe data, not physical-film truth.
- Handoff state: upload the ZIP and `PROMPT_CN.md` together. The reviewer must inspect attachments, resolve the FilmSet 628/638 discrepancy and return an executable evidence-gated plan.

## 2026-07-15 - Preserve and integrate external Roll2Film ultimate audit

- **Node/parent goal:** `ULT > U5.CT0/U5.CT2/U5.CT3/U5.CT4`; preserve the returned expert response and propagate evidence that changes the current experimental order.
- **Skills used:** `dev-research-reliability` as the only writer; `codex-super-research-harness`, DRPT, plan/log and project-structure disciplines as read-only reviewers; `exa-search-preference` with native-web fallback because Exa was not available in the tool surface.
- **Source preservation:** copied the supplied Chinese Markdown byte-for-byte to `docs/reference/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_20260715_CN.md`; 87,776 bytes; SHA-256 `C051CF747A00EB98F50B5654EEE4CEE6DBABAC5834378DB49A2525ABB292E27F`.
- **Local evidence:** decompressed FilmSet contains 21,140 images / 11,262,805,356 bytes, with 4,657 files in each train domain and 628 in each test domain. The enclosing directory also contains the original ZIP, Kaggle metadata and a sample, so its aggregate size must not be reported as the image-tree size. `data/processed/manifest.jsonl` exists; current Windows `film_domain` contains 4,212 JPEGs, without retroactively granting eligibility beyond the earlier 4,210-row fail-closed audit.
- **External verification:** the ICLR 2024 identifiable-UDT source requires paired cross-domain conditional distributions and does not directly identify target-only roll groups; FilmSet primary publication confirms the dataset scope while the local archive resolves runtime count to 628; BlueNeg primary paper/tree/licence confirm restoration/archive use, 290GB public tree and attribution terms; Emulating Emulsion occupies paired controlled compact explicit film modelling.
- **Scientific decision:** reclassify current variable-sample E0 as a restricted estimator/misspecification unit test, not evidence that group labels add information. Product work becomes independently winnable by a fixed explicit champion/bank. Roll2Film remains a challenger until fixed-total-sample, nuisance, prior and real-group controls pass. FilmCase requires strength-adjusted diversity and a legal-supervision Oracle gap; otherwise use champion plus bounded strength. Start support adaptation as an image-level family-valid strength path, not per-cell RGB blending.
- **Visual evidence boundary:** the external audit’s 53/55/56 near-collinearity and ID 11 red-speckle verdict for 09/53/55/56 enter the diversity and worst-case queue, but remain external autonomous evidence until internal frozen blind adjudication. Draft numeric gates are not preregistered results.
- **Files changed:** preserved raw response, added `docs/planning/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_INTEGRATION_20260715.md`, and corrected `AGENTS.md`, `TASK_BOARD.md`, the active tracker and this log.
- **Handoff state:** next ready leaves are local FilmSet evidence/pair-blind freeze, fixed-budget E0, L0–L2/high-precision colour-state work, then diversity/Oracle and image-level support pilots. Official 628 stays closed until full policy freeze; BlueNeg images, external contact, public release and costly GPU work still require approval.

## 2026-07-15 - Run Roll2Film E0 v2 fixed-budget affine controls

- **Node/parent goal:** `ULT > U5.CT3`; remove the sample-count confound before any data-bearing or learned-model claim.
- **Skills used:** `dev-research-reliability` as the sole writer; router, AI/ML, research, general, plan-tracker, DRPT-BI, project-log and structure skills as read-only governance/review layers.
- **Implementation:** added an isolated fixed-budget runner/config, affine composition and simulator partition/boundary utilities, deterministic paired-bootstrap evaluation and focused tests. The v1 config, script and raw result contract remain runnable and unchanged.
- **Evidence:** 64 replicates at exactly 4,096 target pixels for every `1/2/4/8/16/32` frame arm. Exact-pool partition parameter difference is `0.0`. At 32 frames, true nuisance boundaries improve over random boundaries by 46.9%, independent support over repeated support by 78.9%, and correct single-operator groups over mixed groups by 94.5%; every paired bootstrap lower bound exceeds zero.
- **Negative evidence:** shifted source prior increases error from `0.003716` to `0.028871`; a film+scanner composite is recovered much closer to the composite (`0.003990`) than to the film component (`0.019971`). These confirm prior and scanner non-identifiability and are excluded from the pass gate.
- **Decision:** `method_control_decision=pass`; `roll_information_decision=not_established`. The run validates affine mechanics, true frame-boundary nuisance information and independent support, not physical-roll labels, nonlinear film style or stock truth.
- **Files changed:** `src/roll2film/`, focused tests, `configs/roll2film_e0_fixed_budget.json`, `scripts/run_roll2film_e0_fixed_budget.py`, script index, evidence docs, active tracker/task board/current truth and this log. Ignored raw report: `outputs/roll2film/e0_fixed_budget/report.json`.
- **Verification:** 10 focused Roll2Film tests pass; runner exits 0 with the decisions above; config SHA-256 `c90a63f9802df7e5f6c80496f671e3fb32bafa882317cb49480f6d2676c48c06`.
- **Handoff state:** freeze local FilmSet manifest/pair-blind access contract and implement L2 truth/stronger nuisance controls. BlueNeg pixels are not needed yet; official FilmSet 628 remains closed.

## 2026-07-15 - Freeze FilmSet manifest and lockbox implementation contract

- **Changed:** moved `U5.CT2/U5.CT4` to in progress and froze `docs/data/ROLL2FILM_FILMSET_MANIFEST_CONTRACT.md`: four isolated ignored manifests, cluster-before-split pair blindness, resumable integrity cache, test-payload no-decode rule, deterministic report hashes and fail-closed training/lockbox roles.
- **Evidence/propagation:** current FilmSet counts/bytes and 628/638 policy remain unchanged; active tracker, task board and `AGENTS.md` now record the owner’s 2026-07-15 pre-authorization for necessary downloads while retaining scientific/licence/storage gates. BlueNeg full 290GB remains unjustified.
- **Next:** implement the isolated `src/roll2film` manifest module, CLI and tests, then run the 11.26GB local evidence freeze. No source image or legacy manifest may be modified.

## 2026-07-15 - Pass FilmSet pair-blind evidence freeze

- **Node/parent goal:** `ULT > U5.CT2/U5.CT4`; implement and execute the frozen FilmSet data/access contract without exposing aligned training identities or final-test pixels.
- **Skills used:** `dev-research-reliability` as sole writer; router, AI/ML, research, general, plan, DRPT, log and structure skills as read-only reviewers.
- **Implementation:** added isolated `src/roll2film/manifests.py`, a resumable audit CLI and six focused tests. Official test payloads are byte-hashed only; the training view accepts source/target roles and rejects both evaluator lockboxes.
- **Evidence:** all 21,140 files / 11,262,805,356 bytes hashed; source tree SHA-256 `e47c254ce550099a52518f6863fa28a1f9077e889593c0a1b1cd690cda8f1619`. Frozen pools contain 2,096 source, 2,096 target, 465 internal-dev and 628 final identities. All source/target/dev content and duplicate-cluster intersections are zero; exact/dHash/embedding cross-pool leakage is zero; final decoded payloads are zero.
- **Failure shield:** the first live run correctly exposed 80 official train/test basename reuses, invalidating basename-only global identity. IDs were split-namespaced and exact train/test payload overlap became the hard gate. A later audit found Windows CRLF translation made claimed manifest hashes differ from physical files; atomic byte writes plus a physical-hash regression test fixed it before promotion.
- **Verification:** 58 tests pass; compileall and `git diff --check` pass. A cache-backed rerun produced byte-identical manifests/report, and every declared manifest SHA-256 equals the physical file hash. Evidence: `docs/ROLL2FILM_FILMSET_EVIDENCE_FREEZE.md`.
- **Decision/propagation:** `U5.CT4=complete`; the FilmSet portion of CT2 is complete. FilmSet remains internal recipe research, not physical-film truth or release-cleared content. The official 628 stays sealed; CT5 now waits on nonlinear CT3 and a matched-strength baseline freeze rather than data preparation.
- **Handoff state:** implement L2 truth and stronger nuisance/prior/scanner controls, then freeze matched-strength deterministic/classical baselines. BlueNeg remains metadata-only until the CT3/whole-roll gate passes.

## 2026-07-15 - Pass fixed-budget L2 nonlinear method controls

- **Node/parent goal:** `ULT > U5.CT1/U5.CT3`; test nonlinear explicit-operator mechanics before any real-roll or learned-model promotion.
- **Implementation:** added an affine-plus-three-channel monotone rational-quadratic L2 operator, analytic inverse/positive Jacobian diagnostics, deterministic serialization, dense 3D-LUT baking, an alternating unpaired Gaussian-transport/quantile-spline estimator, WB nuisance simulation and an isolated fixed-budget runner/config.
- **Evidence:** 32 replicates at exactly 4,096 target pixels. At 32 frames, true nuisance boundaries improve over random boundaries by 54.0%, independent support over repeated copies by 77.1%, and correct single-operator groups over mixed groups by 93.3%; all paired bootstrap lower bounds exceed zero. Partition grid difference is `0.0`; truth Jacobian minimum is `0.776540`; 33-cube/65-cube maximum errors are `0.0006221/0.0001636`.
- **Negative evidence:** prior swap raises error from `0.005412` to `0.031684`; the estimator recovers an input-side scanner/profile composite closer than its film-only component; unnecessary flexible frame normalization degrades a clean roll. None is used to pass the gate.
- **Verification:** 64 tests pass, compileall and `git diff --check` pass, and the ignored report reruns byte-identically with SHA-256 `dc85c9e035d644f93bbeddbd444393cfadb7b902220d6e19e9cb19086826c04c`. Evidence: `docs/ROLL2FILM_E0_L2_FIXED_BUDGET_V3.md`.
- **Decision/propagation:** `method_control=pass`, `roll_information=not_established`. CT1 retains explicit L0/gauge/shaper work; CT3 retains matched real-roll controls. CT5 baseline/policy freeze is the next data-bearing leaf; final FilmSet 628 remains sealed.

## 2026-07-15 - Freeze CT5 baseline and internal evaluator policy

- **Node/parent goal:** `ULT > U5.CT5`; prevent hidden-target tuning and saturation-only wins before any FilmSet internal-dev payload is decoded.
- **Decision:** froze a CPU-safe v1 ladder from identity/basic adjustments through Lab statistics, quantile, Gaussian/Bures, sliced OT and pooled L2, plus a non-deployable paired oracle. Learned/adaptive LUT challengers are recorded `not-run` until a new implementation/compute freeze.
- **Access/evaluation contract:** training uses only disjoint source/target manifests; 465 internal-dev clusters are deterministically split pilot/confirmatory; final 628 is forbidden. Style strength, hidden recipe fidelity and severe artifacts remain independent scorecards. Comparisons use frozen strength strata rather than post-hoc RGB weakening.
- **Files:** `configs/roll2film_ct5_baselines.json`, `docs/data/ROLL2FILM_CT5_BASELINE_EVALUATOR_CONTRACT.md`, active tracker/task board and this log.
- **Handoff:** implement hash/role-enforcing loaders, deterministic equal-image pixel caches and the basic/Gaussian/quantile/L2 baselines. Commit the implementation before the evaluator opens internal-dev target payloads.

## 2026-07-15 - Run CT5 internal FilmSet pilot and freeze confirmatory adversaries

- **Node/parent goal:** `ULT > U5.CT5`; use only the 227-identity pilot fold to freeze strength strata and best-basic comparators.
- **Data/access evidence:** hash-verified equal-image caches contain 2,096 source and 2,096 target identities per recipe, plus 227 pilot/238 untouched confirmatory identities. ICC decode is linear-sRGB. The pilot runner loads only pilot files; final 628 remains unparsed/undecoded.
- **Failure shield:** the first pilot attempt stopped on repeated 8-bit Velvia tail quantiles violating strict spline knots. A fixed epsilon strictification and quantized-tail regression test were committed before rerun; no partial report or confirmatory data was used.
- **Pilot result:** Lab mean/std leads Cinema (`Delta-E=1.730`), pooled L2 leads ClassNeg (`2.106`) and per-channel quantile leads Velvia (`3.121`). Pooled L2 style strength is `3.686/4.912/4.603` across the three recipes, but raw out-of-range risk is `0.515%/1.808%/1.003%`; no product promotion is made.
- **Freeze:** low/moderate/strong style strata are `[0,1.5)`, `[1.5,2.75)` and `[2.75,infinity)`. `configs/roll2film_ct5_pilot_decision.json` freezes the best-basic candidate per domain/stratum and the cluster-bootstrap confirmatory endpoint.
- **Verification:** 74 tests pass; the complete pilot report reruns byte-identically with SHA-256 `145a3bca313212d625a0245b6366d77ceb5732e883d8f87255bab53aeaa434fa`. Evidence: `docs/ROLL2FILM_CT5_PILOT_RESULTS.md`.
- **Handoff:** commit this freeze, then run the 238-identity confirmatory fold without changing parameters, strata or adversaries. Full-resolution severe adjudication remains required; official 628 stays sealed.

## 2026-07-15 - Complete CT5 sampled confirmatory evaluation

- **Node/parent goal:** `ULT > U5.CT5`; test frozen pilot operators on 238 untouched identities against fixed strength-stratum best-basic adversaries.
- **Method:** no refitting; duplicate-cluster bootstrap with 5,000 resamples; target Delta-E fidelity and input Delta-E style floor are separate gates. Spatial SSIM and visual severe remain not-run.
- **Results:** Cinema passes only Lab mean/std. ClassNeg passes pooled L2, sliced OT and Bures, with pooled L2 strongest (`Delta-E=2.321`, improvement CI `[2.709,2.928]`). Velvia passes pooled L2, sliced OT and moderate-stratum Lab; pooled L2 is the strongest eligible style (`4.407`) with target Delta-E `3.456`.
- **Negative evidence:** no single candidate passes all domains; pooled L2 loses decisively to strong basic on Cinema. Raw OOR is nonzero for pooled L2 (`0.743%/2.286%/1.244%`) and much larger for some Bures/sliced arms, so sampled statistical success is not product safety.
- **Decision:** freeze Lab for Cinema and pooled L2 for ClassNeg/Velvia as primary full-resolution finalists, with simpler/risk comparators. This supports a fixed per-recipe deterministic bank, not per-photo ML routing or a universal Roll2Film operator.
- **Verification:** 75 tests passed before execution; confirmatory report reruns byte-identically with SHA-256 `51e16eb047296e93b3bc7c739adf5a499a7f80f3d48bf95a228db059614d5fe3`; final 628 remained unparsed/undecoded. Evidence: `docs/ROLL2FILM_CT5_CONFIRMATORY_RESULTS.md`.
- **Handoff:** commit the confirmatory decision, render all confirmatory identities at full resolution for automatic metrics, save deterministic worst cases/contact sheets and perform the severe visual veto. Do not open final 628.

## 2026-07-15 - Pass CT5 full-resolution severe visual adjudication

- **Node/parent goal:** `ULT > U5.CT5`; adjudicate every frozen finalist at original resolution before any final-628 or product promotion.
- **Implementation:** added a confirmatory-only targeted audit mode and separated out-of-range frequency from excursion magnitude and newly clipped display pixels versus source/target. The targeted mode rejects IDs outside the frozen 238-identity confirmatory membership and never overwrites the main report.
- **Evidence:** all three recipes and all frozen candidates were evaluated on 238 full-resolution identities. The report SHA-256 is `edc761d25cc87b7321acbb28a744ddb9c0af1800a7e6ce588bcc7d45e2a037d6` at software commit `8bf1f5725fbbd91debfed72f175cbac45b8abd50`; final-628 remained unparsed/undecoded.
- **Result:** Cinema Lab mean/std reaches target/style Delta-E `1.877/3.461`; ClassNeg pooled L2 `2.322/4.841`; Velvia pooled L2 `3.462/4.413`. ClassNeg's apparent 99.4% worst-image out-of-range rate has maximum magnitude only `0.002145`. Velvia p95/max image excursions are `0.06170/0.08901`; original-resolution butterfly, aircraft, neon, night-light and other worst cases show strong coherent colour without confirmed severe clipping, posterization, seams, colour blocks, speckle, text/geometry failure or detail rewrite.
- **Decision:** `U5.CT5=complete on internal confirmatory`. Preserve the fixed per-recipe bank and do not add an arbitrary blend, learned gamut module or per-photo ML router. Production still requires a versioned explicit output transform; gamut protection reopens only on confirmed final/product severe evidence. This is recipe-transfer evidence, not physical-film, named-stock, population-preference or release clearance.
- **Verification:** 77 tests and `git diff --check` pass. Frozen policy and evidence: `configs/roll2film_ct5_fullres_decision.json` and `docs/ROLL2FILM_CT5_FULLRES_VISUAL_AUDIT.md`.
- **Change propagation/handoff:** CT5 unblocks the FilmSet side of the method tree; CT2/CT3/CT6 now own the real-roll information question. BlueNeg metadata, licence and whole-roll split must be frozen before its pre-authorized preview/pseudo-GT download and correct-roll matched-control pilot. Final-628 remains sealed until CT6/CT7 and the complete policy are frozen.

## 2026-07-15 - Freeze BlueNeg metadata and bounded acquisition

- **Node/parent goal:** `ULT > U5.CT2/U5.CT6`; close licence, remote-identity, whole-roll leakage, control-eligibility and storage gates before downloading image payloads.
- **Source freeze:** official Hugging Face dataset revision `b038a1ae68f42067ff12b5e79ddbe62919b7af23`; 491 metadata rows, 53 rolls, 13 film strings; licence credit `Copyrighted by Tien-Tsin Wong`. Full two-lane public inventory is 738 files / 955,748,961 bytes, while the full repository is about 290GB.
- **Failure shield:** the first preflight failed because `meta.json` lists pseudo-GT-looking paths for blue-intact frames that do not exist remotely. Eligibility now requires the path in both metadata and the exact-revision inventory; a regression test covers this discrepancy.
- **Whole-roll result:** 17 rolls containing official test frames are sealed in full. Of 36 remaining rolls, only five have at least four actually published pseudo-GT frames. Only four `Kodak Gold 100-5` rolls have same-film wrong-roll controls: two development and two confirmatory. One `Kodak Gold 100` roll is exploratory only.
- **Decision:** CT2 metadata gate passes, but CT6 is narrowed to a single-film-string mechanism pilot. The exact manifest contains 101 files / 118,929,719 bytes. A pass can support only archive/scanner-specific roll-look information; it cannot support digital-to-film, stock calibration or broad multi-stock claims.
- **Evidence/verification:** the alignment dependency was subsequently added with a restricted NumPy-only unpickler. The fixed file contains 428 matrix/bbox records, including five bboxes extending slightly outside preview bounds. The refreshed metadata report SHA-256 is `2703d07036f7bdc5ac296df7bfa9617e294562dacd2b75bb27623dc037b50232`; acquisition manifest remains `c221016837284c0a8110909852480a118db29ebd9a2e44a59c9dc6ac11798067`; 86 tests pass, whole-roll overlap is zero and no pixels were decoded. Evidence: `docs/ROLL2FILM_BLUENEG_METADATA_FREEZE.md`.
- **Handoff:** implement a resumable exact-path downloader with byte/LFS-SHA verification, acquire only the frozen 101 files under the owner's standing authorization, then freeze fixed-budget correct/wrong-roll evaluator policy before decoding pixels.

## 2026-07-15 - Complete bounded BlueNeg pilot acquisition

- **Node/parent goal:** `ULT > U5.CT6`; acquire only the metadata-approved payloads without decoding them or expanding scope.
- **Implementation:** added an exact-path resumable downloader with repository/revision/report hash gates, path traversal and lane-boundary checks, concurrent cache-safe fetch, byte/LFS-SHA verification, manifest-external file rejection and deterministic reporting. Three focused tests cover corruption repair, extra-file rejection and traversal rejection.
- **Evidence:** all 101 files / 118,929,719 bytes verify against the exact-revision LFS hashes; zero manifest-external lane files; `image_payloads_decoded=false`. After adding the required alignment snapshot, the cached rerun remains byte-identical. Refreshed download report SHA-256: `f77f7f7e2699cc3cc12e73267adfc6a77da44150ad7bebcc90f3fae194abb764`; software commit `c0f6647346ccc055920242d615d0e35ffd052271`.
- **Verification:** 86 tests pass; source bytes remain ignored and the full 955MB lanes/290GB archive remain absent. Evidence: `docs/ROLL2FILM_BLUENEG_ACQUISITION_RESULTS.md`.
- **Handoff:** freeze the fixed-budget CT6 evaluator and stop/ambiguity rules before first pixel decode. The matched confirmatory core remains only two held-out Kodak Gold 100-5 rolls against two development same-film wrong-roll controls.

## 2026-07-15 - Freeze CT6 BlueNeg evaluator before pixel decode

- **Node/parent goal:** `ULT > U5.CT6`; prevent family, budget, control or threshold tuning on the two held-out physical rolls.
- **Policy:** exactly three unpaired support frames and 4,096 pixels/frame/domain per roll; three development or six confirmatory hidden queries at 16,384 aligned pixels/frame. ICC-aware linear-sRGB decode and bbox-intersection alignment are fixed; resize and negative indexing are forbidden.
- **Development gate:** compare identity, basic WB/contrast/saturation, Lab, quantile, Bures affine and pooled L2 on only the two development rolls. Freeze the simplest family within 0.25 Delta-E00 of best that beats identity on both rolls and keeps median style Delta-E at least 0.5. Stop without confirmatory decode if none passes.
- **Confirmatory gate:** correct-roll inference must beat identity, pooled development, each development wrong roll, the other confirmatory roll and deterministic shuffled groups at equal budget. Both rolls must be positive and the 10,000-resample query-frame-cluster bootstrap lower bound versus best control must exceed zero; one-roll or nuisance-dependent gains are ambiguous.
- **Claim ceiling:** at most narrow archive/scanner-specific group-information evidence in four Kodak Gold 100-5 rolls. Even a pass cannot establish multi-stock Roll2Film, digital-to-film or calibration.
- **Evidence/handoff:** `configs/roll2film_blueneg_evaluator.json` and `docs/data/ROLL2FILM_BLUENEG_EVALUATOR_CONTRACT.md`. Implement alignment/cache/development runner, test it, commit it, then decode development pixels only.

## 2026-07-15 - Select BlueNeg operator family on development rolls only

- **Node/parent goal:** `ULT > U5.CT6`; select one explicit family without decoding held-out roll pixels.
- **Implementation/access:** ICC-aware linear-sRGB decoder, strict development-only loader, official-bbox intersection cropping, independent support source/target sampling and aligned query sampling. Confirmatory role access fails before file decode. Four focused tests cover negative bbox intersection, shape mismatch, role rejection and deterministic sampling.
- **Result:** Lab mean/std is the development winner (`Delta-E=5.576`, style `5.184`) and beats identity (`8.289`) on both rolls. Basic/Bures score about `6.26`; pooled L2 is stronger (`style=7.544`) but worse overall (`6.962`) and unstable (`5.297` versus `8.627`, with 8.72% OOR on the second roll).
- **Decision:** freeze Lab mean/std for all correct/wrong/pooled/shuffled confirmatory arms. Do not reopen L2 or tune strength after held-out decode. Report SHA-256 `d66cfbae2777e2522b8587164c21af46c916e5203ec10e723dc75a24fbf3a897` reruns byte-identically; `confirmatory_roll_pixels_decoded=false`.
- **Nuisance risk:** both development rolls are Stanford/date coupled; held-out support/query locations overlap within rolls. Query-location strata are mandatory, and location-dependent gain is ambiguous rather than a pass.
- **Evidence/handoff:** `configs/roll2film_blueneg_development_decision.json`, `docs/ROLL2FILM_BLUENEG_DEVELOPMENT_RESULTS.md`. Implement the fixed Lab confirmatory controls, test/commit, then decode the two held-out rolls once.

## 2026-07-15 - Conclude BlueNeg roll-information test as ambiguous

- **Node/parent goal:** `ULT > U5.CT6`; one-shot evaluation of the frozen Lab family on two held-out physical rolls against each roll's strongest fixed control.
- **Result:** correct-roll aggregate Delta-E is `5.2008` versus composite own-best-control `5.2455`, but physical-roll gains have opposite signs: `19960817H=-0.2201`, `19970620C=+0.3095`. The cluster-equal gain is only `+0.0447`, with 95% interval `[-0.2201,+0.3095]`. Seen/unseen-location aggregate gains are both about `+0.045` but mix the negative and positive rolls and cannot rescue replication.
- **Decision:** `ambiguous_roll_information`; `roll_information_established=false`. CT7 amortized set inference is stopped. No full-resolution promotion audit is run because the statistical gate failed first; sampled Lab OOR is zero but safety cannot convert ambiguity into success. Reopen only with several additional independent matched-control rolls across dates/locations and a new preregistration.
- **Reproducibility:** 92 tests pass; confirmatory report reruns byte-identically with SHA-256 `4c30a9887069e4921233e9b4b4e7ecbce3e3e8a84e4aa4c74708c856be4f6af8`; FilmSet final-628 remains sealed. Evidence: `configs/roll2film_blueneg_confirmatory_decision.json`, `docs/ROLL2FILM_BLUENEG_CONFIRMATORY_RESULTS.md`.
- **Change propagation/handoff:** the special physical-roll paper claim closes under current data; CT5 fixed deterministic recipe-bank/product evidence remains valid. CT8 may now freeze and execute the one-shot final-628 recipe/style/artifact confirmation without human, physical-film or stock-calibration claims.

## 2026-07-15 - Freeze one-shot CT8 final-628 execution

- **Node/parent goal:** `ULT > U5.CT8`; prevent any family, strength, metric, comparator or worst-case-selection tuning on the final FilmSet lockbox.
- **Frozen bank:** Cinema Lab mean/std, ClassNeg pooled L2 and Velvia pooled L2, each against its frozen WB/contrast/saturation best-basic opponent. Operators come only from the CT5 pilot bundle; final refit and spatial resampling are forbidden.
- **Gates:** 10,000 identity-cluster bootstrap resamples require a fidelity-improvement lower bound above zero; primary style must be at least best-basic style. Severe review saves composite worst cases and top-five axis extremes for raw excursion, target-relative new clipping, target Delta-E, red/cyan occupancy and speckle, with original-resolution adjudication.
- **Access/claims:** verify all 2,512 manifest payload hashes before decode, require 628 complete identities and open once only. No human preference, physical-film, named-stock, calibration, release-rights or roll-information claim.
- **Evidence/handoff:** `configs/roll2film_ct8_final_policy.json`, `docs/data/ROLL2FILM_CT8_FINAL_628_EXECUTION_CONTRACT.md`. Implement the access-locked full-resolution runner, test/commit it, then decode final exactly once.

## 2026-07-15 - Reopen Ultimate as a real-film-first program

- **Node/parent goal:** `ULT > RF`; user-authoritative success correction supersedes FilmSet-centred priority. Ultimate must learn from verifiable physical-film scans and generalise across unseen real rolls/sources under nuisance, style and severe-artifact controls.
- **Evidence split:** FilmSet CT5/CT8 is a Capture One digital-recipe control only. CT8 completed once: Cinema and ClassNeg pass; Velvia is stronger but loses fidelity. BlueNeg is real archive film but its two held-out rolls reverse sign, so roll information is not established.
- **Research/data audit:** FILM-R v2 is the nearest bounded rights-clear source: 44 authentic 4K 35mm scans plus 44 expert restorations, CC BY 4.0, 437,570,872 bytes. Apollo raw scans have physical magazine/stock metadata but are about 1.2 GB/image; DOCUMERICA is diverse public-domain film but lacks roll/scanner controls; SillyStill full 41-pair data is unavailable and unlicensed at its official repo; Emulating Emulsion data is not publicly verified.
- **Decision:** Roll2Film becomes one challenger among global explicit experts, hierarchical mixed effects, similar-case retrieval, bounded conditional LUT/curves and optional bounded local grids. GPU is allowed only for finite parameter prediction after real-film gates; no generative RGB.
- **Propagation:** updated `AGENTS.md`, tracker, task board and Roll2Film priority banner; added `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md` and CT8 auxiliary closure. Nearest ready leaf is RF0.1 FILM-R acquisition and exact manifesting.

## 2026-07-15 - Complete FILM-R acquisition and fail-closed integrity gate

- **Node/parent goal:** `ULT > RF0.1/RF0.2`; acquire the nearest bounded rights-clear real-film source, then determine its honest use boundary before any learning experiment.
- **Acquisition evidence:** downloaded and verified all 88 Figshare v2 files (44 authentic damaged scans plus 44 restorations), exactly 437,570,872 bytes. Every supplied MD5 and every recorded local SHA-256 passes. Acquisition report SHA-256 is `07e3a26daec77d754203bff05376ff485b65d09821912a443f8f0622c284b79c` and pair-manifest SHA-256 is `a9075014e4fe6063c0137dc4de6af5b8222c54480a237a513f42027ebcf5b6ba`.
- **Integrity/visual evidence:** all 88 JPEGs decode, pair dimensions match, and no exact or dHash<=4 cross-pair duplicate exists. All 88 lack ICC and EXIF. Both review sheets and largest-change pairs were visually inspected: restoration preserves subject and geometry but may remove dense damage across most of a frame. Integrity report SHA-256 is `f24bfe74a6eee979c3ec0f3d466da0419aa5f38558c22632be9b76d75cb6ec26`.
- **Scientific decision:** `eligible_with_strict_limits`. Originals support real-film scan style/nuisance work; restored siblings support damage controls only. They are not neutral digital targets. Filename families are provisional hints, physical roll/process/scanner are unknown, five families have one sample, and much of the content depicts film/camera/archive objects. Claim ceiling is `real-film-derived/unknown-look`.
- **Propagation/handoff:** added `configs/real_film_filmr_decision.json` and `docs/data/REAL_FILM_FILMR_V2_GATE.md`; RF0.1 and RF0.2 close, RF1.1/RF1.2 become ready. RF1.1 must test content/damage/source nuisance before training any family expert.

## 2026-07-15 - Freeze BlueNeg nested LOO sign-reversal diagnosis

- **Node/parent goal:** `ULT > RF1.2`; explain CT6 roll-sign heterogeneity without adding capacity or relabeling a retrospective rerun as confirmation.
- **Cohort:** all 43 locally complete aligned frames from the four already accessed Kodak Gold 100-5 rolls become a query exactly once. Each query is excluded from support; the extra source-only frame is excluded. No new download is required.
- **Frozen arms:** the existing Lab mean/std operator only; three-frame correct roll, every individual wrong roll, pooled wrong rolls, source-only content retrieval, count-preserving shuffled-roll and identity controls. Raw and query-source-only style-matched results are mandatory.
- **Decision:** all four roll point estimates must be positive and correct-roll must survive every matched control to reopen group information. Retrieval/pooled/shuffle success indicates nuisance; mixed signs or a cluster interval containing zero remains unidentified. This diagnostic cannot promote stock, calibration, digital-to-film or fresh confirmatory claims.
- **Evidence/handoff:** `configs/roll2film_blueneg_nested_loo.json` and `docs/data/ROLL2FILM_BLUENEG_NESTED_LOO_CONTRACT.md`. Implement and test the deterministic runner before executing the 43 folds.

## 2026-07-15 - Conclude BlueNeg sign reversal as source-content nuisance

- **Node/parent goal:** `ULT > RF1.2`; run the frozen 43-query nested LOO diagnosis and propagate the architecture implication without upgrading BlueNeg's claim class.
- **Primary result:** raw wrong-roll content retrieval scores Delta-E `5.1805` versus correct roll `5.6044` and wins on every physical-roll mean. The cluster-equal correct-roll gain is `-0.3856`, 95% interval `[-0.4956,-0.2071]`. Pooled wrong and shuffled roll are approximately tied with correct roll; correct roll remains much better than identity.
- **Matched result:** source-only style matching leaves three rolls negative and one positive versus retrieval; aggregate gain is `-0.1130`, interval `[-0.2303,+0.0574]`. Because extrapolative matching can produce 20.63% sampled OOR, this is secondary; raw outputs have zero OOR and give the stronger same conclusion.
- **Nuisance diagnosis:** retrieval descriptor distance is `1.355` versus correct-support `2.458`. Retrieval rarely shares location (`2.3%`) but more often shares day/night plus indoor/outdoor scene properties (`77.5%`). Physical-roll labels are not identified; content/illumination-conditioned restoration mapping is the supported mechanism.
- **Reproducibility/decision:** 101 tests pass. Two complete executions are byte-identical; report SHA-256 is `3b87a8604667bceae9bdd9506daa60f7380ef574fd20145732c9fcf96c1fb333`. Decision is `nuisance_explanation`; keep roll information closed and promote bounded similar-case retrieval only as an RF2 challenger pending RF1.1 real-film gates.
- **Evidence/handoff:** `configs/roll2film_blueneg_nested_loo_decision.json`, `docs/ROLL2FILM_BLUENEG_NESTED_LOO_RESULTS.md`. Next ready leaf is RF1.1 FILM-R signal/content/damage separability.

## 2026-07-15 - Freeze FILM-R signal/content/nuisance audit

- **Node/parent goal:** `ULT > RF1.1`; determine whether FILM-R filename-family signal is identifiable beyond the visibly severe content imbalance before fitting any expert.
- **Visual labels:** all 44 pairs receive one of four coarse content categories using the already hashed review sheets. The mapping is frozen in config; restored siblings inherit the pair label and may never cross a split.
- **Stage-zero gate:** before feature decode, require each eligible family to have at least two samples in two content cells and require at least three mutually comparable families with a shared supported cell. Structural zeros stop the classifier route rather than permitting random-split accuracy.
- **Conditional controls:** only after support passes, compare colour-only, grayscale-structure-only, damage-only and combined descriptors under grouped leave-one-out, leave-one-content-category-out, stratified permutations and restoration perturbation.
- **Evidence/handoff:** `configs/real_film_filmr_signal_audit.json`, `docs/data/REAL_FILM_FILMR_SIGNAL_AUDIT_CONTRACT.md`. Implement the metadata-only support-matrix gate, test/commit it, then execute before decoding new feature pixels.

## 2026-07-15 - Stop FILM-R family learning at structural identifiability gate

- **Node/parent goal:** `ULT > RF1.1`; execute the stage-zero family-by-content support audit before any feature decode or classifier fit.
- **Result:** among five filename families with at least three pairs, only Velvia 50 has two content cells with at least two samples. The largest comparable clique (Cinestill 800T, Ektachrome 100, Portra 400) overlaps only on product/equipment; there are zero comparable cross-content families. Family/content Cramér's V is `0.6298`.
- **Decision:** `structurally_unidentified`. No new feature pixels were decoded and no classifier was trained. FILM-R cannot support family/stock, Roll2Film or retrieval training; it remains valid for unknown-look, damage and artifact stress evidence.
- **Reproducibility:** 104 tests pass; metadata gate reruns byte-identically with report SHA-256 `88c3d771617469a624e57729320e63462332ca0525f605cbd747e2679c055ba8`.
- **Propagation/handoff:** added `configs/real_film_filmr_signal_decision.json` and `docs/REAL_FILM_FILMR_SIGNAL_AUDIT_RESULTS.md`; RF2 becomes data-gated and RF0.3 is now executing to locate a content-balanced, rights-clear real-film source with honest source-level holdouts.

## 2026-07-15 - Qualify the LOC FSA/OWI real-film archive for bounded Phase C

- **Node/parent goal:** `ULT > RF0.3`; replace the structurally unidentified FILM-R learning lane with a content-diverse, verifiable physical-film source without inventing roll or stock truth.
- **Source/rights:** LOC documents approximately 1,600 original FSA/OWI colour transparencies, common 2003-2004 Sinar scanning and public-domain reuse. The reproducible Commons API mirror retained LOC `fsac.*` identifiers. The raw 1,031-page category failed because it contains repeated crops/derivatives and six rights/category exceptions.
- **Canonical metadata:** strict filtering and deterministic one-per-LOC selection yielded 558 unique scans, removed 419 repeated representations and achieved 81.90% curated creator coverage. Raw failure remains recorded. Metadata report SHA-256 is `67db03a94de75a47cf2f7504ae82739b812e3fcc5340dd7945a6aec758a75416`.
- **Pixel pilot/visual evidence:** 64 creator-balanced 1280px derivatives total 20,570,056 bytes; all decode as RGB JPEG, with zero exact or dHash<=4 duplicates. Two contact sheets were visually adjudicated: no severe glitch, modern colourization or caption mismatch was confirmed. Strong ageing/cast variation, mount borders and related shooting sequences require explicit nuisance controls. Pilot report SHA-256 is `3eac975d4012d3d13cd4953a64930b2ef69e2ccd2441763146e25a70018aaa15`.
- **Grouping result:** sequence-only grouping failed (11 groups >=3 versus 12 required). A preregistered recovery grounded in LOC assignment tables combined creator with caption state/territory and sequence fallback: 457 known-creator records, 272 location-recovered, 43 groups, 30 groups >=3 and maximum group share 14.22%. Unknown-creator 101 stay stress-only. Group report SHA-256 is `4bbf331383e2f1c376e2a95dbd24d3fe29c46fd29ea219d3e5fc84645f0498e9`.
- **Decision/handoff:** `grouped_phase_c_allowed`; not training allowed. Next leaf downloads at most 558 derivatives/1GiB, freezes deterministic border/interior masks and runs content/scanner-age nuisance identifiability before RF2. Groups are leakage guards, not roll/LOT/process/scanner truth.

## 2026-07-15 - Correct Ultimate to stock-first real-film learning

- **Node/parent goal:** `ULT > RF0.4`; propagate the owner's clarification that specific film stocks are the primary classes. Historical/unknown-stock film remains useful but cannot replace or count toward concrete stock coverage.
- **Trigger:** the active RF path had correctly reopened real physical-film evidence, but its next learnable source was the stock-unknown LOC archive. Without an explicit hierarchy, data availability could silently turn the programme into generic archive-film learning.
- **Skills used:** `dev-research-reliability` as the sole writer; router, scientific research, plan/tracker, DRPT-BI, agent-log and structure stewardship as read-only governance.
- **Decisions:** make `film_stock_id` the highest-level expert class; keep physical roll, process, scanner, source and content as nested controls; create separate `S0/S1/S2/S3` stock grades and `H historical/unknown`; require separate named-stock and historical coverage; set a first milestone of at least three distinguishable `S2` stock experts plus one independent `H` expert; keep calibration at `S3` only.
- **Files changed:** `AGENTS.md`, `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md`, `docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`, `docs/ULTIMATE_EXECUTION_TRACKER.md`, `TASK_BOARD.md`, `IMPL_PLAN.md`, `docs/planning/README.md` and this log.
- **Evidence/compatibility:** no code, data, configs, frozen manifests, experiment results or runtime outputs changed. FilmSet remains auxiliary; BlueNeg remains provisional single-stock evidence below `S2`; FILM-R remains stopped; LOC Phase C remains allowed only as the independent `H historical/unknown` lane.
- **Handoff:** `RF0.4` is the nearest named-stock P0 leaf: audit obtainable authoritative stock-labelled sources and select no more than 2-4 first pilots after label, roll/source, content, nuisance, rights, size and evidence-grade gates. `RF0.3/RF1.3` may continue independently but can never fill a named-stock slot.

## 2026-07-15 - Freeze the stock-first programme and first audit pilots

- **Node/parent goal:** `ULT > RF0.4/SF0.1`; turn the stock-first correction into a machine-checkable source registry and executable experiment DAG.
- **Skills used:** `dev-research-reliability` is the sole writer; ML, scientific-research, plan/tracker, DRPT-BI, agent-log and structure stewardship govern evaluation and propagation.
- **Source decision:** BlueNeg is the only immediately executable multi-stock physical-film source with exact revision, roll IDs, dataset-declared film strings, public bounded preview/proxy lanes and frozen hashes. Xi Film is a promising licensed-source candidate but currently exposes only a tiny paid marketplace and no verified ML/roll contract. Apollo is a metadata candidate with severe content-domain confounding. FILM-R, LOC, FilmSet, SillyStill and Emulating Emulsion retain their existing lower/blocked roles.
- **Pilot decision:** the initial paper selection was Kodak Gold 100-5, Kodak Gold 400-5, Fuji NPH400 and Konica Super XG 100. Whole-test-roll preflight then reduced Gold 400-5 from 3 nominal rolls / 38 frames to 1 unsealed roll / 1 frame, so it was removed before download and replaced by Kodak GA 100 5095 (3 unsealed rolls / 16 frames). Selection is not promotion: all process/lab/scanner/market fields remain `unknown`; ambiguous Fuji 100/400 remain `S0`; no expert is `S2`.
- **FSA propagation:** stop the running historical auxiliary expansion and retain 258 FSA/OWI derivatives / 81,016,399 bytes. Do not delete them, resume them as a named-stock source, or let the auxiliary lane block RF0.4.
- **Files changed:** stock registry/config/validator/tests; `docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md`; active authorities, tracker, board, data boundaries and superseded Roll2Film pointer.
- **Verification/handoff:** JSON parses; all 124 repository tests pass; the exact BlueNeg 13-string/53-roll/491-frame cross-check passes with no mismatches. The acquisition contract freezes all four physical input hashes, seals every official-test roll, requires each preview and public proxy in the exact-revision LFS inventory, rejects missing hashes/duplicate paths, and reports aligned-but-unavailable proxies separately. After a scoped commit, rerun at the committed software identity and freeze the final manifest/report before downloading pixels.

## 2026-07-15 - Pass SF0.1 and freeze exact stock-pilot bytes

- **Node/parent goal:** `ULT > RF0.4 > SF0.1`; execute the committed stock registry/acquisition contract before accessing new pixels.
- **Result:** two executions at commit `70fe95bb...` are byte-identical. The exact manifest has 189 objects / 227,287,697 bytes and SHA-256 `1ac2dfce6060d0094d221e8776f4318ebdd144a86194c534daca01ae3749324e`; report SHA-256 is `9ba954cffd5f47839fcb14fe7b025bb8be27ca48256baf7b579dce4c21e756d3`.
- **Split evidence:** 17 official-test rolls are globally sealed. Gold 100-5 retains 7 rolls/50 previews/47 proxies; Konica retains 7/22/1; NPH400 retains 4/53/0; GA 100 5095 retains 3/16/0. Gold 400-5 was removed because only one roll/frame survives sealing.
- **Decision:** `SF0.1 passed`; `SF0.2` may download only the exact manifest into isolated `data/raw/blueneg_stock_pilots_v1`. The old 101-file pilot and 258 FSA derivatives remain untouched. No pixels have yet been decoded and no stock signal/transfer claim is promoted.
- **Files/handoff:** add the acquisition decision, frozen result document and isolated downloader; commit/push, then run resumable download and verify every size/LFS hash before any visual or feature audit.
