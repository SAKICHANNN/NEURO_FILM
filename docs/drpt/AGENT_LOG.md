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
