# TASK_BOARD.md — Algorithm-first film colour-transfer and product path

> Updated 2026-07-12. Detailed DoR/DoD, dependencies, gates and evidence live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.
> SDXL/IP2P/SDEdit is a retired production direction and an optional research/Creative comparator only.

---

## Current truth

| Area | State | Evidence/next action |
|---|---|---|
| Deterministic renderer | current default | `safe_lab`/safe-rich + optional grain/halation/dust |
| Diffusion/IP2P | retired as default | detail/identity drift; tested SDXL full-UNet OOM on 12GB |
| Neural LUT/local maps | research-only | pseudo-teacher or saturation-gate evidence is insufficient |
| Roll2Film research | **primary algorithm-paper hypothesis** | unpaired physical-roll sets → shared explicit colour-operator identification → actual transfer → hidden target/style/artifact gates |
| FARO/ChromaticTail | supporting evaluation/product wrapper | severe-artifact evaluation, fixed-policy audit and fallback; no standalone primary benchmark paper |
| FilmCase ML | baseline/ablation | source-controlled retrieval and hard expert selection remain comparison branches |
| Input pipeline | partial | `render_film` now enters through `WorkingImage`; an explicit legacy sRGB8 adapter remains |
| Product standard | decided | strongly stylized output with no severe glitch/artifact |
| Current local data | insufficient by itself for honest main-paper training | 3,896 local Flickr JPEGs quarantined; 26 unique traceable Velvia references; 100 synthetic pseudo-pairs; no current FiveK/FilmSet/BlueNeg files |
| Remote-verified research data | feasible, not yet local | FilmSet ~11.26GB: official CLI listed files and one member downloaded successfully; BlueNeg ~956MB initial 8-bit lanes: file tree and range read verified for 53-roll pilot |
| User contribution | not a current dependency | no new owner images, pairs, labels or votes are required for the no-data/FilmSet/BlueNeg stages |
| Stock accuracy | deferred calibrated lane | paired evidence is required only for future calibrated claims; it does not block FilmSet/BlueNeg Level-A/B research |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | last recorded full run passed; 6 targeted ingress tests and reproducibility baseline passed after U1.1 partial; CI exists |

---

## Ready queue

| Order | Node | Task | Status | Approval |
|---:|---|---|---|---|
| 0 | U0.1 | Reconcile active README/status/docs; archive stale diffusion instructions | complete | completed 2026-07-10 |
| 0a | U5.FC0 | Freeze autonomous unpaired non-generative FilmCase plan | complete | completed 2026-07-11 |
| 0b | U5.R0 | Freeze FARO novelty audit and publication research program | complete | completed 2026-07-11 |
| 0c | U5.CT0 | Correct publication priority to Roll2Film colour-transfer algorithm and audit data/nearest work | complete | completed 2026-07-12 |
| 1 | U5.CT1 | Freeze invertible explicit-operator API and known-operator pseudo-roll simulator | ready, no data | none |
| 2 | U5.CT2 | Build metadata-only FilmSet/BlueNeg eligibility, grouping and pair-blinding contracts | ready, no images | none |
| 3 | U5.CT3 | Run group-size, coverage and nuisance identifiability curves | pending on CT1 | none; stop before data download if the hypothesis fails |
| 4 | U5.CT4 | Restore/download FilmSet and freeze paired-blind splits only after CT3 passes | approval-gated | explicit large-download approval |
| 5 | U5.CT5 | Compare unpaired transfer baselines with Roll2Film on hidden FilmSet pairs | pending on CT3/CT4 | GPU/cost gate only if required |
| 6 | U5.CT6 | Download only BlueNeg preview/pseudo-GT lanes and run correct-roll matched-control pilot | approval-gated | explicit download approval |
| 7 | U5.CT7/U5.CT8 | Amortised set inference, complete ablation and hidden transfer/preference study | blocked on CT5/CT6 | participant approval for human study |
| 8 | U5.CT9 | Optional controlled named-stock calibration | future | new capture/lab scope and approval |
| 9 | U5.R0T | Audit theorem-level gap in FARO certification | deferred optional support | not a primary paper dependency |
| 10 | U0.3 | Legacy manifest-v2 lineage audit | complete; reference-derived lane blocked | 4,210/4,210 quarantined; new public-data contracts live under U5.CT2 |
| 11 | U0.4 | CPU-safe CI, environment capture and explicit non-gold registry | complete | current U4 evaluation assets remain supporting work |
| 12 | U5.R1A/U4.1/U4.2 | Maintain chromatic ontology/look rubric and stress evidence as CT evaluation support | ready support | none for local schema/tooling; participants later require approval |
| 13 | U1.1 | Make `WorkingImage` the only `render_film` ingress | in progress; first ingress adapter committed | none |
| 14 | U1.2–U1.5 | Color-state contract, 16-bit/ICC export, HDR/HEIF handling | pending | none |
| 15 | U2.1–U2.5 | Profile/recipe schema and deterministic reference renderer | pending | none for local research; U0.2 before public release |
| 16 | U5.R2–U5.R7/U5.FC1–U5.FC8/U6 | FARO/FilmCase baselines, product fallback and artifact-safe effects | supporting/conditional | GPU/cost and participant gates only if later needed |
| 17 | U3.1–U3.4 | Optional Portra 400 + Velvia 50 calibrated profile lane | deferred | not an active user ask or dependency |
| 18 | U7/U8 | Product, beta, release and stock expansion | pending | release/legal approval |

---

## Human decisions required

1. Confirm intended code/profile/data release model and repository license.
2. Approve any large model/data download, paid GPU, external participant contact, push/merge or public release before it occurs.

Still-photo desktop is the default first contract. Paired capture and external human validation remain deferred rather than repeatedly requested.

---

## Protected current work

Do not touch or stage the existing user deletion of `data/raw/.gitkeep`; it is
outside this research-document leaf.

Current branch is local research work with commits not represented by a matching remote branch. Do not push unless explicitly requested.

---

## Success gate

The primary research route advances only when a frozen algorithm applies an
explicit colour operator and group-conditioned inference beats matched
single-reference and pooled-unpaired colour-transfer baselines on hidden
targets while preserving a preregistered style floor and artifact ceiling.
Group-size/coverage scaling, shuffled-roll controls and scanner/exposure/WB
shortcut tests are mandatory. If Roll2Film fails, close the algorithm paper;
do not relabel the supporting benchmark as the main result. FARO/FilmStyleSafe
continues independently as product QA. External population preference remains
approval-gated, and named-stock claims require controlled paired whole-roll
evidence.

---

*Active parent: `ULT > U5.CT` | Next research leaves: `U5.CT1` explicit operator/simulator and `U5.CT2` metadata-only data contracts | Next engineering leaf: continue `U1.1/U1.3` high-precision path | Integration owner: repository owner or explicitly assigned root agent*
