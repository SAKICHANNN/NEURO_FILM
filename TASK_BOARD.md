# TASK_BOARD.md — Algorithm-first film colour-transfer and product path

> Updated 2026-07-15. Detailed DoR/DoD, dependencies, gates and evidence live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.
> SDXL/IP2P/SDEdit is a retired production direction and an optional research/Creative comparator only.

---

## Current truth

| Area | State | Evidence/next action |
|---|---|---|
| Deterministic renderer | current default | `safe_lab`/safe-rich + optional grain/halation/dust |
| Diffusion/IP2P | retired as default | detail/identity drift; tested SDXL full-UNet OOM on 12GB |
| Neural LUT/local maps | research-only | pseudo-teacher or saturation-gate evidence is insufficient |
| Roll2Film research | **conditional algorithm-paper challenger** | fixed-budget affine and L2 method controls pass; real-roll matched controls must still establish special group information |
| FARO/ChromaticTail | supporting evaluation/product wrapper | severe-artifact evaluation, fixed-policy audit and fallback; no standalone primary benchmark paper |
| FilmCase ML | conditional product/research branch | open only if strength-adjusted bank diversity and an applicability/safety Oracle gap exist; otherwise champion + bounded strength wins |
| Input pipeline | partial | `render_film` now enters through `WorkingImage`; an explicit legacy sRGB8 adapter remains |
| Product standard | decided | strongly stylized output with no severe glitch/artifact |
| Current local data | FilmSet paired-blind evidence freeze passed | 21,140 files / 11,262,805,356 bytes; 2,096 source-only + 2,096 target-only + 465 internal-dev identities; official 628 remains sealed; BlueNeg absent |
| Remote-verified research data | BlueNeg feasible, not local | ~956MB initial 8-bit lanes: file tree and range read verified for 53-roll pilot; download remains approval-gated |
| User contribution | not a current dependency | no new owner images, pairs, labels or votes are required for the no-data/FilmSet/BlueNeg stages |
| Stock accuracy | deferred calibrated lane | paired evidence is required only for future calibrated claims; it does not block FilmSet/BlueNeg Level-A/B research |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | 64 tests pass after FilmSet and L2 operator/falsification implementation; CI exists |

---

## Ready queue

| Order | Node | Task | Status | Approval |
|---:|---|---|---|---|
| 0 | U0.1 | Reconcile active README/status/docs; archive stale diffusion instructions | complete | completed 2026-07-10 |
| 0a | U5.FC0 | Freeze autonomous unpaired non-generative FilmCase plan | complete | completed 2026-07-11 |
| 0b | U5.R0 | Freeze FARO novelty audit and publication research program | complete | completed 2026-07-11 |
| 0c | U5.CT0 | Correct publication priority to Roll2Film colour-transfer algorithm and audit data/nearest work | complete | completed 2026-07-12 |
| 1 | U5.CT1 | Freeze invertible explicit-operator API and known-operator pseudo-roll simulator | L2 spline core passes; full CT1 still needs explicit L0/gauge/shaper closure | preserve analytic inverse/Jacobian and bake parity |
| 2 | U5.CT2/U5.CT4 | Freeze local FilmSet manifest, rights caveat, pair blindness and inaccessible 628 lockbox; keep BlueNeg metadata-only | FilmSet CT4 complete; CT2 continues only for BlueNeg metadata | none for local internal work |
| 3 | U5.CT3 | Extend fixed-budget E0 from affine to L2 truth and stronger nuisance/prior/scanner factorials | L2 method controls pass; roll information not established | matched real-roll evidence remains |
| 4 | U5.CT5 | Build matched-strength deterministic/classical baselines and run FilmSet internal paired-blind development | pending on nonlinear CT3 and baseline freeze; FilmSet data gate passed | GPU/cost gate only if required; official 628 remains sealed |
| 6 | U5.CT6 | Download only BlueNeg preview/pseudo-GT lanes and run correct-roll matched-control pilot | scientifically gated; owner pre-authorized necessary downloads 2026-07-15 | CT3/metadata gate, whole-roll split and storage/retention record |
| 7 | U5.CT7/U5.CT8 | Optional amortised set inference, complete ablation and final hidden transfer/preference study | blocked on MAP/diversity/Oracle gates | participant approval for human study; 628 opens once after full policy freeze |
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

The deterministic product route advances independently. Roll2Film becomes the
primary method-paper route only when fixed-total-sample group-conditioned
inference beats matched partitions, pooled/shuffled groups and strong unpaired
baselines while preserving a frozen style floor and artifact ceiling.
Coverage, nuisance-capacity, prior-swap, scanner/exposure/WB and shortcut tests
are mandatory. Failure closes the special group-information claim and permits a
fixed champion or rules-based system to win; it does not relabel a benchmark as
the main paper. External population preference remains approval-gated, and
named-stock claims require controlled paired whole-roll evidence.

---

*Active parent: `ULT > U5.CT` | Next leaves: `U5.CT3` nonlinear fixed-budget identifiability, matched-strength CT5 baseline freeze and `U1.1/U1.3` high-precision path | Integration owner: repository owner or explicitly assigned root agent*
