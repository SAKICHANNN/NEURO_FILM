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
| Current local data | FilmSet freeze plus verified BlueNeg pilot | FilmSet official 628 remains sealed; BlueNeg exact 101 files / 118,929,719 bytes are local and hash-verified, with pixels still undecoded before evaluator freeze |
| Remote-verified research data | BlueNeg narrow pilot ready | only four Kodak Gold 100-5 rolls support same-film matched controls; 13-film-type generalization is unavailable |
| User contribution | not a current dependency | no new owner images, pairs, labels or votes are required for the no-data/FilmSet/BlueNeg stages |
| Stock accuracy | deferred calibrated lane | paired evidence is required only for future calibrated claims; it does not block FilmSet/BlueNeg Level-A/B research |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | 86 tests pass after CT5 severity and BlueNeg metadata/download/alignment-contract diagnostics; CI exists |

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
| 6 | U5.CT6 | Run BlueNeg correct-roll matched-control pilot | evaluator frozen before pixel decode; development-only family selection next | 101 files / 118,929,719 bytes; one film string only |
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
2. Approve paid GPU, external participant contact, merge or public release before it occurs. Necessary research-data downloads and pushes are pre-authorized for the active autonomous goal.

Still-photo desktop is the default first contract. Paired capture and external human validation remain deferred rather than repeatedly requested.

---

## Protected current work

Do not touch or stage the existing user deletion of `data/raw/.gitkeep`; it is
outside this research-document leaf.

The active autonomous goal explicitly authorizes scoped commits and pushes. Preserve unrelated user work and never merge or release without a separate gate.

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

*Active parent: `ULT > U5.CT` | Next leaves: `U5.CT2/U5.CT6` BlueNeg whole-roll contract and matched-control pilot, plus `U1.1/U1.3` high-precision path | Integration owner: repository owner or explicitly assigned root agent*
