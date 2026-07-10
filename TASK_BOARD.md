# TASK_BOARD.md — Ultimate strongly stylized, artifact-safe film-imaging path

> Updated 2026-07-11. Detailed DoR/DoD, dependencies, gates and evidence live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.
> SDXL/IP2P/SDEdit is a retired production direction and an optional research/Creative comparator only.

---

## Current truth

| Area | State | Evidence/next action |
|---|---|---|
| Deterministic renderer | current default | `safe_lab`/safe-rich + optional grain/halation/dust |
| Diffusion/IP2P | retired as default | detail/identity drift; tested SDXL full-UNet OOM on 12GB |
| Neural LUT/local maps | research-only | pseudo-teacher or saturation-gate evidence is insufficient |
| FilmCase ML | primary bounded-ML research hypothesis | source-controlled identifiability → bounded cases → Oracle → simplest retrieval/ranker |
| Input pipeline | partial | `WorkingImage` exists but is not used by final renderer |
| Product standard | decided | strongly stylized output with no severe glitch/artifact |
| User contribution | frozen at existing evidence | no new images, pairs, labels or votes may be project dependencies |
| Stock accuracy | deferred calibrated lane | paired evidence is required only for future calibrated claims; it does not block FilmCase |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | 18 tests passed on 2026-07-10; no CI yet |

---

## Ready queue

| Order | Node | Task | Status | Approval |
|---:|---|---|---|---|
| 0 | U0.1 | Reconcile active README/status/docs; archive stale diffusion instructions | complete | completed 2026-07-10 |
| 0a | U5.FC0 | Freeze autonomous unpaired non-generative FilmCase plan | complete | completed 2026-07-11 |
| 1 | U0.3 | Manifest v2, data lanes, group split, duplicate/leakage repair | ready | none for local audit |
| 2 | U0.4 | CI, environment capture and frozen benchmark registry | ready | none |
| 3 | U4.1/U4.2 | Implement severe-artifact veto + autonomous style/appeal/counterfactual scorecards | pending on U0.4 | none |
| 4 | U1.1 | Make `WorkingImage` the only `render_film` ingress | pending on U0.4 | none |
| 5 | U1.2–U1.5 | Color-state contract, 16-bit/ICC export, HDR/HEIF handling | pending | none |
| 6 | U2.1–U2.5 | Profile/recipe schema and deterministic reference renderer | pending | none for local research; U0.2 before public release |
| 7 | U5.FC1–U5.FC8/U6 | FilmCase identifiability, Oracle, retrieval/OOD and artifact-safe effects | pending on U0.3/U0.4/U4 | GPU/cost gate only if later needed |
| 8 | U3.1–U3.4 | Optional Portra 400 + Velvia 50 calibrated profile lane | deferred | not an active user ask or dependency |
| 9 | U7/U8 | Product, beta, release and stock expansion | pending | release/legal approval |

---

## Human decisions required

1. Confirm intended code/profile/data release model and repository license.
2. Approve any large model/data download, paid GPU, external participant contact, push/merge or public release before it occurs.

Still-photo desktop is the default first contract. Paired capture and external human validation remain deferred rather than repeatedly requested.

---

## Protected current work

Do not touch or stage the existing untracked user files:

- `halationguide.md`
- `scripts/make_velvia50_scheme_comparison_sheets.py`

Current branch is local research work with commits not represented by a matching remote branch. Do not push unless explicitly requested.

---

## Success gate

Research advances when a candidate is strongly stylized under the frozen autonomous visual protocol, produces zero confirmed severe artifacts on the gold set, reports artifact rate on the wider stress set, and remains reproducible. FilmCase additionally requires source-controlled identifiability and an Oracle win over the global champion before any router training. External population preference is a deferred U8 validation; stock/process holdouts are required only for profiles labeled calibrated. If a deterministic or simple retrieval model passes, do not add neural complexity.

---

*Active parent: `ULT` | Next ready leaf: `U0.3` FilmCase asset eligibility/lineage audit | Integration owner: repository owner or explicitly assigned root agent*
