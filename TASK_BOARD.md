# TASK_BOARD.md — Ultimate strongly stylized, artifact-safe film-imaging path

> Updated 2026-07-10. Detailed DoR/DoD, dependencies, gates and evidence live in `docs/ULTIMATE_EXECUTION_TRACKER.md`.
> SDXL/IP2P/SDEdit is a retired production direction and an optional research/Creative comparator only.

---

## Current truth

| Area | State | Evidence/next action |
|---|---|---|
| Deterministic renderer | current default | `safe_lab`/safe-rich + optional grain/halation/dust |
| Diffusion/IP2P | retired as default | detail/identity drift; tested SDXL full-UNet OOM on 12GB |
| Neural LUT/local maps | research-only | pseudo-teacher or saturation-gate evidence is insufficient |
| Input pipeline | partial | `WorkingImage` exists but is not used by final renderer |
| Product standard | decided | strongly stylized output with no severe glitch/artifact |
| Stock accuracy | optional calibrated lane | requires owned paired stock/process/scan data before using calibrated claims |
| License/release | blocked | docs say MIT but root `LICENSE` is absent |
| Tests | baseline passes | 18 tests passed on 2026-07-10; no CI yet |

---

## Ready queue

| Order | Node | Task | Status | Approval |
|---:|---|---|---|---|
| 0 | U0.1 | Reconcile active README/status/docs; archive stale diffusion instructions | complete | completed 2026-07-10 |
| 1 | U0.3 | Manifest v2, data lanes, group split, duplicate/leakage repair | ready | none for local audit |
| 2 | U0.4 | CI, environment capture and frozen benchmark registry | ready | none |
| 3 | U4.1/U4.2 | Implement severe-artifact veto + style/preference optimization scorecards | pending on U0.4 | none |
| 4 | U1.1 | Make `WorkingImage` the only `render_film` ingress | pending on U0.4 | none |
| 5 | U1.2–U1.5 | Color-state contract, 16-bit/ICC export, HDR/HEIF handling | pending | none |
| 6 | U2.1–U2.5 | Profile/recipe schema and deterministic reference renderer | pending | license decision for public schema assets |
| 7 | U5/U6 | Bounded LUT/grid style challenge and artifact-safe effects | pending on U4 | GPU/cost gate if needed |
| 8 | U3.1–U3.4 | Optional Portra 400 + Velvia 50 calibrated profile lane | blocked | budget/lab/rights approval |
| 9 | U7/U8 | Product, beta, release and stock expansion | pending | release/legal approval |

---

## Human decisions required

1. Confirm intended code/profile/data release model and repository license.
2. Approve or reject the two-stock capture pilot and external lab/scanner budget.
3. Decide whether “ultimate” initially targets still-photo desktop only or also video/plugin hosts; still-photo is the recommended first contract.
4. Approve any large model/data download, paid GPU, external participant contact, push/merge or public release before it occurs.

---

## Protected current work

Do not touch or stage the existing untracked user files:

- `halationguide.md`
- `scripts/make_velvia50_scheme_comparison_sheets.py`

Current branch is local research work with commits not represented by a matching remote branch. Do not push unless explicitly requested.

---

## Success gate

The product advances when a candidate is strongly stylized, wins a preregistered preference comparison, produces zero confirmed severe artifacts on the frozen gold set, reports artifact rate on the wider stress set, and remains reproducible on Windows 12GB, M5 and CPU fallback. Stock/process holdouts are additionally required only for profiles labeled calibrated. If a deterministic model passes, do not add neural complexity.

---

*Active parent: `ULT` | Next ready leaf: `U0.3` | Integration owner: repository owner or explicitly assigned root agent*
