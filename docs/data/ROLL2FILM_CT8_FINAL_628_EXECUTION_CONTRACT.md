# CT8 one-shot FilmSet final-628 execution contract

Date: 2026-07-15

Node: `ULT > U5.CT8`

Status: frozen before first final-628 pixel decode

## Purpose

Confirm the independent fixed deterministic recipe bank once on the official
628-identity FilmSet lockbox. This is not a rescue experiment for Roll2Film:
BlueNeg roll information is ambiguous and CT7 is stopped.

## Frozen bank

| Recipe | Primary | Fixed best-basic opponent |
|---|---|---|
| Cinema | Lab mean/std | WB/contrast/saturation |
| ClassNeg | pooled L2 | WB/contrast/saturation |
| Velvia | pooled L2 | WB/contrast/saturation |

All operator bundles come from the already frozen CT5 pilot report. Final
pixels cannot refit operators, select a family, tune strength or add a wrapper.

## Access and execution

- Validate every upstream report/decision hash and the 2,512-row final manifest.
- Verify every input/target payload SHA-256 before decode.
- Require exactly 628 identities with input plus three targets.
- Decode through `WorkingImage` to linear sRGB.
- Render primary and best basic at full resolution without spatial resampling.
- Use explicit hard clip plus sRGB8 only for review artifacts.
- Run once after the policy commit. A deterministic rerun may verify bytes but
  cannot change a decision or candidate.

## Statistical gate

For each recipe, bootstrap final identities/clusters 10,000 times. Primary
target Delta-E00 improvement over the fixed best basic must have a 95% lower
bound above zero. Primary mean style strength must be at least the fixed best
basic's mean style strength. Identity, final target and primary remain separate
roles; style does not substitute for recipe fidelity.

## Severe visual gate

For each recipe, save the 12 composite-worst primary cases and the top five
cases on raw excursion magnitude, newly clipped pixels versus target, target
Delta-E, red/cyan boundary occupancy and chroma speckle. Review the contact
sheets and every unique automatic-trigger extreme at original resolution.

Non-finite output or confirmed content/geometry corruption, posterization,
banding, seams, colour blocks, unstable speckle or unintended large-area
clipping is a hard failure. Crossing `0.05` raw excursion or `0.10` newly
clipped pixels is a mandatory review trigger, not automatic failure.

## Promotion and claim boundary

A domain passes only with the fidelity CI, style ratio and severe veto. The bank
passes only if all three domains pass. A failed domain retains the existing
safe_lab/safe-rich product fallback; there is no post-hoc weakening on final.

Results are FilmSet Capture One recipe-transfer evidence only. No human
preference, real-film, named-stock, calibration, redistribution or Roll2Film
group-information claim is permitted.
