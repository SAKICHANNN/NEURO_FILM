# Roll2Film E0 v2 — fixed-budget affine falsification

> Date: 2026-07-15
>
> Node: `ULT > U5.CT3`
>
> Decision: **method controls pass; physical-roll information remains not established**

## Question

When total target pixels are held fixed, what information comes from frame
partition, true nuisance boundaries, independent colour support and correct
single-operator grouping?

This experiment corrects the central confound in E0 v1. V1 used 128 pixels per
frame, so increasing `frames=1→32` also increased total target pixels
`128→4096`. V2 fixes the budget at 4,096 target pixels for every group size.

## Frozen contract

- config: `configs/roll2film_e0_fixed_budget.json`;
- config SHA-256: `c90a63f9802df7e5f6c80496f671e3fb32bafa882317cb49480f6d2676c48c06`;
- group sizes: `1/2/4/8/16/32`;
- total target pixels: 4,096 in every arm;
- replicates: 64; replicate is the bootstrap unit;
- independent neutral prior: 8,192 pixels;
- independent operator holdout: 4,096 pixels;
- operator/estimator: the preserved SPD-affine Gaussian-transport foundation;
- ignored raw report: `outputs/roll2film/e0_fixed_budget/report.json`;
- no real images, pairs, FilmSet targets or BlueNeg pixels are used.

The four method controls are:

1. **Partition equivalence:** partition the exact same no-nuisance pixel pool
   into different frame counts. Frame labels must not change the estimate.
2. **Nuisance boundary information:** at fixed pixels, compare true exposure
   boundaries with random pixel repartition before the same per-frame exposure
   normalization.
3. **Independent support information:** compare independent pixels with the
   same first frame repeated to fill the identical pixel budget.
4. **Mixed-operator rejection:** replace half the frames with a second operator
   at the same frame count and pixel budget.

Prior shift and scanner composition are diagnostics only. They cannot make the
method gate pass.

## Results

### Fixed-budget curves

| Frames | Independent | Repeated support | Mixed operator | Nuisance, true boundary | Nuisance, random boundary |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003665 | 0.003665 | 0.003665 | 0.042866 | 0.042866 |
| 2 | 0.003687 | 0.004998 | 0.063662 | 0.035392 | 0.037860 |
| 4 | 0.003906 | 0.006448 | 0.063995 | 0.033628 | 0.036711 |
| 8 | 0.003648 | 0.008684 | 0.064042 | 0.020508 | 0.023935 |
| 16 | 0.003724 | 0.012051 | 0.063662 | 0.017385 | 0.023646 |
| 32 | 0.003529 | 0.016726 | 0.063662 | 0.010826 | 0.020381 |

Metric is mean independent-holdout RGB RMSE against the known film component.
At one frame the mixed arm necessarily contains only the first operator, so it
equals the independent arm; the hostile mixture begins at two frames.

### Gate at 32 frames

| Control | Baseline RMSE | Candidate RMSE | Relative improvement | Paired bootstrap 95% CI for absolute improvement |
|---|---:|---:|---:|---:|
| Random → true nuisance boundaries | 0.020381 | 0.010826 | 46.9% | `[0.007574, 0.011615]` |
| Repeated → independent support | 0.016726 | 0.003529 | 78.9% | `[0.012084, 0.014382]` |
| Mixed → correct single operator | 0.063662 | 0.003529 | 94.5% | `[0.059668, 0.060564]` |

The maximum parameter difference across different partitions of the exact same
no-nuisance pixel pool is `0.0`, satisfying the `1e-12` equivalence tolerance.
All three paired controls exceed the frozen 10% relative-improvement floor and
have bootstrap lower bounds above zero.

### Non-promotion diagnostics

- Shifting the neutral prior by `[+0.035, -0.025, +0.020]` raises mean error
  from `0.003716` to `0.028871`. The nominal prior improves by 87.1%, with
  absolute-improvement CI `[0.024660, 0.025617]`. This is strong prior
  sensitivity, not success.
- When a scanner affine is composed after the film affine, the estimate matches
  the composite (`0.003990`) much better than the film component (`0.019971`).
  The 80.0% relative difference, CI `[0.015475, 0.016411]`, directly confirms
  that target-only observations recover a composite look unless scanner/process
  are crossed or otherwise known.

## Decision and claim boundary

`method_control_decision=pass` means:

- the fixed-budget harness behaves correctly when frame partition is
  uninformative;
- true nuisance boundaries can carry useful information;
- independent colour support, not the word “roll,” improves estimation;
- mixed operators remain a strong misspecification control.

`roll_information_decision=not_established` means:

- no physical roll group has been tested;
- no correct-roll versus matched wrong-roll comparison exists;
- the source prior remains assumed and materially affects the answer;
- scanner/process/film remain non-identifiable as separate components;
- affine truth does not establish nonlinear film-style recovery.

The highest honest claim is therefore:

> Fixed-budget affine controls validate grouped-target estimation mechanics and
> show that true nuisance boundaries and independent support can add information.
> They do not establish special physical-roll information.

## Next allowed work

1. Freeze the local FilmSet manifest and pair-blind/internal-lockbox contract;
   this does not require a physical-roll claim.
2. Implement L2 monotone-spline truth and recovery controls before promoting
   nonlinear colour-transfer claims.
3. Add stronger WB/scene/scanner/prior-swap factorial diagnostics.
4. Build matched-strength deterministic and classical FilmSet baselines on the
   4,657 internal identities; keep the official 628 closed.
5. Use BlueNeg only after its whole-roll split and shortcut controls are frozen;
   correct roll must beat same-film/date/location/scene-matched wrong rolls.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_roll2film_operators.py tests/test_roll2film_identification.py
.\.venv\Scripts\python.exe scripts/run_roll2film_e0_fixed_budget.py
```
