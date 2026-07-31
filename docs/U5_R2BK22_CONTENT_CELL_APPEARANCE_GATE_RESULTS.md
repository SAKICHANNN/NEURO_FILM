# U5.R2BK22 Content-Cell Appearance Gate Results

Date: 2026-07-31  
Decision: `retain_content_cell_worst_case_gate_prior`

## Outcome

BK22 kept the BK21 flow, losses and hard-selection semantics fixed. It added
only a worst-case validation gate across four externally supplied generated
content cells.

The valid and confounded views contain exactly the same pooled source and
target colour multisets. The confounded view only rotates target-cell
correspondence by `[1,2,3,0]`. Pooled colour matching therefore cannot
distinguish them.

Two complete CPU float32 runs are byte-identical at
`677dfbd2467ad1857e14923fe242ffbbff0023903d9d661d5ffe5462eb91c73c`.

## Evidence

The two canonicalizers agree at `0.01094` full-cube RGB RMSE and pooled
validation hard-selects the RFF-MMD candidate.

| View | Cell improvements | Worst cell | Policy |
|---|---|---:|---|
| valid | 96.29%, 92.99%, 94.28%, 96.27% | 92.99% | hard explicit candidate |
| cell-shuffled confound | 46.68%, 15.53%, 5.25%, 54.43% | 5.25% | identity fallback |

The selected candidate also passes all structural gates: output range
`[0.05082, 0.96089]`, minimum Jacobian determinant `0.20469`, maximum
Jacobian norm `1.90367`, inverse error `1.10e-9`, and exact replay.

## Boundary

This pass establishes a mechanism, not available real-data evidence. The cell
IDs were generated latent scene identities exposed only to the evaluator.
They were not inferred from RGB, embeddings, uploader, source or scanner.

The result says that independent connectivity can veto a pooled appearance
shortcut when such connectivity is known. It does not provide a legitimate
way to manufacture content cells from current film photographs, identify a
digital-to-film operator, prove a stock mode, or train a router.

The next legal leaf is a metadata/rights/connectivity feasibility audit for
externally observed same-look/different-content cells. If none exist, the
correct product policy remains the global look or identity fallback.
