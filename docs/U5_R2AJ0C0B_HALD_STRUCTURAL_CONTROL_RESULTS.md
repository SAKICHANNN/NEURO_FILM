# U5.R2AJ0C0B Hald structural-control results

## Decision

`U5.R2AJ0C0B` passes its corrected, frozen controls-only contract. Two new
processes at implementation commit `d8fdcb1ccccf58dba2cf4570f542a31af91a0ac2`
produce byte-identical manifests and reports, and the parent process
independently reconstructs the complete canonical evidence bytes.

This is a parser, interpolation, metric and evidence conformance result. It
does not measure an external CLUT. The archive and every primary Hald body
remain unopened by this node.

## Exact evidence

| Field | Value |
|---|---|
| config raw SHA-256 | `46870c84...f35cf` |
| config canonical SHA-256 | `cbb25c54...a19e` |
| run A/B manifest SHA-256 | `745a0596...e9f7` |
| run A/B report SHA-256 | `0d3a466a...7d8a` |
| repeat decision SHA-256 | `7b3d94f0...e05` |
| full CPU suite | `1086 passed` |

Every repeat check passes: manifest bytes, report bytes, identities, child
control expectations and independent strict reconstruction are exact.

## Control conclusions

- legal Hald `(level, cube side, image side)` geometries
  `(2,4,8)`, `(6,36,216)`, `(12,144,1728)` and `(16,256,4096)` pass parser,
  raster and identity-interpolation checks;
- analytic identity and axis-swap Jacobians pass exactly; the independent
  high-precision finite-difference oracle has maximum absolute error
  `3.33e-16` and maximum relative error `1.02e-14`;
- the generated RGB16 TIFF fixture decodes with exact samples, shape, dtype
  and pinned TIFF tag facts;
- all four structural negatives are rejected;
- the warm `0.50/0.75/1.00` controls and its exact duplicate collapse into
  one permutation-invariant strength component;
- the cool `1.00` control remains a separate component;
- post-collapse warm-versus-cool residual novelty is median Delta E76
  `8.728775`.

## Zero-access boundary

The exact access ledger is zero for archive opens/bytes, primary opens,
primary decodes, primary metrics, external-root control pixels and photograph
renders. The implementation API accepts no archive path. The official runner
also requires its declared software commit to equal repository `HEAD` before
creating evidence.

## Branch

The pass opens only design of a separately frozen `U5.R2AJ0C1` contract.
Execution remains closed until that contract commits the full 194-member
universe, native-grid streaming, per-candidate vetoes, strength/duplicate
collapse, representative policy and deterministic cap-12 diversity selection.

No photograph or aesthetic review is allowed yet. No result establishes
preference, photographic severe-artifact safety, stock/process identity,
film authenticity, calibration, a digital-to-film operator, a training
target, product integration or permission to redistribute the external
assets.
