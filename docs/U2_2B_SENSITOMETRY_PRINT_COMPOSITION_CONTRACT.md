# U2.2B sensitometry-to-print composition contract

Date frozen: 2026-07-24  
Node: `ULT > U2.2B`  
Parent: `U2.2A` numerical pass

## Question

Can the new explicit `linear exposure -> layer density` primitive compose with
one existing clean-room `layer density -> print/display RGB` interpretation
without applying a second capture matrix or negative characteristic curve,
while remaining bounded, deterministic, non-affine and locally
orientation-preserving on `[0,1]`?

## Architecture

The composition is exactly:

```text
linear RGB exposure
  -> U2.2A RGBSensitometryOperator (once)
  -> dye absorption matrix
  -> negative transmission
  -> print exposure matrix
  -> paper characteristic curve
  -> print reflectance
  -> explicit endpoint normalization
```

The print stage reuses the already frozen clean-room
`cyan_shadow_warm_highlight_like` dye/print/paper parameters from U5.R2E0. Its
capture matrix and negative logistic parameters are forbidden because U2.2A
already owns those semantics. This is reuse of a project-owned descriptive
witness, not stock evidence.

The print interpretation records the layer-density black/white reference
vectors used for normalization. It rejects density outside the componentwise
reference interval rather than clipping. Output normalization may tolerate
only `1e-12` numerical endpoint roundoff; larger escape is an error.

## Frozen numerical audit

Use a 17^3 grid, 4,096 seeded random probes and central finite-difference
Jacobians on the 15^3 interior grid. Require:

- black/white endpoints exact within `1e-12`;
- every output finite in `[0,1]` within `1e-12`, with no clipping;
- full/partitioned execution byte-exact;
- canonical JSON replay byte-exact;
- minimum finite-difference directional derivative `>=0`;
- minimum finite-difference Jacobian determinant `>0`;
- identity RGB RMSE `>=0.05`;
- best affine residual RGB RMSE `>=0.02`;
- source input unchanged;
- out-of-reference density and out-of-domain RGB reject;
- two complete result hashes identical.

The style/non-affine floors establish only that the composition is a meaningful
algorithm witness rather than identity/basic affine. They do not establish
appeal, film plausibility or artifact safety.

## Engineering boundary

Factor the existing density-to-print math into one reusable function without
changing any U5.R2E0 output. Add a separate interpretation/composition class;
do not retrofit production rendering, profiles or existing frozen configs.
All existing density-domain tests and exact witness hashes must remain valid.

## Decisions

- **Pass:** U2.2 numerical representation closes and U2.3 explicit global
  residual evaluation may be frozen. Product integration and stock claims stay
  closed.
- **Fail:** retain U2.2A intermediate only; do not add capacity or tune the
  print witness after results.
- **Invalid:** repair only implementation/parity/provenance defects and rerun
  unchanged gates.

## Claim ceiling

Uncalibrated clean-room numerical composition of one exposure-to-density curve
bank and one explicit density-to-print interpretation.

## Pre-result implementation note

The first audit attempt wrote no report because the existing U5.R2E0 finite-
difference helper always passes a `strength` keyword, while this composition
intentionally has no strength control. The evaluator now applies the identical
central-difference formula directly without changing step, probes, operator or
gates; neither public operator API was widened.
