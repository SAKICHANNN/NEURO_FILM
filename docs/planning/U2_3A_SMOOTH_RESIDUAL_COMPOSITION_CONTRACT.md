# U2.3A Smooth Residual Composition Contract

**Frozen:** 2026-07-24  
**Node:** `ULT > U2 > U2.3 > U2.3A`  
**Status:** frozen before implementation or result inspection

## Question

Can the passed U2.2B exposure-to-density-to-print primitive be composed with
one explicit, bounded, smooth tetrahedral 3D residual without changing the
base when the residual is identity, violating the RGB cube, losing local
orientation, or introducing non-replayable behavior?

This is a numerical representation experiment. It does not ask whether the
residual resembles a stock, improves a photograph, is visually safe, or should
enter the renderer. U5.R2A already validated the underlying tetrahedral
interpolator and constraint audit; U2.3A tests the previously untested
composition with the U2.2B base.

## Fixed candidates

The base is reconstructed exactly from the frozen U2.2B parent inputs. Two
residual LUTs are baked on the same 17-cube grid:

1. exact identity;
2. one clean-room cyclic residual, frozen as

   `x'_c = x_c + 0.1 * x_c * (1 - x_c) * (x_(c+1) - x_(c+2))`.

The cyclic residual is zero on the neutral axis and at the corresponding
channel endpoints. It is a numerical witness, not a fitted or claimed film
response. No parameter may be changed after the formal run.

## Automatic gates

- identity-residual composition equals the U2.2B base exactly;
- vectorized tetrahedral output matches the independent scalar reference;
- the residual LUT passes the frozen range, residual-amplitude, first- and
  second-difference, neutral-axis and tetrahedral-Jacobian constraints;
- composed output remains inside `[0, 1]` within the frozen tolerance;
- the composed sampled finite-difference Jacobian determinant is strictly
  positive;
- residual RMSE relative to the base is in `[0.002, 0.02]`;
- at least 90% of the base-versus-input RMSE remains, preventing a trivial
  cancellation of the existing look;
- serialization replay is exact, partitioning is exact, source arrays are not
  mutated, invalid inputs fail closed, and two reports are byte-identical.

## Branches

- **Pass:** retain the generic base-plus-residual representation and open a
  separately frozen real-image U2.3B visual/artifact frontier. Production and
  stock fitting remain closed.
- **Constraint or composition failure:** close this residual witness without
  retuning it; retain U2.2B and the already-passed standalone U5.R2A primitive.
- **Nontriviality or retention failure:** treat the residual as redundant or
  look-cancelling; do not add capacity to rescue it.

## Allowed and forbidden actions

Allowed: a small generic wrapper, focused property tests, deterministic
synthetic probes, an ignored report, and evidence propagation after the frozen
decision.

Forbidden: fitting current real-film pixels; changing training or operator
permissions; tuning after results; production/profile integration; calling the
witness a stock response; neural or generative RGB output; visual promotion
from numerical gates alone.

## Claim ceiling

At most: an uncalibrated clean-room numerical composition of a passed
sensitometry-print base and a bounded smooth tetrahedral residual. It is not an
identified digital-to-film operator, calibrated response, stock expert,
semantic-safety result, or product integration.
