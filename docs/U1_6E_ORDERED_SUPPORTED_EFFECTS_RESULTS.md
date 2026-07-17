# U1.6E Ordered Supported-Effects Results

**Date:** 2026-07-17

**Node:** `ULT > U1.6 > U1.6E`

**Decision:** **closed at the observable-order gate** — retain no combined
adapter and do not weaken the frozen gate.

## Question and development probe

The frozen contract required a combined simple-halation then sparse-dust pass
to preserve renderer order and required reversed active layers to be observably
different. During development, an irregular `97 x 137 x 3` float32 image used:

- simple-halation strength `0.8`, threshold `0.35`, edge threshold `0.01`;
- dust strength `1.0`, seed `15`;
- output margin `4`.

The tiled candidate matched the forward reference within the frozen `1e-6`
tolerance. However, reversing dust and halation changed the result by only
`1.1920928955078125e-07`, also inside that same equivalence tolerance. The
required order discriminator therefore failed.

## Why the result is structural

For one channel, white alpha dust with opacity `a` is:

`D_a(x) = 1 - (1 - x)(1 - a)`.

The current screen halation with effective colour/alpha factor `q` is:

`H_q(x) = 1 - (1 - x)(1 - q)`.

Therefore:

`D_a(H_q(x)) = H_q(D_a(x)) = 1 - (1 - x)(1 - a)(1 - q)`.

The two current layer families commute algebraically per pixel. Only float32
rounding and clipping order create the observed sub-tolerance difference. An
output-only test cannot prove their order without lowering the equivalence
tolerance or changing effect math, both forbidden after the contract freeze.

## Branch action

- the uncommitted combined implementation and its tests were removed;
- U1.6C and U1.6D remain valid independent numerical passes;
- no existing renderer, effect, compositor, CLI, profile or recipe changed;
- no threshold was lowered and no artificial non-commutative marker was added;
- a later orchestration layer may record declared order as provenance, but
  cannot claim output-identifiable order for these two current operators;
- U1.6 continues through a different child, with grain/global context the next
  candidate.

## Verification and claim boundary

After removing the failed candidate, the focused established U1.6C/U1.6D
suite passes **44 tests** and the complete CPU suite passes **328 tests**.
The branch remains clean except for this evidence propagation.

This result does not show a visual defect or renderer regression. It shows that
the preregistered order-identifiability requirement is incompatible with the
current compositor algebra. It opens no integration, realism, stock,
calibration, streaming or memory claim.
