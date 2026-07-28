# U5.R2AK1 — Time-Dependent Cube-Flow Capacity Contract

**Status:** frozen before implementation or target fitting

**DRPT level:** L2

**Parent:** U5.R2AK0 method-prior-only close + retained U5.R2O0 representation

**Primary writer:** current Codex Goal thread

## Question

Can a compact time-dependent, boundary-preserving explicit colour flow
represent two deliberately noncommuting colour-reaction stages more
efficiently than a stationary colour flow, while retaining cube range,
positive orientation, inverse accuracy and exact replay?

This is the one legal question inherited from the NCT audit. It does not
reimplement NCT, learn a reference image, map a photograph to a uniform latent,
or claim that a curved path identifies a film operator.

## Candidate

The candidate has three `3×3×3×3` velocity grids. At time \(t\), the grids are
mixed by the quadratic Bernstein basis:

\[
f(x,t)=(1-t)^2 f_0(x)+2t(1-t)f_1(x)+t^2f_2(x).
\]

The actual channel velocity remains:

\[
v_i(x,t)=x_i(1-x_i)f_i(x,t).
\]

The boundary factor makes each RGB-cube face invariant in continuous time.
The fixed float64 RK4 solver uses 32 steps. Reversing time and the integration
direction supplies the inverse. The 243 coefficients are explicit,
serializable and clipped only in parameter space to `[-6,6]`; output clamping
is forbidden.

This is not a neural final-RGB generator. No encoder is present.

## Controlled capacity comparison

Compare the 243-parameter time-dependent candidate to:

- stationary O0 `4³×3`, 192 parameters;
- stationary O0 `5³×3`, 375 parameters.

All three receive the same fit/confirmation points, optimizer steps, seed,
float64 CPU execution and parameter ceiling. The comparison asks whether
temporal structure is useful between a smaller and a larger stationary spatial
grid, not whether extra uncounted capacity wins.

## Synthetic noncommuting truth

Two fixed, analytic boundary-preserving source fields A and B are sampled on
one `4³` grid, scaled by 2.0 and integrated for 64 RK4 steps. Their formulas
are frozen in the config.

The two target operators are:

1. A then B;
2. B then A.

Their confirmation-grid RGB RMSE must differ by at least `.01` before fitting.
Otherwise the stated noncommuting mechanism was not instantiated and the
experiment is invalid.

These are original synthetic colour-reaction stages. They do not use a
photograph, film scan, Hald, owner anchor, retained output, external code,
checkpoint or teacher.

## Split and optimization

- Fit: untouched-for-confirmation `8³` midpoint grid.
- Confirmation: `11³` midpoint grid.
- Jacobian: separate `7³` interior grid.
- Optimizer: single-thread deterministic CPU float64 Adam.
- Seed: `280728`.
- Steps: `1,800`.
- Learning rate: `.03`.
- Coefficient L2: `1e-5`.
- Spatial smoothness: `1e-4`.
- Candidate-only temporal smoothness: `5e-5`.
- Gradient norm cap: `10`.

## Frozen gates

For both target orders:

- candidate confirmation RMSE at most `.008`;
- candidate improves at least 25% over stationary K4;
- candidate RMSE is no more than 1.10 times stationary K5;
- all outputs remain in `[0,1]` without clamp;
- cube corners fixed within `1e-12`;
- minimum finite-difference Jacobian determinant at least `.02`;
- maximum Jacobian spectral norm at most `8`;
- inverse roundtrip maximum error at most `2e-5`;
- maximum absolute coefficient at most `6`;
- serialization and arbitrary-row partition errors exactly zero.

Identity must remain exact within `1e-12`. Two complete formal reports must be
byte-identical and independently reconstructable.

## DoR

- AK0 source/method decision is immutable.
- O0 and AJ0C1 results remain immutable.
- This contract and config are committed before implementation or fitting.
- Target formulas, parameter counts, optimizer, split and gates are frozen.
- Tracked worktree must be clean for formal execution.

## Branches

- **All gates pass:** retain the representation for later evidence-eligible
  film-inspired operator work. Do not render the synthetic fitted controls on
  photographs.
- **Fidelity/efficiency fails:** close this fixed basis without adding grids,
  stages, steps, optimizer search or capacity rescue.
- **Regularity/range/inverse/repeat fails:** close without clamp,
  determinant repair, integrator or tolerance rescue.
- **Only one order passes:** record order-specific evidence; no general
  staged-reaction capacity claim.
- **Truth noncommutation fails:** invalidate before fitting.

## Claim ceiling

Clean-room synthetic representation evidence for one compact
time-dependent cube-preserving explicit colour flow. No NCT implementation,
image encoder, generative model, photograph, film-pixel fit, unpaired
identification, stock, process, calibration, preference or product claim.
