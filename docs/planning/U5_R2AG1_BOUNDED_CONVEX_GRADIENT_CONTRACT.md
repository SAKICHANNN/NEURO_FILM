# U5.R2AG1 — Bounded convex-gradient representation contract

Date: 2026-07-28

## Parent and purpose

AG0 derives a project-owned analytic map with exact cube and positive
orientation guarantees. AG1 tests whether its fixed 16-anchor/65-scalar form
has enough capacity to reproduce the same two nonlinear paired synthetic
controls already passed by O0.

O0 remains the retained safe representation. AG1 is allowed only because it
is substantially smaller, analytic and palette-native.

## Frozen operator

For anchor rows `a_k`, centered biases `b_k`, fixed temperature `tau=.15`
and strength `0<=alpha<=.65`:

```text
p_k(x) = softmax((a_k dot x + b_k) / tau)
T(x)   = (1-alpha)x + alpha sum_k p_k(x)a_k
J_T(x) = (1-alpha)I + alpha/tau Cov_p(a)
```

Sixteen initial anchors are fixed in the config. Anchor components use
sigmoids, biases use bounded tanh followed by exact mean centering, and
strength uses `.65*sigmoid`. Identity is a separate exact `alpha=0` instance.
No affine or rotational wrapper is allowed.

The family has exactly 65 fitted scalars. Its output is in cube and its
Jacobian is symmetric positive definite by construction. The frozen minimum
eigenvalue/determinant floors are `.35/.042875`.

## Data and fitting

- working space: linear sRGB D65;
- development grid: `8^3`;
- untouched confirmation grid: `11^3`;
- Jacobian audit grid: `7^3`;
- targets: exact O0 identity, density-cyan `s0.50` and positive-warm `s0.35`;
- deterministic CPU float64 Adam, one thread, seed 29042, 2,500 steps;
- no image, external code, film pixel, unpaired distribution or GPU.

The optimizer sees only development pairs. Confirmation and gate metrics are
computed after parameters freeze.

## Gates

Every target must pass the same substantive O0 gates:

- identity error at most `1e-12`;
- exact `[0,1]` range without output clamp;
- confirmation RGB RMSE at most `.015`;
- at least 20% improvement over the fitted global affine baseline;
- analytic minimum eigenvalue/determinant at least `.35/.042875`;
- analytic maximum Jacobian spectral norm at most `8`;
- inverse roundtrip error at most `2e-5`;
- exact serialization and arbitrary-row partition parity;
- two byte-identical complete runs.

The parameter count, strength and centered-bias bounds are additional
conjunctive gates.

## Inverse

The forward map is primary. A fixed damped Newton audit uses the exact SPD
Jacobian, at most 40 iterations, residual tolerance `1e-12` and a frozen
minimum line-search scale. It is an evaluator, not a rendering dependency.

## Branches

- A faithful implementation that violates analytic range or SPD bounds is
  invalid and must be repaired before reading capacity results.
- Fidelity, affine-gain, inverse or exactness failure closes without more
  anchors, lower temperature, affine wrapper, optimizer search or gate change.
- A full pass retains compact synthetic representation feasibility only.
  It cannot trigger images or unpaired fitting.

No outcome identifies a film transform, opens stock/LSM learning, or changes
the O0 or AF1 historical results.
