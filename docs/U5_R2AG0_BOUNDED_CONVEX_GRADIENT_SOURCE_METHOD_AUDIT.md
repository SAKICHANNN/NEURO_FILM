# U5.R2AG0 — Bounded convex-gradient colour-map audit

Date: 2026-07-28
Decision: `open_clean_room_compact_representation_pilot`

## Research question

Can a substantially smaller, palette-native explicit colour operator make
RGB-cube range and positive orientation true by construction, while retaining
enough nonlinear capacity to challenge the already safe 192-parameter O0
diffeomorphic flow?

This is a representation question. It does not reopen unpaired operator
identification or claim that optimal transport recovers a film process.

## Primary evidence

The source audit used four ignored, exact PDFs:

| Source | Bytes | SHA-256 |
|---|---:|---|
| Amos et al., *Input Convex Neural Networks*, ICML 2017 | 412,672 | `aac084d9572254c784670db7dd825a14b3732d5b618d819a13d686821f9b9c2b` |
| Makkuva et al., *Optimal transport mapping via input convex neural networks*, ICML 2020 | 2,590,216 | `d9f166b0768d11a25d4c7c5c137465bc982842619c3555fae8501d5ae9093175` |
| Uscidda and Cuturi, *The Monge Gap*, ICML 2023 | 2,838,608 | `5e16aa49ba51aeb66f419538811ade5db977303537f5d2624f3baf0bc841858f` |
| Pitié, *Advances in colour transfer*, IET CV 2020 | 7,101,085 | `ebcb5154df02dce1f29faa45fe6e888bb9768ac3d08b7388327265bb40bc62e0` |

The literature supports three bounded conclusions:

1. input-convex networks can represent convex scalar potentials;
2. under the quadratic-cost density assumptions, the Brenier OT map is a
   gradient of a convex potential;
3. colour-distribution matching is not the same as recovering the real
   pixelwise grading operator. Many maps can match a palette, and content
   mismatch causes practical artefacts.

The Monge Gap paper also cautions that forcing the Brenier architecture on
finite samples introduces constraints and does not solve the general map
learning problem. Those limitations agree with the project's S2/S3 and
shared-author identifiability failures.

## Project-owned construction

AG0 does not copy an ICNN or OT solver. It synthesizes a small analytic
potential from the primary principles:

```text
phi(x) =
  (1-alpha)/2 * ||x||^2
  + alpha * tau * log sum_k exp((anchor_k dot x + bias_k)/tau)

T(x) = grad phi(x)
     = (1-alpha) * x + alpha * sum_k softmax_k(x) * anchor_k
```

with:

- `x` and every `anchor_k` in `[0,1]^3`;
- `0 <= alpha <= alpha_max < 1`;
- fixed `tau > 0`;
- bounded, mean-zero biases.

This gives exact structural properties:

- `T(x)` is a convex combination of in-cube values, so it remains in cube
  without output clipping;
- `J_T = (1-alpha)I + alpha/tau * Cov_p(anchor)` is symmetric positive
  definite;
- the minimum Jacobian eigenvalue is at least `1-alpha`;
- the determinant is at least `(1-alpha)^3`;
- identity is exact at `alpha=0`;
- the forward renderer is explicit and deterministic.

With 16 anchors, 16 biases and one strength, the family has 65 fitted scalars,
versus O0's 192 velocity coefficients and numerical ODE integration.

## Non-duplication and limitations

O0 is already the retained general safe representation. AG1 is justified only
as a compact analytic challenger with stronger closed-form bounds and a
palette-anchor interpretation. If it cannot fit both frozen O0 controls under
the same confirmation gates, it closes without capacity growth.

The symmetric positive-definite Jacobian forbids arbitrary local colour
rotation. The canonical quadratic-cost OT map is only one possible
distribution-matching map and is not a physical film operator. Anchor/bias
fitting from an unpaired film histogram would still be unidentified and is
forbidden.

## Branch decision

Open one clean-room, synthetic-only AG1:

- same identity, density-cyan `s0.50` and positive-warm `s0.35` targets used by
  O0;
- fixed 16 anchors, fixed temperature and bounded strength/bias;
- paired synthetic fitting only;
- exact analytic Jacobian/range bounds plus the existing confirmation,
  affine-relative, norm, inverse, replay and partition gates;
- no images, film pixels, unpaired distribution fit, ICNN, external code,
  generative model or production integration.

An all-gate pass retains only a simpler explicit representation. A failure
closes the construction without more anchors, lower temperature, asymmetric
Jacobian, affine wrapper or gate relaxation.
