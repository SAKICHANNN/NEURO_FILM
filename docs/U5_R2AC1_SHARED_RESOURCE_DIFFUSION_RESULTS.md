# U5.R2AC1 — Shared-resource diffusion results

## Decision

**`effect_too_weak`; close the fixed representation.**

Two reports from software
`df4b170d1837f120462e214855bc9ecb539c0de1` are byte-identical at SHA-256
`839aadd43009cc218cb6ff5e7c76c39d6500290c00590d45037fb1273c12dca2`.
The frozen config hash is
`5766ba680bad132ded878ec6f2109d23305d3dd1011e04eed3f51cac8991da12`;
payload hash is
`83b56bc682016acef2636c7511419b2f5a65b55f9e859dfe2b4b5adeecbc7d57`.

## Structural result

Every structural and artifact-prevention check passes:

- complete state/output range: `[0,1]`;
- exact zero-strength identity;
- exact constant-field uniformity and neutral axis;
- 90° rotation error: `2.22e-16`;
- exact repeat;
- minimum constant-field monotone delta: `0.08826`;
- edge-gradient amplification: `0.5185`;
- flat-region ringing: `0`;
- inactive-channel crosstalk: `0`;
- finite support radius: 24 pixels.

The implementation is therefore deterministic, bounded and structurally
clean at the one frozen policy.

## Independent-value failure

The largest RMSE between the spatial-diffusion result and the same
reaction/resource model with diffusion disabled is:

`0.001946755301184502`

The preregistered minimum was `0.002`. The shortfall is about `0.0000532`.
The 33-pixel square is the strongest pattern; the smooth ramp reaches only
`0.0000821`.

Because the contract froze the threshold and explicitly prohibited changing
steps, timestep, diffusion, reservoir, boundaries or postprocessing after the
result, the near miss is still a failure. Increasing the effect now would be
parameter rescue.

## Consequence

AC2 and photograph rendering do not open. Do not port Filmulator, retune this
system, add a clamp or learned mask, or call the generic mechanism a stock
response. The result remains useful negative evidence: a safe shared-resource
spatial model can collapse to almost the same result as its per-pixel global
control.

Ultimate continues through a distinct evidence-authorized algorithm or data
leaf.
