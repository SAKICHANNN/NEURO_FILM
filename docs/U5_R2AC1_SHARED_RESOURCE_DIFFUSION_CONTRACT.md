# U5.R2AC1 — Shared-resource diffusion representation contract

## Purpose and independence

Test whether a newly defined, dimensionless shared-resource reaction/diffusion
system can create bounded local tone interaction that is absent from global
colour operators.

This is not a Filmulator port. No Filmulator code, constants, defaults,
discretization or parameter names may be used. The model is defined entirely
by this contract.

## Fixed state and update

For a linear-RGB activation field `A` in `[0,1]`, initialize:

- three per-channel resources `S = 1`;
- one shared scalar resource `D = 1`;
- three-channel deposit `P = 0`.

For 24 float64 steps with `dt = 0.04`:

1. `R = 0.8 * A * S * D * dt`
2. `P += R`
3. `S -= R`
4. `D -= 0.6 * mean_channel(R)`
5. apply one reflect-boundary five-point heat step to `D` with Courant
   coefficient `0.12`;
6. mix `D` toward a unit reservoir by `0.02`.

Every update must remain in `[0,1]` from its own bounded equations; no hidden
output clamp or postprocessing is allowed. Output strength one returns `P`;
strength zero must return the input exactly.

The finite dependency radius is at most one pixel per step, hence 24 pixels.

## Frozen population and controls

Use 64×64 constant fields, centred squares of widths 1/5/17/33, a 0.1/0.8
step edge, a 0.02–0.98 ramp and isolated red/green/blue patches.

Controls:

- zero-strength identity;
- the same reaction with diffusion disabled independently per pixel;
- shared resource held at one;
- separate resource fields for each channel.

## Gates

All state/output values must be finite in `[0,1]`. Required invariants:

- identity error exactly zero;
- constant-field spatial span ≤ `1e-12`;
- neutral channel spread ≤ `1e-12`;
- 90° rotation error ≤ `1e-12`;
- exact repeat;
- increasing constant input changes output by at least `0.001`;
- local RMSE versus global control in `[0.002, 0.12]`;
- edge-gradient amplification ≤ `2.0`;
- flat-region ringing ≤ `0.01`;
- no response in an unactivated channel;
- finite support radius ≤ 24.

Failure closes the representation without timestep, clamp, smoothing,
boundary or parameter rescue. A pass opens only a separately frozen safety
contract; it does not authorize photographs, stock claims or production.
