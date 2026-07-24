# Roll2Film CT1 L0, gauge and shaper closure contract

Date frozen: 2026-07-24  
Node: `ULT > U5.CT1C`  
Parent: `U5.CT1` explicit invertible operator contract

## Purpose

Close the three explicit CT1 gaps left after the L1 affine and L2 monotone
spline core passed:

1. a canonical L0 exposure/white-balance operator;
2. an explicit within-roll nuisance gauge;
3. a reversible linear/HDR input shaper and shaped dense-LUT bake.

This is infrastructure and known-truth numerical evidence. It does not reopen
the failed physical-roll hypothesis, fit film pixels or change any production
renderer.

## L0 operator

The only allowed parameterisation is

```text
T(x) = exp(e) * diag(exp(w_R), exp(w_G), exp(w_B)) * x
sum(w) = 0
```

`e` is scalar log exposure and `w` is log white balance with unit geometric
mean. Forward, inverse, diagonal Jacobian/determinant, identity and canonical
JSON replay are required. There is no bias, clipping, quantisation or hidden
transfer function.

## Gauge

For a roll of raw per-frame log exposure and three-channel log-WB values:

- each WB row is centred to sum zero;
- the removed WB row mean moves into that frame's exposure;
- mean frame exposure is centred to zero;
- the removed roll mean is returned explicitly as shared log exposure that
  belongs to the base/global operator.

Recomposing centred frame nuisance with the shared offset must reproduce every
original per-channel log gain. Empty, nonfinite or incorrectly shaped inputs
fail closed. This utility does not infer the gauge from image appearance.

## Linear/HDR shaper

The shaper is a bounded monotone `log1p` mapping from declared nonnegative
linear domain `[0, domain_max]` to `[0, 1]`:

```text
u = log1p(compression * x) / log1p(compression * domain_max)
```

It has an analytic inverse and derivative. Values outside its declared domain
fail closed. A shaped LUT bundle applies `shaper -> dense 3D LUT`; the LUT is
baked by uniformly sampling shaper coordinates, analytically returning to
linear RGB, and evaluating the explicit operator once. Output stays in the
operator's declared linear working space and is never implicitly clamped.

## Frozen known-truth audit

The audit uses only deterministic generated values and the already tested CT1
L2 operator family. It performs two hash-identical repeats and requires:

- L0 forward/inverse maximum absolute error `<=1e-12`;
- L0 JSON replay byte-exact output;
- positive analytic Jacobian determinant;
- gauge row-WB sums, roll exposure mean and recomposition error `<=1e-12`;
- shaper roundtrip maximum absolute error `<=1e-12`;
- strictly positive shaper derivative;
- shaped 33-cube maximum RGB error `<=0.02` on the fixed HDR probes;
- shaped 65-cube maximum RGB error `<=0.006`;
- 65-cube maximum error `<=45%` of 33-cube error;
- shaped bundle JSON replay byte-exact output;
- out-of-domain values fail closed.

The fixed HDR domain is `[0, 16]`, shaper compression is `4`, the seed is
`20260724`, and the audit uses 8,192 probes. Thresholds may be repaired only if
the preregistered audit is mathematically impossible because of a documented
implementation-contract contradiction, before a valid result exists.

## Allowed files

- `src/roll2film/photometric.py`;
- `src/roll2film/lut.py` additions that preserve existing APIs;
- isolated CT1 audit script/config/tests;
- CT1 evidence and governance propagation.

Production renderer, stock data, frozen experiment configs and existing
operator semantics are forbidden from modification.

## Decisions

- **Pass:** CT1 explicit L0/gauge/shaper contract closes. This opens only use
  of these primitives by future independently gated operator experiments.
- **Fail:** retain the existing L1/L2 core and close the failed primitive; do
  not reinterpret the result as film or roll evidence.
- **Invalid:** implementation/provenance failure; repair only the contract
  violation and rerun unchanged gates.

## Claim ceiling

Deterministic numerical closure of L0 photometric, roll-gauge and linear/HDR
shaped-LUT primitives for explicit colour-operator research.

