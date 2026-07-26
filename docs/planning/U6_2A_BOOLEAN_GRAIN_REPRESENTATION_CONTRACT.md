# U6.2A — Boolean/Poisson Grain Representation Contract

**Status:** frozen before implementation or result inspection

**DRPT level:** L2

**Parent:** U6.0 heuristic baseline and U1.6F exact legacy-grain execution

**Primary writer:** current Codex Goal session

## Question

Can a clean-room inhomogeneous Boolean model provide a deterministic,
resolution-aware explicit grain representation whose mean and stochastic
structure follow the model's physical invariants, without copying GPL code or
claiming measured film calibration?

U6.2A is a synthetic representation audit. It does not complete U6.2, whose
stock/scanner NPS, autocorrelation and repeat-scan validation remain gated on
real evidence.

## Primary-source and licence boundary

Newson et al. model film grain as the union of random disks whose centres follow
an inhomogeneous Poisson process. For local normalized intensity `u` and
constant radius `r`, the intensity is:

```text
lambda = -log(1-u) / (pi*r^2)
```

This makes the probability of Boolean coverage equal to `u` in a homogeneous
region. The filtered output averages binary coverage over Gaussian-shifted
Monte Carlo evaluations and can be rendered at arbitrary output zoom.

The article is CC-BY-NC-SA and its reference implementation is GPL-3.0-or-later.
No reference code, constants, random generator, data structure or optimization
is copied. Only the published model equations and broad grain-wise rendering
description are used for an original Python implementation. The temporary
low-resolution paper PDF is pinned by SHA-256 in config and is not added to the
repository.

## Frozen clean-room candidate

For each input pixel:

1. cap validated intensity at `255/256`;
2. compute `lambda` from the equation above;
3. sample `Q ~ Poisson(lambda)` with the frozen PCG64 seed;
4. sample `Q` centres uniformly inside the pixel;
5. use one constant radius for the complete run.

For each of 32 frozen Gaussian offsets, transform grain centres and radii to
the output zoom, rasterize binary disk coverage, and average the 32 masks.
Outputs are floating coverage in `[0,1]`; no clamp is allowed. Grain context
and offsets are serializable. Region rendering uses the same global coordinates
and context, so arbitrary row partitions must be exact.

Two fixed radius policies are audited:

- small: `.22` input pixels;
- large: `.38` input pixels.

These are synthetic research settings, not estimates of a film stock.

## DoR

- U1.6F remains immutable and is not replaced or retuned.
- formula, radii, seed, shape, intensity levels, Monte Carlo count, zoom,
  partitions and gates are committed before code.
- no real image, film scan, stock label, preference target or neural model is
  accessed.
- the implementation must be isolated under `src/filmfx/` and must not enter
  renderer/profile/CLI paths.

## Frozen audit

Render `24x24` constant fields at `u=.1,.5,.9`, output zoom 4. Metrics exclude
three input pixels at every edge.

Required evidence:

- all output values are finite and in `[0,1]`;
- each flat-field interior mean is within `.04` of its input level;
- midtone variance is at least `1.4x` the larger endpoint variance;
- mean horizontal/vertical lag-1 autocorrelation for large grains exceeds
  small grains by at least `.03`;
- grain counts are nonzero and at most 100,000;
- repeat output, arbitrary row-partition assembly and serialization replay are
  bit-exact;
- two formal reports are byte-identical.

## Branches

- **All gates pass:** retain the primitive and freeze a separate bounded
  existing-image crop visual/style/severe frontier.
- **Mean fails:** close; do not renormalize or force-match the output mean.
- **Variance or autocorrelation fails:** close; do not tune radii, offsets,
  Monte Carlo count or metric crop.
- **Determinism/partition fails:** close; do not cache a hidden full output.
- **Later visual severe failure:** veto regardless of physical motivation.

## DoD

- isolated explicit context/renderer/serialization module;
- formula, mean, correlation, determinism, partition and failure tests;
- two formal reports;
- full CPU suite;
- result/decision propagation;
- scoped commits and pushes.

## Claim ceiling

Clean-room data-independent numerical and stochastic-structure evidence for a
deterministic explicit Boolean/Poisson grain representation. No copied GPL
code, measured NPS, real-film calibration, stock/process/scanner identity,
preference, production or universal realism claim.
