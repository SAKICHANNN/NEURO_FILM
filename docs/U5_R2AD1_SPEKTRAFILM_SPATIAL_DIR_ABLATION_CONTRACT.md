# U5.R2AD1 — spektrafilm spatial-DIR ablation contract

## Question

Does the pinned external spatial diffusion of DIR inhibitor fields add a
stable, non-basic and artifact-safe local response beyond RF2.C0's retained
non-spatial Ektar100/fixed-e0 control?

This is an external mechanism ablation, not stock learning. It cannot identify
an Ektar response or a digital-to-film operator.

## Frozen source, runtime and images

- spektrafilm revision
  `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`;
- isolated CPython 3.13.14 and the exact RF2.C0 package versions;
- the nine frozen gold display-sRGB proxy inputs at maximum long edge 1024;
- Ektar100 negative to Portra Endura, fixed exposure zero;
- direct spectral path with LUTs, grain, halation, glare, optical diffusion,
  lens blur, scanner sharpening and stochastic effects disabled.

The external code/profile licences remain outside the project. No tracked
external source, profile or LUT is allowed.

## Only experimental difference

Both arms retain the same non-spatial DIR coupler correction.

- `spatial_off`: inhibitor diffusion core is zero;
- `spatial_on`: core 20 micrometres, tail 200 micrometres, tail weight 0.06.

These are source defaults frozen before metrics, not calibrated chemistry.
The spatial-off PNGs must exactly reproduce all nine retained RF2.C0 hashes.

## Nuisance control

For each image, compare spatial-on with a simple local-contrast family derived
only from spatial-off:

`control = off + alpha * (off - Gaussian(off, sigma))`

Use reflect boundaries, sigma in `{0.5, 1, 2, 4}` pixels, and the analytic
least-squares alpha clipped to `[-1, 1]`. Choose the lowest RGB-MSE member.
This is a nuisance explanation, never a film candidate or fitted operator.

## Automatic gates

Two complete runs must be hash-identical. All outputs must be finite and in
`[0,1]`. The spatial-on arm must:

- have gold-median spatial effect Delta E76 in `[0.25, 4.0]`;
- retain at least `0.1` median Delta E76 after the matched local-contrast
  nuisance control;
- allow the simple local-contrast control to explain at most 90% of paired
  RGB difference energy;
- add at most 0.5% worst per-channel hard clipping relative to spatial-off;
- amplify 99th-percentile luma gradients by at most 1.5x;
- amplify 99th-percentile high-frequency chroma by at most 1.5x;
- add at most 0.1% isolated red-speckle pixels;
- preserve RF2.C0's style floor of 7.0 Delta E76 and matched-basic residual
  floor of 4.9 Delta E76 relative to input.

No spatial-on image may be opened before these metrics and the branch decision
are written. Only an automatic pass opens a blind all-nine comparison and
full-resolution review of 09, 11 and `FS_FACE_01`.

## Branches

- Runtime, package or replay mismatch: close without revision substitution.
- Weak effect: close without diffusion/tail search.
- Simple local-contrast equivalence: close without capacity rescue.
- Clipping, gradient, chroma, speckle or severe visual failure: close without
  smoothing, clamping or strength reduction.
- Pass: retain one external comparison slot only.

No branch opens fitting, training, teacher use, stock truth, calibration,
production integration or a licensing decision. Ultimate remains active.
