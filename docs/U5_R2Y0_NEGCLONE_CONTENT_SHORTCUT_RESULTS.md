# U5.R2Y0 NegClone Content-Shortcut Results

Date: 2026-07-27

Decision: **content, histogram and texture shortcuts reproduced; stock-learning
branch closed**

## Reproducibility

- software commit:
  `a824be87b83d2c16930d7b6a8f6a931aa1404a0a`;
- config SHA-256:
  `4be7cca20d58194e889a1de5bf38180324f0afba865e1648c7ff646a7c263763`;
- exact PyPI source archive SHA-256:
  `4672f11cf98b9d037debd0eab60a3ee9d278a73515b8184d7223b976d6a46387`;
- both formal reports are byte-identical at
  `187dda30d1f2af443220444878cf52e2da61fc97b8e2c8723cf69ba9762b2212`;
- both stderr logs are empty.

No network request, external image payload, current stock pixel, generated
stock preset or visual shortlist was used.

## Published estimator boundary

The hash-pinned NegClone 0.2.0 source directly shows:

- colour is computed from within-image luminance percentile regions and their
  mean RGB;
- tone is computed from the scene luminance histogram/CDF and PCHIP;
- grain samples arbitrary image patches and measures standard deviation,
  autocorrelation and FFT without separating film grain from scene texture;
- multiple images are aggregated by medians;
- public image and grain-patch sampling use the global `random` module without
  a seed parameter;
- scanner offsets are described as approximate;
- the Lightroom output contains a tone curve, three-way colour grading,
  shadow tint and grain controls, not a 3D LUT.

Taking a median can reject isolated outliers. It cannot identify and subtract
a nuisance factor that systematically changes with content, exposure, source
or texture.

## Film-free counterfactuals

All inputs below are generated arrays with no film response.

| Probe | Result | Frozen gate |
|---|---:|---:|
| red-flat versus blue-flat midtone bias L2 | `0.848528` | at least `.5` |
| dark versus bright neutral-gradient curve difference | `254.4 / 255` | at least `100 / 255` |
| flat-grey grain intensity | `0` | at most `1e-12` |
| checker-texture grain intensity | `.4` | at least `.2` |
| checker-texture clumping | `1.0` | at least `.9` |

The red and blue arrays yield `[.4,-.2,-.2]` and `[-.2,-.2,.4]` alleged
midtone biases. Moving the same neutral gradient from `[0,.5]` to `[.5,1]`
changes the exported tone curve without adding a stock. A deterministic
grey checkerboard becomes strong grain while a flat grey image becomes none.

These are direct content, exposure-histogram and texture shortcuts, not a
subtle loss of statistical power.

## Project decision

The branch is `content_histogram_texture_shortcut_close`.

NegClone is retained only as a useful negative control for the exact failure
mode the Ultimate architecture must avoid: averaging uncontrolled photographs
and naming their scene statistics a stock fingerprint. More images, a median,
a neural encoder or preset tuning do not open as rescues. The result does not
claim that every carefully controlled use of the package must fail; it proves
that the published estimator itself does not identify stock evidence under
the tested nuisance interventions.

No film/stock operator, calibrated scanner response, product visual candidate,
current-pixel fitting, training or LSM permission opens.
