# U5.R2AD1 — spektrafilm spatial-DIR ablation results

## Decision

**Close the spatial-DIR route.** The frozen exact-float replay gate fails, and
the independently frozen isolated-red-speckle budget also fails. Visual review
is forbidden; no spatial-on image was opened.

## Reproducibility and lineage

Two complete isolated runs used spektrafilm revision
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`, CPython 3.13.14 and the exact
RF2.C0 core-package versions. The tracked evaluator ran from
`80cfc0e2d91d836caa7a0ef0fcd910ad42f6cd13`.

- config SHA-256: `d669bb46...a19d8`;
- manifest A SHA-256: `504b12c4...0e52f`;
- manifest B SHA-256: `fb798a1f...6ddf8`;
- formal report SHA-256: `241c7ac3...99de`;
- normalized payload SHA-256: `b617cc48...ee5ce`.

Running the evaluator twice produced byte-identical formal reports.

All nine spatial-off PNGs exactly reproduce the retained RF2.C0 hashes. Both
runs also produce identical spatial-on PNGs. The spatial-on float64 arrays,
however, differ in every sample, with maximum absolute differences from
`4.66e-15` to `7.61e-15`. The external spatial filtering path is therefore
visually/quantization stable but not byte-exact at the frozen float boundary.
The contract required exact complete-run replay, so the primary decision is
`close_runtime_or_replay_failure`.

## Automatic metrics

The mechanism is not bland:

| Metric | Result | Frozen gate |
|---|---:|---:|
| median spatial effect Delta E76 | 0.5458 | 0.25–4.0 |
| median residual after local-contrast match | 0.5696 | >= 0.1 |
| worst simple local-contrast explained energy | 77.60% | <= 90% |
| median style Delta E76 vs input | 8.1852 | >= 7.0 |
| median non-basic residual vs input | 7.3924 | >= 4.9 |
| worst new hard clipping | 0.0000% | <= 0.5% |
| worst luma-gradient p99 amplification | 1.1609x | <= 1.5x |
| worst chroma-HF p99 amplification | 1.2402x | <= 1.5x |
| worst new isolated-red-speckle fraction | 0.1481% | <= 0.1% |

The paired red-speckle detector fails on sample 29 at 0.1481%; sample 08 also
exceeds the gate at 0.1332%. This is a separate automatic failure, not an
interpretation of physical film texture. The frozen gate exists to prevent a
strong local colour mechanism from recreating the project's known red-speckle
regression.

## Branch boundary

The route closes without:

- changing FFT/thread settings to seek a passing float hash;
- weakening exact replay or red-speckle limits;
- reducing diffusion strength, changing its tail or adding smoothing/clamps;
- opening the images for post-result visual exception;
- creating AD2, a parameter grid, a learned mask or a neural rescue;
- copying external GPL/CC-BY-SA code or profiles;
- fitting/training from outputs or claiming an Ektar response.

Useful negative evidence remains: source-default spatial DIR diffusion creates
a measurable local, non-basic effect beyond a simple Gaussian local-contrast
control, but the exact reproducibility and severe-artifact-prevention contract
rejects it. Ultimate continues through a distinct algorithm/data leaf.
