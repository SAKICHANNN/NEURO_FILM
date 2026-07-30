# Reference-Match External Baseline Audit

Date: 2026-07-27

Decision: **none of the currently obtainable published baselines may replace
the guarded local matcher or its identity delivery default**.

This audit asks a narrower question than paper leaderboard ranking: can a
published reference-photo colour-transfer implementation satisfy this
module's one-reference/N-source, bounded-output, repeatable-recipe and
cross-content known-look contract?

## Current primary-source frontier

| Method | Release state | Operator | Local status | Product boundary |
|---|---|---|---|---|
| StatLUT (2026) | paper; no official code or weight located | reference statistics to 16³ 3D LUT | strongest reported user-study result, not locally reproducible | architecture prior only |
| CanonCGT (CVPR 2026) | official code/weights; Apache-2.0 | source canonicalizer 17³ LUT plus reference-conditioned 17³ LUT | exact weight load and repeated local inference; fails frozen matrix | rejected as current product matcher |
| SA-LUT (ICCV 2025) | official code; non-commercial S-Lab 1.0 | spatial context plus 4D LUT | published inference weight is absent from repository | research-only and non-reproducible from release |
| NLUT (AAAI 2023) | official code/weight; MIT | source/reference-conditioned residual 33³ LUT plus test-time tuning | weight loads; shared-batch variants fail | comparison control only |
| Neural Preset (CVPR 2023) | metrics only; CC BY-NC-SA 4.0 | learned preset | no inference implementation or weight released | non-commercial paper/control only |

Reported paper preference is not treated as local evidence. Conversely, this
single frozen Velvia-look matrix does not reproduce any paper's complete
dataset or user study.

## CanonCGT asset and execution audit

- official repository commit:
  `229a7d3ec24af19ea5d97bb136f0317762bb9261`;
- repository and pretrained weights are explicitly Apache-2.0;
- official download bundle:
  SHA-256 `ab460d12...6cc51`;
- evaluated member `SSL_updated_251111.pth`:
  SHA-256 `5630c6dd...ad362`;
- all checkpoint keys load strictly into the 5,056,383-parameter model;
- inference uses a source-conditioned canonicalizer LUT and a
  source-plus-reference-conditioned restyler LUT, both 17³;
- official sample inference agrees with the bundled output to at most one
  8-bit code value, consistent with its image-save path.

The evaluator leaves official source and weights unchanged, uses targets only
after rendering, retains no rendered RGB, and runs twice byte-identically.
Both reports have ID `ec825926...e3efe` and SHA-256
`9b954501...e8a3f`.

### Published source-conditioned mode

| Gate statistic | Result |
|---|---:|
| Improved cross-content rows | 1/30 (3.33%) |
| Median improvement | -61.71% |
| Worst improvement | -157.12% |
| Maximum new boundary | 8.532% |
| Maximum raw pre-clip out-of-gamut | 9.447% |
| Maximum same-reference restyler-LUT source RMSE | .11725 |

The same reference does not produce one target LUT: source content changes the
restyler LUT materially. This is a valid published design choice, but it does
not satisfy this module's strong shared-grade interpretation.

### Fixed-reference-pivot innovation

The clean-room adaptation extracts the target LUT once:

```text
reference
  -> reference style token
  -> canonicalize reference
  -> fixed target 17³ LUT

each source
  -> source-specific canonicalization
  -> same fixed target LUT
```

This removes source dependence from the target LUT and improves numerical
containment, but does not recover the held-out known look:

| Gate statistic | Result |
|---|---:|
| Improved cross-content rows | 0/30 |
| Median improvement | -57.52% |
| Worst improvement | -118.81% |
| Maximum new boundary | 1.083% |
| Maximum raw pre-clip out-of-gamut | 4.269% |

The model also does not reconstruct its own references closely through this
fixed pivot (reference mean absolute errors roughly .022-.110), so that route
is closed rather than tuned.

## SA-LUT release mismatch

The repository's 286,604,792-byte checkpoint contains only
`style2vlognet.*`, discriminator and loss keys. That network is the frozen
image-to-image teacher used to synthesize pseudo V-Log during training.
The actual `vlog2stylenet` 4D-LUT/context-map weights required by inference are
absent. The CLI expects a different state layout not supplied in the release.

Independently, SA-LUT's spatial context map means identical RGB can map
differently by location/context. It belongs in a separately labeled local or
Creative mode even if a reproducible commercially licensed weight becomes
available; it cannot silently replace the global Style-safe batch path.

## NLUT bounded check

The official Google Drive checkpoint is 236,254,785 bytes at SHA-256
`b8c9bbb7...15c828`, loads all current model keys and contains a residual
33³ LUT generator. Fifteen unexpected keys belong to an older unused block.

A no-tuning source-batch/shared-LUT adaptation improves 5/30 rows, with median
improvement -59.50%, worst -171.59% and maximum new boundary 25.21%.
Applying the published 40-step test-time objective to one representative
five-source batch makes the median -100.13% and introduces up to 51.07% new
boundary pixels. The tuning route is therefore stopped without a full grid.

## Product decision

1. Keep the existing transactional module, replayable LookRecipe, composition
   binding and fail-closed identity default.
2. Keep CFSM and external neural methods behind research-only execution; none
   is promoted.
3. Retain the StatLUT-like spatially invariant Lab extractor as a future
   mapper input, but do not add capacity on the failed generated identity and
   replicate contract.
4. Treat exact-look copying from one arbitrary final photograph as
   underidentified. A future user-facing distinction must be explicit:
   `reference aesthetic approximation` accepts ambiguity; `exact look copy`
   requires aligned neutral/styled pairs or another independently identified
   canonical pivot.
5. Film simulation remains a downstream, separately identified stage. A
   reference photo may select or condition an aesthetic grade, but it never
   upgrades a film profile's evidence grade or authenticity claim.

CanonCGT remains a useful architecture and distillation control for
neuro-film: its canonicalizer/restyler split validates the module boundary,
and its explicit LUT output is compatible with eventual portable execution.
Its current weights do not validate this project's look recovery.
