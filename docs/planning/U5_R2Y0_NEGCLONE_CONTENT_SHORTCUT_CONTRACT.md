# U5.R2Y0 NegClone Content-Shortcut Audit Contract

Date: 2026-07-27

Status: **preregistered implementation frozen; formal repeat pending**

## Parent and question

The user hypothesizes that earlier ML/statistical methods became bland because
they averaged unrelated photographs rather than conditioning on a genuinely
similar case. NegClone 0.2.0 is a useful direct negative control because it
advertises per-stock grain, colour and tone fingerprints from as few as five
film scans, then exports Lightroom/Darktable presets.

U5.R2Y0 asks:

> Does the published fingerprint isolate a stock characteristic, or can scene
> colour, exposure distribution and ordinary scene texture alone determine
> its exported parameters?

This node audits the method. It does not evaluate a claimed stock on real
project pixels.

## Frozen source

- PyPI package: `negclone==0.2.0`;
- upload time: `2026-03-20T05:41:40.526593Z`;
- exact source distribution:
  `4672f11cf98b9d037debd0eab60a3ee9d278a73515b8184d7223b976d6a46387`;
- 29 safe relative archive members;
- MIT source license;
- five exact source-file hashes in the frozen config.

No network request or community image download is allowed during the audit.
The exact pinned analysis functions may run only on generated arrays.

## Source-level hypotheses

The audit verifies directly from the pinned source:

1. image selection uses `random.sample` without a seed argument;
2. grain patches use global `random.randint`;
3. colour regions are within-image luminance percentiles and their channel
   means;
4. tone is the scene luminance histogram/CDF followed by PCHIP;
5. grain is local standard deviation, autocorrelation and FFT over random
   patches, without a flat-patch or scene-texture separator;
6. multi-image aggregation is the median of these measurements;
7. scanner offsets are declared approximate;
8. exported XMP capacity is tone curve, three-way colour grading, shadow tint
   and grain controls, not a 3D LUT or nonlinear cross-channel stock operator.

Median aggregation can reduce outliers. It cannot identify or subtract a
confounder that is systematically tied to content, source or exposure.

## Frozen counterfactual probes

Every probe contains no simulated film response.

1. **Scene colour:** uniform red and blue arrays. If their midtone fingerprint
   biases differ by L2 at least `.5`, colour content alone drives the alleged
   stock colour.
2. **Exposure distribution:** the same neutral gradient over `[0,.5]` and
   `[.5,1]`. If exported PCHIP outputs differ by at least 100/255, scene
   exposure alone drives the tone curve.
3. **Scene texture:** flat grey versus a deterministic grey checkerboard. If
   the flat is at most `1e-12` grain while the checker is at least `.2`
   intensity and `.9` clumping, non-film texture is classified as grain.

The exact source functions run with seed `27070` around grain probes so formal
reports can repeat. This wrapper seed does not repair the package's public
reproducibility contract.

## Branches

- `content_histogram_texture_shortcut_close`: exact source and all three
  counterfactuals reproduce; retain NegClone only as a negative control.
- `source_or_probe_not_reproduced`: preserve the discrepancy and stop.
- `shortcut_not_demonstrated`: do not close without a new preregistered
  control.

No branch authorizes more community pixels, a neural rescue, a real stock
preset, current-pixel fitting, visual candidates or authenticity claims.
