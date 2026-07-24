# U5.R2H0C1 CAVE conditional-variability contract

Date frozen: 2026-07-24
Parent: `U5.R2H0C`
Status: **preregistered before witness-output evaluation**

### Pre-result source-format erratum

The official-CRC audit, before any witness output was computed, found that all
31 `watercolors_ms` bands are 8-bit RGBA with opaque alpha, while the other
961 spectral PNGs are 16-bit single-channel images. The source page describes
the database bands as 16-bit grayscale, so `watercolors_ms` is excluded rather
than converted or reinterpreted. The remaining 31 scenes and every scientific
gate below are unchanged. This is a source-integrity exclusion, not a result-
conditioned population change.

## Hypothesis and non-hypothesis

Question: among approximate measured real-material reflectances, do
cross-scene samples that are close in D65 CIE Lab remain much closer after the
frozen Velvia datasheet witness than H0A's adversarial feasible metamers?

This does not hypothesize that display RGB identifies the surface spectrum.
H0A's theoretical non-identifiability remains true under every H0C1 outcome.
The strongest possible result is only that one measured-material population
supports a useful empirical prior for a later explicit Look Approximation.

## Immutable lineage and transfer verification

The source of truth is the Columbia official ZIP at
`https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip`:

- expected bytes: `405988465`;
- HTTP ETag: `"1832e471-4630073df3380"`;
- official ZIP entries: `1120`;
- central-directory offset/bytes: `405876217` / `112226`;
- official central-directory SHA-256:
  `cad6a8d0c5b8f83c3d3695200ebc03784bb275e9d4eba093fc232018625fe7a6`.

Because the official host is currently throttled, a Hugging Face copy may be
used only as a transfer channel. Every retained spectral PNG must match the
official central directory by normalized path, uncompressed size and CRC32.
An absent, additional, mismatched or non-decodable required member fails the
data gate. Mirror metadata and its unknown licence do not replace the official
research-only boundary.

## Population and grouping

- wavelengths: exactly `400..700 nm` in `10 nm` steps;
- reflectance decoding: unsigned 16-bit PNG divided by `65535`;
- `watercolors_ms` is excluded by the pre-result format audit above;
- scene is the primary group; spatial cell is the sampling unit;
- each 512x512 scene is partitioned into a fixed 16x16 grid of 32x32 cells;
- each cell spectrum is the per-band median of its pixels;
- retain cells with finite bounded values, D65 relative `Y` in `[0.01, 0.95]`,
  and at least 90% structurally valid pixels;
- do not read supplied RGB previews for selection, fitting or gating;
- do not infer material labels from scene names.

Neighbouring pixels are never treated as independent observations. The scene
group must be retained in every manifest, bootstrap and pair-selection step.

## Overlap-only colour and witness math

H0C1 uses official CIE 1931 2-degree CMFs and D65 restricted to 400--700 nm.
The D65 colour matrix is re-normalized on this exact overlap. H0A's digitized
sensitivity, characteristic and dye-density curves are immutable and sampled
on the same 31 wavelengths. Film-layer neutral gain and output tristimulus
normalization are recomputed on the overlap; no 380--390 or 710--720 nm value
is fabricated. This is an `overlap-only witness`, not the full H0A witness.

## Conditional-pair construction

Pair construction is independent of witness output:

1. compute D65 Lab for every retained cell representative;
2. enumerate cross-scene pairs within input Delta E76 `<=1.0` using a fixed
   Euclidean Lab neighbour search;
3. require spectral RMS difference `>=0.01` so duplicates are not evidence;
4. sort by `(input_delta_e, scene_a, cell_a, scene_b, cell_b)`;
5. greedily accept each representative at most once, at most 16 pairs per
   unordered scene pair, and at most 64 pair incidences per scene.

Binding support requires at least 256 pairs, 16 represented scenes, 32 scene
pairs, maximum scene incidence share `<=0.15`, and maximum scene-pair share
`<=0.0625`. Radii 2 and 3 may be reported only as non-binding support
diagnostics if the radius-1 population is insufficient.

## Metrics, controls and uncertainty

For every frozen pair report input Delta E76, spectral RMS and output Delta
E76. Primary summaries are median and p95 output Delta E76.

Fixed H0A comparators are median `82.69825514272178` and p95
`144.34857413603848`. Report empirical/comparator ratios. Also report:

- exact same-spectrum replay (must be numerical zero);
- deterministic repeat hashes (two runs);
- unconditioned cross-scene pairs selected with seed `20260724` as a
  non-binding conditioning control;
- scene-bootstrap 95% intervals from 1,000 deterministic resamples, sampling
  scenes first and retaining induced pairs;
- per-scene-pair support and tails; no pixel-level confidence interval.

## Frozen decisions

After data/integrity/support gates:

- `empirical_ambiguity_broad`: median exceeds `20.674563785680445` or p95
  exceeds `36.08714353400962` (25% of H0A adversarial spread);
- `materially_narrower_but_not_bounded`: the 25% gates pass, but median exceeds
  `5.0`, p95 exceeds `12.0`, median bootstrap UCB exceeds `7.5`, or p95
  bootstrap UCB exceeds `18.0`;
- `bounded_empirical_prior_candidate`: all preceding absolute and bootstrap
  gates pass.

Data invalidity or insufficient radius-1 support is reported separately and
cannot be converted into a scientific pass by widening the radius after
results are observed.

Only `bounded_empirical_prior_candidate` may open H0C2: a separately frozen,
simple deterministic spectrum-prior/canonicalizer comparison. It still may
not open film-pixel fitting, neural training, product integration, authenticity
claims or visual candidate selection.

## Claim ceiling

Source-limited empirical conditional variability under one overlap-only,
datasheet-prior Look Approximation. No identified digital-to-film operator,
stock response, calibration, preference or product value.
