# U7.21A Strict SDR HEIC Product Ingress Contract

Status: prospectively frozen before product implementation and formal execution.

## Product defect

The product accepts a strict ordinary SDR AVIF subset, but an otherwise ordinary
single-image iPhone HEIC with an embedded Display P3 ICC profile is currently
reported as an unknown input because Pillow has no HEIF decoder. Users therefore
cannot render a common phone-photo container even when its colour identity and
SDR structure are explicit.

## Frozen source and dependency

- Source repository: `bigcat88/pillow_heif`, exact tag `v1.5.0`, commit
  `16c3dd8249f56fa07ab4f1350bd73e7a20b95bb1`.
- Positive fixture: `tests/images/heif_other/arrow.heic`, Git blob
  `0cf18cabbdacc70cf080f6b7528cd2543c2c9ebe`, 792,052 bytes, SHA-256
  `73f604d7353df4848505a1f45a5517f736857fe047372f95c2f3d671df92c22c`.
- Decoder dependency: decode-only `pi-heif==1.4.0`; the exact Windows CPython
  3.12 wheel is 2,407,596 bytes with SHA-256
  `81cb473227b2da35bfd7c69a12764f664d91e44b0ec2547cdfe5ca6b8b52cab9`.
  Its bundled notice identifies only LGPLv3 `libheif` and `libde265`. The
  encoder-bearing `pillow-heif` wheel is explicitly forbidden from the product
  requirements because it additionally bundles GPLv2 `x265` and is unnecessary
  for input decoding.
- All retained media live under the repository-relative P-backed
  `data/external/u7_21a_strict_sdr_heic_v1/` namespace.

The pre-contract investigation decoded the positive fixture once in an isolated
P-backed scratch environment to establish feasibility and the independent
oracle below. This is product defect repair, not a sealed scientific cohort.

## Frozen admissible subset

A HEIC/HEIF input is admitted only when metadata parsing before sample access
proves all of the following:

1. still-image `image/heic`, not a sequence brand;
2. exactly one top-level image and primary index zero;
3. primary image is exactly 8-bit `RGB`, with no alpha;
4. no auxiliary image, depth image, gain map, mastering-display, content-light
   or ambient-viewing metadata;
5. colour identity is either a non-empty parseable embedded ICC profile or
   exact sRGB NCLX (`1/13/6/full`);
6. decoded dimensions equal the metadata dimensions and libheif's already
   applied container transform is not applied a second time.

The implementation must use a narrow private adapter rather than globally
registering a Pillow HEIF plugin. Native Pillow AVIF behavior and every existing
HDR/gain-map rejection remain unchanged.

## Frozen success gates

- exact source, dependency, wheel-license and source-immutability identities;
- positive inspection reports 3024x4032, one frame, 8-bit RGB and embedded ICC;
- positive decode yields owned finite float32 `(4032, 3024, 3)` linear-sRGB;
- independent decoded sRGB8 SHA-256 is
  `cdb91bfabe81c965abd49b4aa21fb157dea79579af2d12b7fb845cf1907ed624`;
- independent linear-float32 SHA-256 is
  `068b9e4ed3b091be6ceefe3d75a814c5acaa3791f407bc705b6a201bcf52440a`;
- 10-bit, alpha, missing-profile, burst, auxiliary/depth/gain-map, corrupted and
  truncated fixtures all reject before pixel materialization;
- one public Ektar 100 Look Approximation render writes PNG plus strict recipe,
  and recipe replay is byte-exact;
- a bounded desktop three-look preview succeeds without changing its
  `film-inspired / Look Approximation` claim;
- the canonical pre-existing Ektar `.65` raster output remains byte-exact at
  `fc51547d1e00a0a4a36dce96847afe09d27b3b531b65172d3f32fa2a46d2087d`;
- targeted HEIC, AVIF/HDR, raster-ingress, CLI/recipe and desktop-preview
  regressions pass; Ruff, format, compile, JSON and diff checks pass;
- a new absent versioned P-backed private runtime installs from the exact
  product requirements, validates both native launchers and records its
  dependency inventory. Existing runtimes remain immutable.

## Stop rule and claim ceiling

If the exact positive fixture, decoder identity, pre-pixel negative controls,
ICC conversion, public render/replay, preview, runtime install or existing
AVIF/HDR boundaries fail, U7.21A closes without changing fixture, dependency,
profile policy, bit depth, auxiliary policy, colour identity, renderer, look,
strength, gates or output format.

Pass claims only strict single-image 8-bit colour-identified SDR HEIC input
compatibility for the private deterministic `film-inspired / Look
Approximation` product. It does not claim arbitrary HEIF/HEIC, sequence, alpha,
HDR/gain-map, camera colour accuracy, calibrated stock response, physical-film
reproduction, public release, standalone packaging or cross-platform parity.
