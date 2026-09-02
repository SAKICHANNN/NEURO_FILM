# U1.5H strict SDR AVIF ingress contract

Date: 2026-09-02

Node: `ULT > U1 > U1.5 > U1.5H`

Status: frozen before implementation

## Product defect

The product currently rejects every AVIF before pixels. That preserves the
U1.5A HDR/gain-map boundary, but it also prevents ordinary, single-image SDR
AVIF photographs from entering the existing deterministic Look Approximation
workflow. Pillow can decode AVIF, but its decoded RGB mode does not prove the
source bit depth, NCLX identity, item graph, alpha, sequence, or gain-map
absence. Admission therefore requires a bounded ISO-BMFF inspection before
Pillow reads pixels.

## Frozen supported subset

An input is admitted only when all of the following are proven before pixel
decode:

1. AVIF still-image brand (`avif`) with no sequence or tone-map brand;
2. exactly one primary coded image item of type `av01`, with no grid, overlay,
   identity, tone-map, alternate, derived-image, or auxiliary relationship;
3. primary properties contain one 8-bit three-channel `pixi`, one `ispe`, one
   `av1C`, and one NCLX `colr` equal to primaries `1`, transfer `13`, matrix
   `6`, full range;
4. no ICC profile, alpha auxiliary item/property, HDR mastering property,
   gain-map/tone-map item, or unknown essential primary property;
5. Pillow independently reports `AVIF`, RGB, one frame, no alpha, 8-bit output,
   and dimensions matching the container inspection.

Optional Exif/XMP metadata items may be present only as non-image items with
`cdsc` references to the primary image. Metadata does not change the frozen
colour identity.

## Frozen gates

- the exact official libavif 8-bit SDR fixture
  `kodim03_yuv420_8bpc.avif` at commit
  `265f1524ef5fa9a736005a73a75430a1ac35cfe5` is accepted and decoded to a
  finite `linear_srgb` WorkingImage;
- a generated ordinary 8-bit RGB AVIF is accepted without an
  `assumed_srgb` warning because NCLX is explicit;
- AVIF with alpha, multiple frames, non-8-bit `pixi`, PQ/HLG or non-sRGB
  NCLX, ICC, derived-image references, unknown essential properties, or
  malformed/truncated boxes rejects before pixels;
- all six exact P278 gain-map/structural fixtures remain rejected before
  WorkingImage pixels, including the fixture that omits the `tmap` brand;
- HEIF/HEIC and the existing JPEG/PNG HDR/gain-map gates remain unchanged;
- the public `render_film.py --product-look ektar_100` path produces an image
  plus strict replayable recipe from the admitted official AVIF and replays
  byte-exactly;
- targeted and adjacent ingress/product regressions, Ruff, compile, evidence
  binding, and a final installed-runtime smoke test pass.

## Stop rule

If the exact official SDR fixture cannot pass without weakening any frozen
container, colour, item-graph, alpha, sequence, or bit-depth gate, close the
leaf without support. If any P278 gain-map/structural fixture is admitted,
close the leaf and retain U1.5A's all-AVIF rejection. Do not add a dependency,
infer missing colour identity, accept decoded-RGB mode as source proof, or tune
the subset after observing a failure.

## Claim ceiling

At most: strict single-image 8-bit sRGB-NCLX AVIF input compatibility for the
private deterministic film-inspired / Look Approximation product. This is not
arbitrary AVIF or HEIF support, HDR/gain-map reconstruction, metadata fidelity,
AVIF output, calibrated film-stock response, or physical-film reproduction.

