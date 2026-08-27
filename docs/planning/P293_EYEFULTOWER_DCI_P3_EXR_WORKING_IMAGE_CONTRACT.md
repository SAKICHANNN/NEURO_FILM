# P293 EyefulTower DCI-P3 EXR to WorkingImage contract

## Question

Can the exact single EXR admitted by P292 be independently downloaded,
strictly identified as an uncompressed float32 linear DCI-P3 image, and
converted through an explicit colorimetric transform into an owned scene-linear
WorkingImage without clipping or silently assuming missing container metadata?

## Frozen source and staged-read boundary

- Parent evidence:
  `docs/evidence/P292_EYEFULTOWER_DCI_P3_EXR_SOURCE_READINESS_RESULT.json`.
- Official object:
  `EyefulTower/playroom_small/images-1k/40/40_DSC0001.exr`.
- Frozen byte length: `9,399,313`.
- Official MD5: `29c574af8fc3a543cd773eb812d56a49`.
- Multipart S3 ETag: `"a6f750cf77db90d9ca949523c1ed929b-2"`;
  it is transport identity only and is not substituted for the official MD5.

The object may be downloaded exactly once into the repo-relative P-backed
source root. Its size and official MD5 must pass before its local SHA-256 is
frozen in an additive source-lock commit. No OpenEXR header, channel or pixel
read is allowed before that lock.

After source lock, P293 may inspect the OpenEXR and proceed only if the file
itself contains enough metadata to identify the declared DCI-P3 chromaticities
and linear float32 encoding. The README statement alone is not accepted as a
silent substitute for absent or conflicting container color identity. The
same-source white-balanced/tone-mapped JPEG is forbidden as an independent
target or comparison.

## Frozen conformance gates

All gates must pass simultaneously:

1. parent evidence, source URL, size, official MD5 and locked local SHA-256 are
   exact;
2. the file contains exactly one non-deep scanline part with positive bounded
   dimensions and exactly RGB float32 channels;
3. compression is `NO_COMPRESSION`, data/display windows are exact and no
   unbound extra image channel is accepted;
4. container chromaticities identify DCI-P3 primaries and its declared white;
   absent, malformed, ambiguous or conflicting metadata rejects before pixel
   transformation;
5. decoded pixels are finite, preserve negative and greater-than-one values,
   and remain within the frozen absolute component ceiling;
6. an explicit float64 DCI-P3-to-XYZ transform plus explicit chromatic
   adaptation to D65 and XYZ-to-linear-Rec.2020 produces owned, writable,
   C-contiguous float32 pixels without clipping;
7. an independently configured color-science oracle agrees within the frozen
   float32 error ceilings on a fixed bounded probe and the real image;
8. source bytes and caller-owned arrays are unchanged; malformed, wrong-space,
   half-channel, extra-channel, non-finite and over-ceiling controls reject
   before output or destination publication;
9. two fresh forward/reverse reports are byte-identical, with zero JPEG reads,
   zero fitting/training/inference and no persistent decoded pixels or media.

## Stop rule and claim ceiling

Any failed gate yields `FAIL_CLOSED_EYEFULTOWER_DCI_P3_EXR_WORKING_IMAGE`.
Do not add chromaticities, infer a white point from the README, replace the
selected file, normalize/clamp pixels, relax thresholds, or use the JPEG
derivative as a target.

A pass proves only one private exact-file Windows/Python linear DCI-P3 EXR
ingress into the existing `linear_rec2020_d65` WorkingImage contract. It does
not establish EyefulTower bracket quality, arbitrary EXR/DCI-P3 support,
display rendering, HDR preference, package/schema/capability/product admission,
stock evidence or candidate-3 change.
