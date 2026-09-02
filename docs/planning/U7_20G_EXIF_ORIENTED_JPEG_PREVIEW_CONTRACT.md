# U7.20G EXIF-Oriented JPEG Preview Contract

## Problem

The public desktop preview path opts ordinary JPEG files into decoder-scaled
loading before float32 expansion.  It currently rejects every non-identity
EXIF Orientation even though the full-resolution WorkingImage path applies the
same orientation safely.  The direct-preview controller also derives preview
geometry from the stored JPEG axes, so a rotated portrait can be resized back
to landscape even when full decode is used.

This is a common-input compatibility defect, not a colour-operator change.

## Frozen scope

U7.20G may modify only oriented-dimension handling in
`src/inference/three_stock_preview.py` and the opt-in JPEG scaled decoder in
`src/preprocess/raster_decode.py`.  It may add its own config, audit, tests and
evidence.  It must not change product looks, strength, effects, recipes,
outputs, final full-resolution decode, catalog entries, or claims.

## Required behavior

1. EXIF Orientation values 2 through 8 are accepted by the opt-in scaled JPEG
   preview decoder.
2. Orientation is applied with axis flips/transposes only.  No additional
   resampling may be introduced by orientation handling.
3. Orientations 5 through 8 swap the source width and height before preview
   geometry is selected and before decoder-scale sufficiency is evaluated.
4. The scaled result must cover the requested *oriented* preview dimensions;
   undershoot and upsample requests still fail closed.
5. Missing/identity orientation retains the prior path.  Values outside 1--8
   fail closed.
6. The high-level three-look preview reports oriented source geometry and a
   portrait source remains portrait through its input-basis and look cards.
7. The full-resolution product renderer, output bytes, strict recipes and
   `film-inspired / Look Approximation` claim ceiling remain unchanged.

## Stop rule

Any geometry ambiguity, silent invalid-orientation fallback, full-render byte
drift, new interpolation in the orientation step, or regression in the
identity JPEG preview closes U7.20G without metadata stripping, forced crop,
rotation heuristics, format expansion or threshold rescue.

## Claim ceiling

A pass establishes only private compatibility of the current JPEG preview
path with EXIF Orientation 1--8.  It does not establish arbitrary JPEG/EXIF
support, camera colour accuracy, calibrated stock response, physical-film
reproduction, public release or cross-platform parity.
