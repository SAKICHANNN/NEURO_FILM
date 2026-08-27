# P270 HDR+ Split-Capture RAW Mechanism D0 Contract

## Scope

P270 is a one-scene, source-only mechanism D0 under the Goal's genuinely new
capture-time observation lane. It tests whether four aligned same-exposure RAW
captures predict four independently held-out captures better than the strongest
frozen single-frame smoothing control and an unaligned temporal mean. It uses
only the exact P269 burst. Official merged DNG/JPG pixels remain unread.

This is not HDR truth, natural-image confirmation, a learned model, candidate-3
admission or product evidence. A PASS can open only a fresh-group prospective
program with new bursts and an unchanged mechanism.

## Frozen roles and read order

- Anchor/development frames: `N000`, `N002`, `N004`, `N006`.
- Held-out frames: `N001`, `N003`, `N005`, `N007`.
- The implementation must fully build and hash the aligned-even candidate,
  unaligned-even temporal control, single-frame control family and exactly 512
  source-only reliable blocks before the first held-out DNG is opened.
- Held-out frames are then aligned by the same frozen phase-correlation/sign
  rule and averaged only as an evaluation proxy.
- Lens-shading maps, official merged DNGs, final JPGs and result pixels are not
  consumed in this D0. Timing/RGB sidecars are provenance only.

## Frozen representation, alignment and controls

1. Read each uncompressed uint16 Bayer plane through exact TIFF strip geometry;
   normalize by the frozen white level and split into four half-resolution CFA
   planes without demosaic or tone mapping.
2. Estimate translation from the mean of the two green planes at 1/4 scale via
   OpenCV phase correlation. Evaluate both shift signs using source-anchor
   central RMSE; choose the strictly lower one, with positive sign on exact tie.
3. Warp each CFA plane to `N000` with float32 linear interpolation and NaN
   borders. Require bounded shift and overlap.
4. Candidate: equal mean of the four aligned even frames.
5. Temporal control: equal mean of the same four even frames without alignment.
6. Single-frame controls use only `N000`: identity and Gaussian blur with fixed
   sigma `0.5`, `1.0`, `1.5`; target-exposed scoring may select the strongest
   control per block, but candidate generation cannot.
7. Build 32x32 blocks over all four CFA planes. Rank source-only by even-frame
   aligned temporal standard deviation plus `0.25` times candidate gradient;
   ties resolve by `(plane,y,x)`. Freeze exactly the lowest-risk 512 valid
   blocks before held-out access.

## Frozen gates

- candidate/control/block hashes freeze before held-out reads;
- maximum absolute half-resolution shift <= 64 pixels;
- aligned-even overlap >= 0.80 and 512 source-only blocks freeze;
- at least 480 frozen blocks remain valid under held-out alignment;
- candidate beats the per-block strongest single-frame control on >= 75% of
  valid blocks, median RMSE reduction >= 10%, worst reduction >= -25%;
- candidate beats unaligned temporal mean on >= 75% of valid blocks with median
  RMSE reduction >= 5%;
- forward/reverse runs reproduce the complete scientific identity;
- result-pixel, fit, train and inference reads remain zero.

Any failure closes this exact one-burst split-capture family without changing
roles, block count, sigmas, alignment, thresholds or cohort. Do not use the
held-out frames to tune masks, shifts, normalization or controls.

## Claim ceiling

Private one-scene split-capture RAW mechanism evidence only. No HDR truth or
quality, natural-scene generalization, fresh-group result, package, schema,
capability, product mapping or candidate-3 admission.
