# P90 DNG to ACES 2 HDR PQ PNG contract

## Question

Can the retained U1.4F generic scene-linear decode and exact official ACES 2
HDR Rec.2020-PQ view publish the same four rights-cleared DNG outputs through
the unchanged U1.4G RGB16 PNG rail, while the production loader and product
capability remain closed?

P90 is frozen before its first file publication. It does not alter the LibRaw
decode, ACES config/view, RGB values, range, quantizer or PNG metadata.

## Frozen pipeline

1. read the exact four U1.4F DNGs through `load_raw_working_image`;
2. require the existing `linear_srgb` / `scene_linear` WorkingImage boundary;
3. run pinned OpenColorIO 2.5.2 built-in ACES 2 target
   `Rec.2100-PQ - Display / ACES 2.0 - HDR 1000 nits (Rec.2020)` unchanged;
4. require finite encoded output inside `[0,1]` without clipping;
5. quantize full-range RGB16 as `floor(code*65535+0.5)`;
6. publish create-only through `StreamingRec2100PqPngWriter` with exact
   `cICP=09 10 00 01` and no invented `mDCV`, `cLLI` or ICC facts.

## Frozen gates

- exact U1.4F/U1.4G evidence, U1.4F config and ACES adapter hashes;
- all four source identities and decoded WorkingImage boundaries exact;
- unquantized HDR output hashes exactly equal the frozen U1.4F hashes;
- finite/in-unit outputs, half-up RGB16 quantization and strict sample readback;
- canonical and reversed row partition produce the same PNG and sample hashes;
- source DNGs remain unchanged; wrong target/range/shape/existing output fail
  before publication;
- the production loader continues to reject generated PQ PNGs;
- forward/reverse and two fresh-process normalized reports byte exact.

Any failure closes P90 without target, quantizer, profile, metadata, source,
decoder, threshold or row rescue.

## Claim ceiling

Private exact four-DNG Windows CPU mechanics from generic LibRaw scene-linear
decode through pinned official ACES 2 HDR 1000-nit Rec.2020-PQ view to the
existing U1.4G RGB16 PNG rail only. No calibrated camera IDT, photographic or
display quality, HDR10/mastering/content-light metadata, arbitrary DNG,
default renderer/loader, public package/schema/capability, film/stock, product
or delivery admission.
