# P99 — opt-in DNG baseline-exposure contract

Status: frozen after metadata-only tag inventory and before implementation or
P98 pixel execution.

## Question

Can the exact P98 camera-linear `ForwardMatrix` raster be followed by the DNG
1.7.1.0 default-exposure adjustment, using only source-bound DNG metadata and
an explicit scene-linear multiplier, without clipping or changing the generic
RAW loader?

This is a narrow RAW/DNG explicit-operator leaf. It is not a renderer-quality
test, a three-stock proxy rescue, or an after-only reference-matching
candidate.

## Authority and frozen observations

- Adobe DNG Specification 1.7.1.0 PDF SHA-256
  `abdecfd8e104e8b86cc054d3a40b677bb2ebe87131bfde080a1df3ee16eb2f8d`.
- P98 evidence SHA-256
  `e73c99fe61781988f61a56fec46278d715431c04ec432173639deb16ba7cfac5`.
- P98 implementation SHA-256
  `eb50a4b0957b88a40b2da6da929a90cc5d9e390d95589914b41e0ff0711091e9`.
- P98 config SHA-256
  `5fb17f82cfecdae40a4c812f2ea9c26641e91bcab43e17c8a6a4bc94b2f03c35`.

The specification defines `BaselineExposure` (tag 50730) as an EV shift of
the default exposure zero point, with default `0.0`; positive values brighten.
It defines `BaselineExposureOffset` (tag 51109) as an additional profile EV
offset, also default `0.0`.

The metadata-only inventory is frozen as follows:

- Motorola: explicit `0/100`, offset absent;
- LG: exposure absent, offset absent;
- Xiaomi: explicit `0/100`, offset absent;
- Blackmagic: explicit `265412/100000`, offset absent;
- Huawei: explicit `0/100`, offset absent.

No tag value, row, default, or source may be changed after pixel execution.

## Frozen implementation

Add a private opt-in primitive that:

1. verifies the source byte count and SHA-256 before P98 execution;
2. reads IFD-0 tags 50730 and 51109 directly;
3. uses the exact standard default `0.0` only when the corresponding tag is
   absent, and records explicit/default provenance for both values;
4. rejects wrong TIFF types/counts, zero rational denominators, non-finite
   values, or total exposure outside `[-8, +8]` EV;
5. calls the unchanged P98 opt-in loader and multiplies its scene-linear
   float32 Rec.2020 pixels by `2 ** (BaselineExposure +
   BaselineExposureOffset)` in float64 row blocks, quantizing once to float32;
6. returns a new `WorkingImage` with no clipping, gamut map, tone curve, output
   encoding, or mutation of the P98 image or source file.

The primitive remains private and is not exported from `src.preprocess` or
wired into the generic loader.

## Gates

- all five exact P98 rows and five makes are mandatory;
- metadata values and explicit/default provenance exactly match the frozen
  inventory;
- total EV equals `0.0` on four rows and `2.65412` on Blackmagic;
- scale equals an independent float64 `exp2` oracle and remains in `[2^-8,2^8]`;
- output shape/state/profile lineage is preserved and all values are finite;
- output agrees byte-for-byte with an independently staged float32 oracle;
- zero-EV rows equal P98 pixels byte-for-byte, while Blackmagic changes;
- P98 pixels and source bytes remain unchanged;
- maximum absolute output is at most `64` and no clipping is introduced;
- invalid metadata and source drift fail before a returned output;
- forward/reverse enumeration and two fresh-process scientific reports are
  byte exact.

## Stop rule and claim ceiling

Any binding, metadata, range, mutation, oracle, finite, or replay failure
closes this exact leaf. Do not add a tone curve, clamp, exposure rescue, row
replacement, or RF3.D0R rerun.

PASS means only that the exact five-DNG private P98 mechanism can honor the
standard DNG baseline-exposure metadata as an explicit scene-linear operator.
It does not establish arbitrary DNG support, vendor/Adobe rendering parity,
sensor calibration, photographic quality, a scene-to-display tone map,
default-loader integration, output files, public package/schema/capability,
product, film, stock, or preference admission.
