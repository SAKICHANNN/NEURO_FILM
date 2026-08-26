# P251 strict ACES2065-1 OpenEXR WorkingImage ingress contract

## Purpose

P251 closes one concrete private RAW/HDR core gap: convert a strictly identified
ACES2065-1/AP0 OpenEXR interchange file into an owned ACEScg/AP1
`WorkingImage`. It is an opt-in loader and does not change generic raster/RAW
dispatch.

## Frozen input

- one existing local file;
- exactly one non-deep scanline OpenEXR part;
- exactly one packed `RGB` channel group with float32 samples;
- AP0 chromaticities and D60 adopted neutral matching the frozen float32 header
  values;
- integer `acesImageContainerFlag=1`;
- string `colorInteropID=lin_ap0_scene`;
- positive dimensions, each at most 16,384, and at most 100,000,000 pixels;
- every component finite with absolute value at most 65,504.

Multipart, deep, tiled, half/uint, separated/extra channels, missing or wrong
metadata, malformed files, empty shapes, oversized images and non-finite or
out-of-range components fail before a usable `WorkingImage` is returned.

## Frozen transform and output

The input AP0 values are multiplied in float64 by the fixed ACES2065-1/AP0 to
ACEScg/AP1 matrix, then cast once to owned C-contiguous float32. The result must
be finite and preserve negative and above-one samples. The returned object is:

- `working_space="acescg_ap1_d60"`;
- `transfer_state=source_transfer_state="scene_linear"`;
- `SourceProfile(kind="icc", description="ACES2065-1 AP0/D60 OpenEXR")`;
- no alpha, no orientation transform, 32-bit input;
- exact source path and strict-container metadata in `hdr_metadata`.

The source file is immutable. No clipping, tone mapping, gamut mapping,
display transform or output encoding is allowed.

## Dependency and execution boundary

Formal execution uses the exact already retained official OpenEXR 3.4.15
CPython 3.12 Windows wheel in an isolated temporary target. The project does
not add a default dependency. Absence or identity mismatch of this wheel fails
closed.

## Gates

1. Exact OpenEXR wheel identity and zero network.
2. Exact R1DT/P249 synthetic AP0 container and pixel identity.
3. Exact metadata/type/shape validation and source immutability.
4. AP1 result matches the original P249 ACEScg lattice within one float32
   roundtrip ULP envelope, with preserved negative/highlight samples.
5. Returned array is owned, writable and C-contiguous.
6. Wrong metadata, wrong storage, malformed input and boundedness controls all
   reject without a usable output.
7. Forward/reverse fresh-process scientific reports are byte-identical and
   owned temporary roots are empty after controller exit.

## Stop rule and claim ceiling

Any identity, metadata, storage, transform, ownership, boundedness, replay or
cleanup failure closes P251 without accepting approximate metadata, converting
half/uint samples, normalizing values, clipping, changing matrices, substituting
dependencies or routing generic EXR through this loader.

A pass proves only a private Windows/Python opt-in ACES2065-1-to-ACEScg
`WorkingImage` ingress on frozen synthetic/control files. It does not establish
arbitrary OpenEXR support, a default dependency/dispatch path, display or image
quality, SMPTE certification, package/schema/capability or product admission.
