# U1.5A HDR/gain-map fail-closed ingress results

Date: 2026-07-17

Decision: **pass — unsupported dynamic-range inputs now fail closed**

## Implemented policy

The raster inspector records deterministic unsupported-dynamic-range signals.
The loader rejects those signals before pixel conversion, so a successfully
decoded base image can no longer silently masquerade as preservation of HDR,
wide-gamut or gain-map intent.

Rejected inputs are:

- decoded HEIF, HEIC and AVIF containers;
- metadata keys signalling HDR, gain map, CICP/NCLX or mastering information;
- JPEG/PNG payloads carrying recognized Adobe/Android `hdrgm` namespace or
  version markers, `GainMap` item semantics or Apple's auxiliary gain-map URI.

Payload detection reads at most the first and last 4 MiB and retains only the
matched marker name. ICC metadata alone remains legal.

## Evidence

- the local Pillow runtime confirms AVIF decode support; a valid SDR AVIF is
  inspected as AVIF but load is rejected as `container:AVIF`;
- official Adobe/Android `hdrgm` and Apple auxiliary marker fixtures reject;
- PNG gain-map metadata rejects;
- renderer subprocess rejection occurs before output creation;
- all existing JPEG/PNG/TIFF8, PNG/TIFF16, ICC and RAW tests remain green;
- 29 focused ingress/renderer tests pass;
- the complete CPU suite passes: 230 tests.

The policy follows the official Ultra HDR signal definition and Apple's
auxiliary gain-map model:

- https://developer.android.com/media/platform/hdr-image-format
- https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos

## Known detection ceiling

This is conservative prevention, not a complete ISO 21496-1 parser. Unknown or
future binary-only gain-map signalling may require more fixtures and a pinned
container parser. All HEIF/AVIF is therefore rejected even when it is genuinely
SDR. A later support leaf must validate reconstruction, colour metadata,
reference vectors and display mapping rather than merely delete this guard.

## Claim ceiling

The product honestly rejects currently unsupported HDR/gain-map/HEIF/AVIF
inputs. It does not support, preserve, tone-map or render those formats yet.
