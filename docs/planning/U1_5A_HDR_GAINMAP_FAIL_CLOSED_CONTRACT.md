# U1.5A HDR/gain-map fail-closed ingress contract

Date: 2026-07-17

Node: `ULT > U1 > U1.5 > U1.5A`

Status: frozen before implementation

## Defect

The current inspector emits `limited_heif_hdr` for HEIF/AVIF but the loader
continues through Pillow's decoded base image. In the checked environment,
Pillow reports AVIF decode support. An AVIF or gain-map container can therefore
be silently reduced to an SDR/base rendition while auxiliary HDR information,
transfer metadata or wide-gamut intent is discarded.

This violates the unknown-colour-state fail-closed rule. The project has no
validated HDR/gain-map reconstruction or tone-map implementation yet.

## Primary-source basis

- Android's official Ultra HDR v1.1 specification says a conforming JPEG can
  be signalled by `hdrgm:Version="1.0"` and the Adobe HDR gain-map namespace;
  it uses XMP/GContainer or MPF to associate a `GainMap` item with the base
  rendition: https://developer.android.com/media/platform/hdr-image-format
- Apple documents HDR gain maps as auxiliary data for JPEG/HEIF and notes that
  an HDR result requires applying that map to the SDR base; a legacy/default
  decode may expose only the SDR appearance:
  https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos

Therefore successful base-image decode is not evidence that dynamic-range and
colour intent were preserved.

## Frozen policy

Until a separately validated path exists, `load_working_image` must reject:

1. all decoded HEIF, HEIC and AVIF containers;
2. raster metadata keys indicating HDR, gain map, CICP/NCLX or mastering data;
3. JPEG/PNG payloads containing recognized Ultra HDR/Adobe/Apple gain-map
   namespace or semantic markers.

Inspection remains read-only and returns dimensions/format plus a structured
warning where the backend can inspect the file. Rejection happens before pixel
conversion and states that HDR/gain-map reconstruction is not implemented.

Ordinary SDR JPEG/PNG/TIFF, including supported sRGB ICC, must continue to
load. ICC alone is not an HDR signal.

## Frozen gates

- a valid SDR AVIF fixture is inspected but rejected at load rather than
  silently treated as sRGB8;
- valid JPEG fixtures carrying official `hdrgm` namespace/version and Apple
  auxiliary gain-map markers are rejected;
- a PNG metadata gain-map fixture is rejected;
- ordinary profiled/unprofiled SDR fixtures remain unchanged;
- inspection records deterministic signal names without retaining payload;
- renderer subprocess fails before creating output for rejected inputs;
- targeted tests, complete CPU suite, diff check and propagation pass.

## Boundaries

No HEIF/AVIF/HDR decoder, tone map, gain-map application, OCIO/ACES transform,
new dependency or output format is added. This leaf makes unsupported behavior
honest; it does not claim format support.

## Reopen condition

Support requires owned/redistributable fixtures for Apple auxiliary maps,
Ultra HDR/ISO 21496-1 and AVIF/HEIF transfer/profile variants; pinned decoder
semantics; reconstruction/reference vectors; display mapping; metadata
round-trip; severe-artifact/full-resolution tests; and explicit product scope.
