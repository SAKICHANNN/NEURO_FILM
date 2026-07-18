# U1.5B structured gain-map signal scan contract

Date: 2026-07-18

Node: `ULT > U1 > U1.5 > U1.5B`

Status: frozen before implementation

## Parent evidence and defect

U1.5A correctly rejects HEIF/AVIF and recognized HDR/gain-map signals, but its
JPEG/PNG payload check samples only the first and last 4 MiB. A recognized
signal stored in a legal metadata segment/chunk in the unsampled middle can
therefore be missed and the SDR base can still be decoded.

Primary sources establish the container structure behind this defect:

- Android Ultra HDR associates a gain-map image with a primary JPEG through
  XMP/GContainer or MPF; a legacy reader can display only the SDR primary:
  https://developer.android.com/media/platform/hdr-image-format
- ITU-T T.86 defines JPEG APPn application metadata segments on top of T.81:
  https://www.itu.int/rec/T-REC-T.86/en
- PNG Third Edition defines a signature followed by length-delimited chunks
  and permits textual chunks after IDAT:
  https://www.w3.org/TR/png-3/

## Frozen implementation

Preserve the U1.5A marker vocabulary and fail-closed policy. Add bounded,
format-aware discovery as follows:

1. JPEG: walk length-bearing marker segments from SOI through the first SOS;
   inspect APP1, APP2 and COM payloads for the existing recognized markers.
2. PNG: walk the complete length-delimited chunk stream through IEND without
   reading pixel IDAT payloads; inspect `tEXt`, `zTXt` and `iTXt` metadata,
   including safely bounded decompression for compressed text.
3. Keep the existing first/last bounded byte scan as an independent fallback
   for appended or nonstandard recognized marker placement.
4. Return only deterministic marker names. Do not retain metadata payloads.

Malformed structured metadata, unsafe compressed metadata, or traversal that
exceeds the frozen metadata budget must fail closed before pixel conversion.
Ordinary image payloads are skipped by length and do not count against the
metadata budget.

## Frozen limits

- maximum bytes retained from any one metadata payload: 1 MiB;
- maximum cumulative decoded textual metadata: 4 MiB;
- JPEG segment payloads remain bounded by the two-byte T.81 segment length;
- PNG chunk lengths above the PNG 31-bit limit are invalid;
- compressed PNG text is decoded with a hard output cap and must reach a clean
  zlib end-of-stream inside that cap.

## Frozen gates

- a valid JPEG with an official marker more than 4 MiB from both file ends is
  detected in APP metadata and rejected before pixel conversion;
- a valid PNG with an official marker more than 4 MiB from both file ends is
  detected in a textual chunk after IDAT and rejected;
- compressed `zTXt` and compressed `iTXt` recognized markers are detected;
- oversized or over-expanding textual metadata fails closed without unbounded
  allocation;
- ordinary large JPEG/PNG fixtures still inspect and decode;
- U1.5A AVIF, appended-marker, metadata and SDR regressions remain green;
- renderer rejection creates no output;
- focused tests, complete CPU suite, compile check and diff check pass.

## Boundaries and claim ceiling

This is structured recognition of the already frozen signal vocabulary. It is
not a complete JPEG/PNG validator, MPF decoder, ISO 21496-1 parser, gain-map
reconstruction path, HDR tone map, HEIF/AVIF implementation or format-support
claim. Unknown binary-only and future signalling may remain unidentified.

## Branches

- any U1.5A regression or false rejection of the frozen ordinary fixtures:
  reject U1.5B and retain U1.5A;
- bounded structured detection passes: retain the parser as a fail-closed
  ingress guard and record its detection ceiling;
- successful detection does not open HDR decoding, renderer integration or a
  claim that dynamic range was preserved.
