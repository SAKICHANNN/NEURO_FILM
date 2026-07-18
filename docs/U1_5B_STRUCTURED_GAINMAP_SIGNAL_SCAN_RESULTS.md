# U1.5B structured gain-map signal scan results

Date: 2026-07-18

Decision: **pass — retain as a bounded fail-closed ingress guard**

## Implemented boundary

The existing U1.5A first/last-4-MiB marker sample remains intact. A new
format-aware guard additionally:

- traverses JPEG APP1, APP2 and COM segments from SOI to the first SOS;
- traverses PNG chunks through IEND while skipping IDAT by length;
- validates scanned PNG text CRCs and reads `tEXt`, `zTXt` and `iTXt`;
- bounds a retained metadata payload at 1 MiB and cumulative decoded text at
  4 MiB;
- bounds zlib output, requires a complete stream and rejects trailing data;
- reports only a deterministic marker name or an untrusted-structured-metadata
  signal, then rejects before pixel conversion.

No pixel payload is retained by this scan and no new dependency was added.

## Evidence

- valid JPEG APP1 and PNG post-IDAT `tEXt` witnesses place the official Adobe
  namespace more than 4 MiB from both file ends; both are detected and reject;
- equivalent large JPEG and PNG negative controls inspect and decode normally;
- recognized markers in compressed `zTXt` and compressed `iTXt` reject;
- a zlib text expansion beyond 1 MiB becomes
  `structured:png_metadata_untrusted` rather than allocating without a cap;
- the renderer rejects the middle-of-file JPEG witness and creates no output;
- all U1.5A appended-marker, AVIF, metadata, ICC and ordinary SDR regressions
  remain green.

Configuration SHA-256:
`a42d326c2957f8921a9a9ee7aeee2ee12d1891baf6930810ccb5c1155b14c908`.
Implementation commit: `0047d93b04caabe4413132637ee4b6a6750103a6`.

Verification:

- 48 focused ingress/renderer tests pass;
- 639 complete CPU tests pass;
- Python compile check passes;
- `git diff --check` passes.

## Primary-source interpretation

This closes a concrete sampling defect because Ultra HDR metadata can reside
in JPEG application metadata and PNG textual chunks may legally occur after
IDAT. It does not prove that all gain-map signalling is recognized.

- https://developer.android.com/media/platform/hdr-image-format
- https://www.itu.int/rec/T-REC-T.86/en
- https://www.w3.org/TR/png-3/

## Detection and claim ceiling

Traversal stops before JPEG entropy data and recognizes only the frozen U1.5A
marker vocabulary. Unknown/future binary-only signalling, complete MPF and ISO
21496-1 semantics remain unresolved. HEIF/AVIF remains rejected. The project
still has no HDR reconstruction, gain-map application, tone map or general
wide-gamut production renderer.

The valid claim is only: recognized HDR/gain-map signals in the covered JPEG
and PNG metadata structures fail closed without the former edge-sampling blind
spot.
