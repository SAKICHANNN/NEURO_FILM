# P269 HDR+ One-Burst Acquisition Contract

## Scope and parent

P269 is a DRPT L2 bounded source-integrity leaf opened only by P268. It acquires
the exact hash-selected official Google HDR+ burst
`0382_20150912_131056_666` and its two matching result inventories into the
repo-relative P-backed source root. It verifies immutable object identities and
basic file/metadata structure. It does not fit, train, infer, score HDR quality,
read a fresh cohort, consume candidate 3 or create a product interface.

## Frozen source and objects

- Parent evidence: `P268_HDRPLUS_BURST_SOURCE_FEASIBILITY_RESULT.json`, tracked
  SHA-256 `2f9fc7a4875031b16d0488bcde7fe4e107bed03f80ba0147870e4b16e5da5505`.
- Exact 24-object manifest: eight burst DNGs, eight lens-shading TIFFs,
  `rgb2rgb.txt`, `timing.txt`, plus `merged.dng`, `final.jpg` and
  `reference_frame.txt` from each of the two official result roots.
- Total expected bytes: `256482210`, below the frozen 1 GiB ceiling.
- Each object is bound by path, size, GCS generation and base64 MD5 from P268.
- Destination: `data/external/p269_hdrplus_one_burst_v1/`; all access remains
  repository-relative. No mirror, replacement burst or bulk subset is allowed.

## Acquisition and validation

1. Re-enumerate metadata for exactly the 24 frozen object URLs and require
   unchanged size/generation/MD5 before any body read.
2. Download each missing object to an owned staging file, verify byte length and
   MD5, then publish create-only. Existing correct files are reused; existing
   mismatches fail closed and are never overwritten.
3. Validate every DNG/TIFF/JPEG/text signature. Record DNG dimensions, CFA/RAW
   structure and capture tags through metadata-only readers where available.
4. Decode only a bounded centre probe from each input DNG and full tiny
   sidecars. Do not render, merge, compare against result pixels, fit or score.
5. Run forward/reverse audits and require a canonical scientific identity. Raw
   timing/download counters may differ and are excluded from that identity.

Any source drift, checksum mismatch, missing member, unsafe path, unsupported
decode structure, over-budget total, publish collision or non-finite bounded
probe closes this exact leaf without burst replacement or parser rescue after
formal lock.

## Claim ceiling

A PASS establishes only a private exact local source lock and one-burst RAW /
capture-sidecar decode feasibility for a later separately preregistered D0.
CC-BY-SA product/share-alike obligations remain unresolved. No HDR truth,
quality, general burst support, package, schema, capability, product mapping or
candidate-3 admission opens.
