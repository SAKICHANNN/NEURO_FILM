# P283 libavif gain-map to P87 bridge contract

## Node

`ULT > U1 > U1.5 > P283`

## Question

Can the exact higher-headroom rendition selected by the official P282 libavif
runtime be converted from its explicit BT.709-primary/PQ RGB16 representation
into an owned, immutable P87 absolute-linear Rec.2020 MatchView without hidden
color-state assumptions or more than two RGB16 codes of roundtrip error?

This is the only bridge opened by P282. It is a private explicit color-state
adapter, not a production-loader change or an HDR-quality experiment.

## Prospective source and role freeze

Reuse the exact P278 fixtures, P282 official libavif v1.4.2 runtime and P282
evidence. P282's already-consumed `avifdec --info` output fixes all three
higher-headroom renditions as BT.709 primaries (`1`), PQ transfer (`16`),
BT.601 matrix (`6`) at the container/YUV boundary and full range. The official
runtime performs YUV-to-RGB before P283.

- `seine_hdr_gainmap_small_srgb.avif`: base headroom 1.3, alternate 0.0;
  choose the HDR base decoded by `avifdec`.
- `seine_sdr_gainmap_big_srgb.avif`: base headroom 0.0, alternate 1.3;
  choose the official alternate-headroom `avifgainmaputil tonemap` output.
- `seine_sdr_gainmap_srgb.avif`: base headroom 0.0, alternate 1.3;
  choose the official alternate-headroom `avifgainmaputil tonemap` output.

The exact higher-rendition PNG sample hashes already frozen by P282 are parent
bindings. No lower-headroom output, invalid fixture, new source or different
headroom may enter P283.

## Frozen adapter

The private adapter accepts:

- exact non-empty source bytes and expected SHA-256;
- an exact HxWx3 `uint16` RGB array decoded from the selected official PNG;
- exact decoder version `1.4.2`;
- source primaries `1`, transfer `16`, full-range flag, and the frozen
  higher-rendition role.

It performs, in order:

1. normalize RGB16 by 65535 in float64;
2. apply the existing exact BT.2100 PQ EOTF component-wise to absolute linear
   BT.709/sRGB-primary cd/m2;
3. apply the existing dependency-free D65 linear-sRGB-to-linear-Rec.2020
   matrix without clipping;
4. cast once to owned contiguous float32 absolute Rec.2020 cd/m2;
5. publish an immutable P87-compatible MatchView with 203-nit reference white,
   exact source/sample/provenance bindings and no alpha.

The adapter performs no media I/O, network access, tone mapping, gamut mapping,
clipping, fitting or filesystem publication. Caller arrays remain unchanged;
all validation fails before an output object is returned.

## Formal execution

1. Verify exact P282/P87 evidence, runtime/source manifests, executables and
   fixtures.
2. Run two fresh processes in forward/reverse fixture order.
3. Materialize each frozen higher rendition with only the official runtime and
   require its P282 parent sample identity before adapter invocation.
4. Apply the private adapter and validate the complete P87 descriptor, owned
   read-only float32 buffer, finite/nonnegative range and source immutability.
5. Independently convert the absolute Rec.2020 output back to absolute linear
   sRGB, encode with the existing BT.2100 PQ inverse EOTF, quantize RGB16
   half-up, and compare with the frozen input samples.
6. Exercise wrong source hash, wrong decoder, wrong primaries, wrong transfer,
   wrong role, non-uint16, nonfinite/impossible shape and caller-ownership
   controls.
7. Remove all runtime PNGs and formal scratch; formal network reads are zero.

## Frozen gates

All gates are conjunctive:

1. all source/runtime/parent evidence and higher-rendition sample identities
   are exact;
2. all three adapters return P87 absolute Rec.2020 MatchViews at 400x300;
3. all output components are finite, nonnegative and at most 10,000 cd/m2;
4. each output has maximum luminance above the 203-nit reference white;
5. independent RGB16 roundtrip maximum error is at most two codes, median error
   is at most one code, and no component changes by more than the frozen bound;
6. inputs are unchanged and outputs are owned, contiguous and read-only;
7. all invalid controls reject without output;
8. forward/reverse reports and stable scientific identities are byte-exact;
9. network, fitting, hidden targets, retained PNG/pixel outputs and formal
   residue are zero.

Any failure closes P283 without changing the runtime, headroom, CICP, transfer,
matrix, quantizer, error gates or fixture set.

## Claim ceiling

At most `PASS_PRIVATE_LIBAVIF_GAINMAP_P87_BRIDGE`: private exact conversion of
three frozen official libavif higher-headroom RGB16 renditions into the existing
P87 absolute Rec.2020 MatchView boundary. No complete ISO 21496-1, arbitrary
AVIF/HEIF, HDR reconstruction quality, metadata preservation, display
validation, public API/dependency/package/schema/capability/product admission,
film-stock evidence or candidate-3 change is allowed.
