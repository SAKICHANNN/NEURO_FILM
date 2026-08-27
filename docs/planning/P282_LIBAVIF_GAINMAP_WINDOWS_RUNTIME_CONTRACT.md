# P282 libavif gain-map Windows runtime contract

## Node

`ULT > U1 > U1.5 > P282`

## Question

Can the exact official libavif Windows release independently recognize, decode
and apply the gain maps in the three official-valid P278 AVIF fixtures, while
failing closed on the three frozen invalid/ignored-gain-map controls?

P282 follows the P278 container gap. It does not modify libultrahdr, P87, the
production loader or any default. It is a private runtime/mechanics test, not
an HDR-quality or display-validation test.

## Prospective runtime freeze

- Official repository: `https://github.com/AOMediaCodec/libavif`.
- Release: `v1.4.2`, commit
  `c5240fc79fe5c2407e10afd35f5505ef6333ea49`, annotated tag object
  `bd32fb40a3b2028eb555b67f769c57a32ed8a1b4`.
- Release asset: `windows-artifacts.zip`, 15,810,507 bytes, publisher digest
  `cb2d9fea43dcbab1d0707e3b37eb7b08070ad2fb60a2c188c39ec12382c0484a`.
- The official release is the only permitted runtime source. No local rebuild,
  package-manager substitution, alternate codec or modified binary is allowed.
- After acquisition, bind the exact retained runtime member set, sizes and
  SHA-256 values before any fixture execution. Retain only the executables,
  runtime libraries and notices required by `avifdec` and
  `avifgainmaputil` under the repository-relative P-backed P282 data root.

The libavif v1.4.2 source API and applications expose gain-map metadata through
`avifImage::gainMap`, enable gain-map parsing in `avifdec --info`, and provide
`avifgainmaputil printmetadata` plus `tonemap`. P282 calls only those official
tools; it does not recreate the ISO 21496-1 equations.

## Frozen fixtures and roles

Reuse the exact P278 source manifest and bytes without refetching or replacing
any fixture.

Valid:

- `seine_hdr_gainmap_small_srgb.avif`;
- `seine_sdr_gainmap_big_srgb.avif`;
- `seine_sdr_gainmap_srgb.avif`.

Invalid or required-to-ignore:

- `seine_hdr_gainmap_wrongaltr.avif`;
- `seine_sdr_gainmap_gammazero.avif`;
- `seine_sdr_gainmap_notmapbrand.avif`.

The exact P278 roles, source hashes and retained-file set are immutable. No
new fixture, repaired file or excluded file may enter P282.

## Frozen execution

1. Verify the release archive identity, retained runtime manifest and every
   runtime file before execution.
2. Run two fresh Python processes in forward and reverse fixture order.
3. For every valid fixture:
   - require `avifdec --info` and `avifgainmaputil printmetadata` to succeed;
   - require the information output to expose gain-map metadata;
   - decode the base image to deterministic 16-bit PNG;
   - tone-map once at the exact alternate headroom parsed from the official
     metadata, using the official tool and deterministic PNG settings;
   - require both outputs to decode at the exact 400x300 dimensions and differ
     materially in RGB samples.
4. For each invalid fixture, require `printmetadata` or `tonemap` to fail
   without a usable tone-mapped output. The ordinary base image may remain
   decodable when the official format requires ignoring an invalid association.
5. Run a 64-byte truncation control and a missing-runtime-member control; both
   must fail before usable output publication.
6. Canonicalize report records by fixture name, prove source and runtime
   immutability, then remove every decoded/tone-mapped output and owned scratch.
7. Formal execution has zero network, encoder, source replacement, target,
   training, fitting or quality-metric reads.

## Frozen gates

All gates are conjunctive:

1. release tag/commit/asset and minimal runtime member identities are exact;
2. all three valid fixtures expose gain-map metadata;
3. all three valid fixtures produce deterministic 400x300 base and
   alternate-headroom PNGs;
4. every valid fixture has a nonzero base-versus-tone-map RGB difference;
5. every invalid fixture rejects gain-map metadata/application without a usable
   tone-mapped output;
6. truncation and missing-runtime controls reject atomically;
7. sources and runtime files remain immutable;
8. forward/reverse reports and stable scientific payloads are byte-exact;
9. formal output residue, network, encoding, fitting and hidden-target reads are
   zero.

Any failure closes P282 without rebuilding libavif, changing fixtures,
relaxing strictness, selecting a different headroom, editing metadata, changing
PNG depth or substituting a backend.

## Claim ceiling

At most `PASS_PRIVATE_LIBAVIF_GAINMAP_WINDOWS_RUNTIME`: private Windows
mechanical recognition, base decoding and official gain-map application for
three exact official fixtures plus three exact invalid controls. A pass may
open only a separately frozen explicit bridge into the existing P87 boundary.
It does not establish complete ISO 21496-1 conformance, arbitrary AVIF/HEIF,
HDR quality, metadata preservation, display correctness, public dependency or
package, schema/capability/product admission, film-stock evidence or candidate
3.
