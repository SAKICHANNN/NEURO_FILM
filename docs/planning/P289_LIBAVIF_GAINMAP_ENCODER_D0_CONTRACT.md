# P289 official libavif gain-map encoder D0 contract

Date frozen: 2026-08-27

## Question

Can the already retained official libavif 1.4.2 Windows runtime create a valid
gain-map AVIF from independently materialized HDR and SDR endpoints, then
reproduce both endpoints within a fixed 12-bit code-domain error budget?

This is a mature explicit HDR container/runtime experiment. It does not change
the renderer, input defaults, P87, P282-P284, stock evidence, or candidate 3.

## Frozen source and roles

- Runtime: exact P282 official libavif v1.4.2 Windows release and its retained
  `avifdec.exe` and `avifgainmaputil.exe` identities.
- Source: the lexicographically smallest SHA-256 among the three exact valid
  P278 fixtures, `seine_hdr_gainmap_small_srgb.avif`, SHA-256
  `573a67fdd581f6e634da198a819cc92a071539dfb720c5d7dfdf02e4e87a0346`.
- The source base is HDR at headroom `1.3`; its alternate is SDR at headroom
  `0.0`.
- Before candidate encoding, materialize and hash the HDR base with the fixed
  P282 `avifdec` command and the SDR alternate with the fixed P282 `tonemap`
  command. These endpoint pixels are the only encoder inputs.

## Frozen candidate

Invoke official `avifgainmaputil combine` with HDR as base and SDR as
alternate, fixed base CICP `1/16/0`, alternate CICP `1/13/0`, gain-map
downscaling `1`, color and gain-map quality `100`, 12-bit color and gain map,
YUV444 for both, maximum headroom `1.3`, speed `10`, grid `1x1`, and one job.

Decode the candidate base through official `avifdec`; apply its gain map at
headroom `0.0` through official `avifgainmaputil tonemap`. Compare decoded
uint16 BGR pixels to the frozen HDR and SDR endpoint pixels after exact
bit-depth normalization to 12-bit codes.

## Frozen gates

All must pass:

1. parent manifests, runtime binaries, fixture and local execution files are
   exact;
2. endpoint build succeeds before candidate encoding, dimensions are
   `400x300`, both endpoints are uint16 RGB-family arrays, and their pixel
   hashes freeze before `combine`;
3. `combine`, metadata inspection, base decode and alternate tonemap succeed;
4. candidate is recognized as a gain-map AVIF with HDR base headroom `1.3`
   and SDR alternate headroom `0.0`;
5. HDR and SDR endpoint median absolute 12-bit code error are each at most
   `1`, p95 at most `4`, and maximum at most `8`;
6. at least 1% of endpoint RGB components differ, proving material gain-map
   action;
7. repeated candidate media, decoded pixels and full scientific payload are
   exact across two fresh forward/reverse processes;
8. truncated input, missing endpoint and foreign pre-existing destination
   reject without partial overwrite;
9. fixture/runtime bytes are immutable and owned scratch is empty afterward.

The numeric ceilings are fixed before candidate encoding as bounded code-domain
budgets for two cascaded 12-bit RGB/YUV conversions, with tighter median and
p95 gates than the maximum-error ceiling.
No threshold, CICP, endpoint, headroom, quality, chroma, bit-depth, fixture or
backend rescue is allowed after observation.

## Decision and claim ceiling

- PASS opens only a separately frozen consumer-facing integration question.
- FAIL closes this exact libavif 1.4.2 encoder/profile/fixture family.

At most this leaf can establish private Windows runtime/container mechanics on
one official fixture. It cannot establish complete ISO 21496-1 conformance,
arbitrary AVIF/HEIF or HDR quality, display validation, public dependency/API,
package/schema/capability, product admission, stock evidence or candidate 3.
