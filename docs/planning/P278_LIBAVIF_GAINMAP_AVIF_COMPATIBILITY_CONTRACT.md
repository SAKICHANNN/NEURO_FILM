# P278 libavif gain-map AVIF compatibility contract

## Node

`ULT > U1 > U1.5 > P278`

## Question

Can the unchanged pinned libultrahdr v2.0.0 Windows decoder and unchanged P87
absolute-Rec.2020 MatchView ingress consume valid gain-map AVIF files from the
official AOMedia libavif test corpus, while rejecting fixtures whose gain-map
association or metadata is invalid?

This is a format/runtime compatibility test. It is not an HDR-quality test and
does not reopen the P277 JPEG pair metric, the production loader or candidate
3.

## Prospective source freeze

- Official repository: `https://github.com/AOMediaCodec/libavif`.
- Frozen main commit: `b6fb1860837541d6e0c94386f2d09ba7c2341770`.
- The official `tests/data/README.md` identifies the selected files and states
  that each selected asset uses the same licence as libavif. The root licence
  permits source and binary redistribution with notice and disclaimer.
- Valid formal fixtures, all unread before this contract:
  - `seine_sdr_gainmap_srgb.avif`;
  - `seine_hdr_gainmap_small_srgb.avif`;
  - `seine_sdr_gainmap_big_srgb.avif`.
- Invalid/ignored-gain-map formal controls, all unread before this contract:
  - `seine_sdr_gainmap_notmapbrand.avif` — missing the `tmap` brand;
  - `seine_hdr_gainmap_wrongaltr.avif` — reversed `altr` preference;
  - `seine_sdr_gainmap_gammazero.avif` — zero gain-map gamma.
- `seine_hdr_gainmap_srgb.avif` is excluded from formal evidence. A bounded
  pre-contract infrastructure probe downloaded that one file and the existing
  executable returned `input uhdr image does not contain any valid images`.
  The file and all probe outputs were deleted. It cannot be selected, scored
  or used to tune the formal cohort.

Only the six selected AVIF objects, exact repository README/licence bytes and a
canonical source manifest may persist under the repository-relative P-backed
P278 data root. No repository clone or unrelated corpus file may persist.

## Frozen runtime

- Decoder: the exact libultrahdr v2 executable already frozen by P88/P277,
  SHA-256 `cffdfcae90b2e999260877b646c36c1d04b87d81745c12417c089ca7099d5df0`.
- Decode command: `-m 1 -j {input} -o 0 -O 4 -z {output}`.
- Successful output is interpreted only through unchanged P87.
- No decoder rebuild, codec dependency installation, alternate backend,
  encoder roundtrip or file repair is permitted.

## Execution

1. Fetch only the exact eight frozen Git objects (README, licence and six AVIF
   files) through the repository-relative P-backed data entry.
2. Bind byte count, SHA-256 and Git blob SHA-1 for every retained object before
   formal decoder execution.
3. Run two fresh processes in opposite fixture order. Canonical reports sort
   records by file name before serialization.
4. For every valid fixture, invoke the unchanged decoder. If it succeeds,
   derive dimensions through the existing Pillow AVIF reader, require exact
   RGBA16F byte length and pass the bytes through unchanged P87.
5. For every invalid fixture, require the decoder to return nonzero and leave
   no usable output.
6. Re-run the production loader refusal for every selected AVIF and one
   64-byte truncation atomicity control.
7. Delete all decoded payloads and temporary outputs. Network access during
   formal execution is zero.

## Frozen gates

All gates are conjunctive:

1. official commit, README, licence, object hashes and retained-file set are
   exact;
2. all three official-valid fixtures decode successfully;
3. every successful payload has the exact `width * height * 4 * 2` byte count
   and passes unchanged P87 finite, alpha, profile and absolute-light rules;
4. all three official-invalid/ignored-gain-map fixtures fail without a usable
   HDR payload;
5. sources remain immutable, production ingress rejects every selected AVIF,
   and truncated input fails atomically;
6. forward/reverse reports and stable scientific identities are exact;
7. encoder calls, alternate decoder calls, source substitutions, hidden
   targets, fitting, training and retained decoded payloads are zero.

Any failure closes P278 without changing the decoder build, codec stack,
fixture set, metadata, output transfer, threshold or P87 contract.

## Claim ceiling

At most `PASS_PRIVATE_LIBAVIF_GAINMAP_AVIF_DECODER_COMPATIBILITY`: private
Windows compatibility for three exact official libavif gain-map AVIF fixtures
and three invalid controls through the pinned libultrahdr-v2-to-P87 chain.
No complete ISO 21496-1 conformance, HEIF support, arbitrary AVIF/media, HDR
quality, display validation, public dependency/package/schema/capability,
default-loader change, product admission, film-stock claim or candidate-3
change is allowed.
