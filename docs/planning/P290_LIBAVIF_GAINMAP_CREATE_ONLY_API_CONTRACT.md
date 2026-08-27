# P290 private libavif gain-map create-only API contract

Date frozen: 2026-08-27

## Purpose

Turn the exact P289 official-runtime encoder mechanism into one private,
caller-driven and source-hash-bound publication primitive. The caller supplies
already-rendered HDR-PQ and SDR-sRGB PNG endpoints; this leaf does not choose or
derive an SDR tone policy.

## Frozen interface

`encode_gainmap_avif_create_only_v1(...)` accepts:

- absolute HDR and SDR endpoint paths plus their expected SHA-256 identities;
- an absolute, absent destination path;
- the exact retained `avifgainmaputil.exe` path and expected SHA-256;
- explicit base/alternate CICP triples and headrooms.

The v1 profile is fixed to P289: HDR base CICP `1/16/0`, SDR alternate CICP
`1/13/0`, headrooms `1.3/0.0`, 12-bit YUV444 color and gain map, quality 100,
gain-map downscaling 1, speed 10, grid 1x1 and jobs 1. Inputs must be distinct
uint16 three-channel PNGs of the same positive dimensions. The function stages
beside the destination, validates official metadata after encoding, then uses
the existing exFAT-compatible same-volume create-only publisher. It returns a
plain deterministic receipt and never deletes or overwrites caller inputs or a
foreign destination.

## Frozen formal fixture and gates

Formal execution reconstructs the exact P289 HDR/SDR endpoints from the exact
P278 fixture before invoking the new function. Two fresh forward/reverse
processes must pass:

1. exact P289 parent, source, runtime, endpoint PNG and endpoint pixel hashes;
2. output bytes and SHA exactly equal P289's 726,506-byte media
   `94556d9f...f6da70`;
3. official metadata exposes base/alternate headrooms `1.3/0.0` and exact CICP;
4. independent official base decode and headroom-0 tonemap reproduce the exact
   P289 decoded endpoint pixel hashes;
5. receipt bytes, full report bytes and candidate media are exact across both
   processes;
6. source-hash mismatch, wrong endpoint dtype/layout/dimensions, wrong CICP,
   wrong headroom, nonfinite headroom, missing runtime, wrong runtime hash and
   foreign destination all reject before publication;
7. forced encoder failure leaves no destination or sibling stage;
8. endpoint/runtime/source bytes remain immutable and owned scratch/stages are
   empty after execution.

Any failure closes this exact API/profile. No encoder parameter, source,
threshold or error-tolerance rescue is allowed.

## Claim ceiling

At most this establishes one private Windows/Python create-only API around the
exact P289 profile and official libavif runtime. It does not establish an SDR
tone policy, complete ISO 21496-1, arbitrary AVIF/HEIF or HDR quality, display
validation, public dependency/API/package/schema/capability, default loader,
product admission, stock evidence or candidate 3. This is the terminal P289
wrapper leaf; adjacent wrappers and target runtimes do not open automatically.
