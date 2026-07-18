# U1.5C libultrahdr real-reference fail-closed regression contract

Date: 2026-07-18

Node: `ULT > U1 > U1.5 > U1.5C`

Status: frozen before fixture import and formal regression execution

## Question

Do exact, rights-cleared real gain-map JPEG references from Google's
libultrahdr project fail closed before raster pixel decoding and renderer output
through the existing multi-frame boundary, without claiming semantic gain-map
recognition that the current code does not implement?

## Parent evidence and pre-freeze observation

U1.5A/B reject recognized HDR/gain-map signals but do not claim complete MPF or
ISO 21496-1 parsing. U1.2C independently rejects every raster with more than
one exposed frame/page before pixel conversion.

The Android Ultra HDR specification states that a gain-map JPEG is backward
compatible, stores its gain map as an additional MPF image, and may be signaled
through GContainer or MPF/XMP metadata. The Google libultrahdr repository is a
reference codec and contains two small Apple gain-map JPEG test assets under a
CC-BY-4.0 data licence.

A read-only development check at the pinned upstream commit observed both
assets as `MPO` with `frame_count=2` and saw the existing U1.2C rejection. This
observation establishes DoR and the expected regression only; it is disclosed
and cannot be presented as unseen confirmatory discovery.

Primary sources:

- https://developer.android.com/media/platform/hdr-image-format
- https://github.com/google/libultrahdr
- https://github.com/google/libultrahdr/tree/ad4a92eea0d2f39f18b5ecae3165fdd56c6a478b/tests/data

## Frozen inputs and placement

Import unmodified only the two SHA-256-pinned assets in the committed config:

- `apple_gainmap_old.jpg`;
- `apple_gainmap_new.jpg`.

Place them under `tests/fixtures/u1_5c_libultrahdr/` with a source manifest and
the upstream CC-BY-4.0 licence text. No other upstream file, executable or
dependency is imported. Fixture bytes and source revision must match before a
test may interpret them.

## Frozen gates

For each exact fixture:

1. source kind is raster, format is `MPO`, and frame count is exactly two;
2. no `unsupported_dynamic_range` warning is required or claimed;
3. `load_working_image` raises the existing U1.2C multi-frame rejection before
   returning a `WorkingImage`;
4. integrated `scripts/render_film.py` returns nonzero and creates no output;
5. ordinary single-frame JPEG regressions remain green.

The implementation may add fixture/provenance validation and tests only. If an
exact pinned fixture silently decodes or renders, that is a valid fail and may
open a separately frozen parser/ingress repair. If both already reject, do not
add MPF/ISO parsing merely to duplicate U1.2C.

## Branches

- **Pass:** retain the real-reference regression as evidence that these exact
  MPF gain-map files fail closed through the multi-frame boundary.
- **Fail:** freeze the smallest ingress repair before changing source code.
- **Input/licence/hash mismatch:** invalid result; repair provenance only.

No branch opens HDR reconstruction, gain-map application, tone mapping, HEIF,
complete MPF/ISO 21496-1 recognition, wide-gamut renderer integration or a
claim that every gain-map file is detected.

## DoD and claim ceiling

Commit this contract/config before importing fixtures. Then verify exact hashes,
licence/source manifest, focused ingress/renderer tests, the complete CPU suite,
compile and diff checks. Propagate the measured result and commit/push it as a
separate leaf.

Claim ceiling: two exact libultrahdr real references fail closed through the
existing multi-frame raster boundary; no general HDR/gain-map format support or
complete semantic detection claim.
