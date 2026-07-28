# P163 — SDR JPEG/TIFF file-format matrix contract

P163 tests whether the existing `WorkingImage` file adapter executes its
declared compressed and high-bit-depth SDR raster paths through a complete
reference-match transaction. It does not add a decoder or broaden supported
colour rails.

## Frozen matrix

Every case uses one deterministic 2048-by-1536 reference, one distinct source,
default identity fallback and two fresh detached-worker runs on exact candidate
`6ae40df1ee87e7e994b0571da6d81c5e24e26ff6`.

| Case | Input | Output |
|---|---|---|
| `jpeg8` | RGB8 JPEG | RGB8 JPEG |
| `tiff8` | RGB8 TIFF | RGB8 TIFF |
| `tiff16` | profiled RGB16 TIFF | profiled RGB16 TIFF |

Each decoded input must report display-linear linear-sRGB. Output format and
bit depth must exactly match the case.

## Frozen gates

- Six workers complete within 60 seconds each, below 1.5 GiB peak RSS, with
  per-case two-run peak ratio at most 1.15.
- Output bytes, recipe and path-normalized report are exact within each case.
- Every case returns identity fallback.
- No staging temporary, orphan process or detached worktree remains.

JPEG identity fallback refers to decoded working pixels before deterministic
re-encoding; it does not imply input JPEG bytes equal output JPEG bytes.

## Claim ceiling

A pass covers only these three local Windows/Python SDR raster transactions.
It is not arbitrary JPEG/TIFF metadata support, HEIF/AVIF/JXL, RAW, HDR,
gain-map, video, native/device or product-readiness evidence.
