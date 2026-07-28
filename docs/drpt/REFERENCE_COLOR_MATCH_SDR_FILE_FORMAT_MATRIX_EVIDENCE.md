# P163 — SDR JPEG/TIFF file transaction evidence

## Result

The frozen 2048-by-1536 local Windows/Python matrix passes all three complete
file transactions twice:

| Case | Peak process-tree RSS | Worker time | Repeat ratio |
|---|---:|---:|---:|
| JPEG8 -> JPEG8 | 357,687,296 / 336,494,592 B | 4.5824 / 4.3355 s | 1.062981 |
| TIFF8 -> TIFF8 | 360,718,336 / 352,264,192 B | 4.3651 / 3.8746 s | 1.023999 |
| TIFF16 -> TIFF16 | 308,346,880 / 322,600,960 B | 4.2397 / 4.1009 s | 1.046227 |

Every input decodes as display-linear `linear_srgb`. Each output retains the
declared JPEG/TIFF format and 8/16-bit depth. Within each case, output bytes,
recipe bytes and normalized report identity reproduce exactly. All six
product decisions remain `identity-fallback`; no candidate is promoted.

The frozen config, runner and raw report SHA-256 identities are
`0248d87a...fafae`, `a3e8f533...8a569` and `e3d5d44e...9cea2`.
The temporary candidate worktree was removed successfully.

## Failed-attempt accounting

Two earlier invocations stopped at the frozen preflight because foreign
`vmmemWSL` exceeded the 8 GiB competing-process ceiling. One direct-entry
attempt exposed and fixed a missing repository-root import path. Two later
launches selected Python environments without NumPy or psutil. None of these
attempts started a pixel worker; their report/stderr hashes are bound in the
decision record. The successful run used the existing project Python 3.12
environment without installing or changing dependencies.

## Interpretation

P163 closes only the tested consumer file transaction matrix for ordinary
SDR JPEG8, TIFF8 and profiled TIFF16 on this Windows/Python host. It proves
format/depth retention, deterministic fallback artifacts and bounded local
resources for these exact cases.

It does not establish arbitrary metadata/profile conversion, HEIF/AVIF/JXL,
RAW, HDR/gain maps, video, native/mobile/Apple runtime, non-identity visual
quality or product readiness.
