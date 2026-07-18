# U1.5C libultrahdr real-reference regression results

Date: 2026-07-18

Decision: **pass — retain two exact real-reference fail-closed regressions**

## Result

Two unmodified gain-map JPEG references from Google `libultrahdr` at commit
`ad4a92eea0d2f39f18b5ecae3165fdd56c6a478b` are now pinned with their
CC-BY-4.0 licence and source manifest:

| Fixture | Bytes | SHA-256 |
|---|---:|---|
| `apple_gainmap_old.jpg` | 50,742 | `2e4310a0dd37a98678e057e25d936bbc4f936bb6fe17380c0ce99e58ecaa603b` |
| `apple_gainmap_new.jpg` | 50,824 | `492a94bd0636bcf15b3f560c68142a7783936c1597e78b0d0461d8be7fddc078` |

Both files inspect as raster `MPO` with two frames. The existing U1.2C
multi-frame boundary rejects each file before `WorkingImage` pixels are
returned, and the integrated renderer exits non-zero without an image or
metrics sidecar. No production source change or duplicate MPF parser was
needed.

## Verification

- implementation commit: `a31ea2e91f0eb9279730f09f2207f0ca685657a6`;
- frozen config SHA-256:
  `e36c5227e7c4430be02c9184c3d4cc063cc410400acc914dee6418010a6e336e`;
- source manifest SHA-256:
  `b06a3221ec7fc45d3765968fedcb7bf387d532629831dc3b794bd50d461b3567`;
- licence file SHA-256:
  `ba46d0677219de9bfec9ded6d99d3c21b8153702478fe4cb5b84f9c5b2ea859f`;
- 53 focused ingress, structured-scan and renderer tests pass;
- 652 complete CPU tests pass;
- Python compile check and `git diff --check` pass.

## Evidence boundary

The absence of an `unsupported_dynamic_range` warning is expected: current
inspection identifies the container as a two-frame MPO, not the gain-map
semantics. This result therefore does not establish complete MPF parsing,
ISO 21496-1 recognition, HDR reconstruction, gain-map application, tone
mapping or HDR output. Unknown single-frame or differently packaged gain-map
inputs remain outside the proven boundary.

The permitted claim is only: **these two exact pinned libultrahdr gain-map
references fail closed through the existing multi-frame ingress boundary and
cannot silently produce an SDR render.**

