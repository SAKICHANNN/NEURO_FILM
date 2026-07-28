# U5.R2AJ0B2 — lane-aware HaldCLUT integrity confirmation

## Parent and question

Parents:

- AJ0A licensed bounded-source pass;
- AJ0B v1 frozen failure;
- exact diagnostic decode-profile inventory recorded only after v1 closed.

Question: does the one exact archive reproduce the now-frozen path-family
format, mode, bit-depth and Hald geometry profiles in two independent complete
processes?

This is a versioned correction to a false all-RGB inventory assumption. It
does not reopen or rewrite AJ0B v1.

## Fixed source and evidence identities

- exact bytes: `421602289`;
- MD5: `4742e362a70c1a1c0fb9042a17d285e1`;
- SHA-256:
  `0ffca81f30c72d7bbb85adab0ed98bbdeb0e1034cd6f5986f9f993bb66999dc2`;
- primary path hash:
  `83b22255e6eea6a99e779c72acf75917481f14d144d9e442b44b6d716256115f`;
- complete image-record hash:
  `5e59e95eb17661d3e08d731aba567382b958354fb81529246efc85de966d03e2`;
- output:
  `outputs/u5_r2aj0b2_haldclut_lane_aware_acquisition_v2/`.

The exact local archive may be reused, but each run must independently rehash
it, reopen the ZIP, read every CRC and decode all image members.

## Exact decode profiles

Every image must match exactly one frozen profile:

1. 226 Color PNGs other than exact Ektar: RGB, 8-bit x3, level 12,
   cube side 144;
2. exact `HaldCLUT/Color/Kodak/Kodak Ektar 100.png`: RGB, 8-bit x3,
   level 16, cube side 256;
3. 66 Black-and-White PNGs: L, 8-bit, level 12, cube side 144;
4. exact `HaldCLUT/Negative.png`: RGB, 16-bit x3, level 12,
   cube side 144;
5. exact `HaldCLUT/Hald_CLUT_Identity_12.tif`: TIFF RGB, 16-bit x3,
   level 12, cube side 144.

All must be single-frame and have no embedded ICC. No RGBA, palette, CMYK,
unmatched `L`, new path, new level, new bit depth or unassigned image is
allowed.

The 194-member primary universe stays byte-for-byte identical to AJ0A. B&W,
CreativePack, Negative and the identity file remain inventory/control
members, not primary candidates.

## Automatic pass and branches

B2 passes only if two new child processes produce byte-identical canonical
manifest/report files and all v1 path, CRC, README, inventory and acquisition
gates plus the exact SHA/profile/path/image-record gates pass.

- Any profile mismatch closes B2; no entry may be dropped or converted.
- A repeat mismatch closes B2; no decoder/thread rescue opens.
- A pass opens only a separately frozen AJ0C synthetic structural audit.
- No photographic or aesthetic render is allowed in B2.

The implementation must retain v1 schema/test behavior and add v2 behavior
without mutating the v1 config or decision.

## Future AJ0C precision boundary

AJ0C may use only the 194 primary Color CLUTs, which are all verified 8-bit
RGB PNGs. Its identity input must be generated analytically in float32/float64
under the frozen RawTherapee red-fastest semantics. It must not use Pillow
pixel samples from the 16-bit TIFF or Negative PNG. If a future leaf needs
those samples, `tifffile` or another bit-preserving decoder must first pass an
exact synthetic 16-bit RGB conformance test.

## Forbidden

No v1 rewrite, photograph render, aesthetic preview, stock/process/scan
interpretation, current-film fitting, pseudo-teacher use, training, LSM,
tracked asset extraction, product integration, release or distribution
decision.
