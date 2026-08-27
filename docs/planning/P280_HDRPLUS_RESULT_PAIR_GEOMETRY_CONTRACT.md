# P280 — HDR+ official result-pair geometry contract

Status: **frozen before first official `final.jpg` pixel decode**  
Parents: P269 exact source lock; P279 exact full-resolution merged-DNG ingress  
Claim ceiling: private two-version paired-source eligibility only

## Question

For the exact Google HDR+ burst `0382_20150912_131056_666`, are each release's
official `merged.dng` and `final.jpg` results demonstrably the same full-view
image, rather than merely colocated files? A pass may open only a separately
frozen low-capacity tone/render representation D0 on this already-consumed
single burst.

## Frozen roles

- Version 20161014:
  - observation: `result_20161014/merged.dng`, SHA-256 `ab3a6bf3...624f2`;
  - target: `result_20161014/final.jpg`, 5,596,112 bytes, SHA-256
    `d64cfc02cea91205b7fa06d26dd4f023ab6f8d6ef05df75d560e9e7264fd1246`.
- Version 20171023:
  - observation: `result_20171023/merged.dng`, SHA-256 `3a21417f...fbdd`;
  - target: `result_20171023/final.jpg`, 5,393,899 bytes, SHA-256
    `70714bcb7104b69a2fde8ecc1c8c798da5d129f16bfb92bcfef2ef764845c3a3`.

P279 fixes each observation as a 3024x4032x3 `linear_srgb` WorkingImage.
Target pixels remain unread until this contract is committed.

## Frozen representation and gates

For each pair, decode the observation through unchanged `load_working_image`
and the target through Pillow RGB. Require exact 3024x4032 dimensions. Convert
both to fixed Rec.709 luma, area-downsample to 256x192 float32, then compare
Sobel gradient-magnitude Pearson correlation.

Primary gates, required for both versions:

1. identity correlation `>=0.70`;
2. identity minus horizontal-flip correlation `>=0.20`;
3. identity minus a fixed 16-column circular-shift correlation `>=0.15`;
4. finite/nonconstant arrays, exact source identities and source immutability.

Two fresh processes enumerate versions forward/reverse; canonical reports must
be byte-exact. No crop, registration, search, affine warp, target-guided mask,
alternate luma, threshold change, resize change or same-pair rescue is allowed.

## Stop and claim boundary

Failure closes this exact pairing assertion. Pass proves only that these two
official result pairs are same-view paired observations. It does not establish
HDR truth, color/radiometric equivalence, tone-map quality, independent scenes,
generalization, commercial rights, package/schema/capability, product admission
or candidate-3 change.
