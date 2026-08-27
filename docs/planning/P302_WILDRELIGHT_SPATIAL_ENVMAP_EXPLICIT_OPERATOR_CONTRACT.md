# P302 — WildRelight spatial environment-conditioned explicit operator

Status: **prospectively frozen before new-scene member acquisition or pixels**  
Date: 2026-08-28  
Parent: P286 source feasibility; P287 exact global-gain family closed

## Question and claim ceiling

Does a group-trained, low-capacity explicit spatial gain operator use a paired
capture-time HDR environment map to predict illumination change on wholly new
WildRelight scene groups better than equal-budget image-only, global-gain and
identity controls?

This is a private candidate-3 precursor D0. Candidate count remains `2/3`
unless both development and untouched confirmation pass and a later, separately
frozen A4/A5 authorization leaf passes. It is not relighting quality,
radiometric calibration, a source-free shared look, stock evidence or product
admission.

## Frozen source and group roles

- Official dataset `Lez/wildrelight`, revision
  `90ab579145da9f3ea998d706c8defd0d4b87641f`, CC BY 4.0.
- Use `small-aligned` only. The consumed P287 `lake` scene is excluded.
- Rank the remaining exact P286 scene names by
  `SHA256(revision + ":p302:" + scene)`.
- First 12 groups are training, next six development, next six untouched
  confirmation, final five reserve. Roles are enumerated in the frozen config.
- Each scene contributes only disjoint one-way pairs `0->1`, `2->3`, `4->5`.
  A photo is never both source and target within an evaluation role.
- A source-lock amendment must freeze every selected `meta.json`, photo EXR and
  environment-map EXR path, byte count and LFS SHA-256 before any selected EXR
  request. Metadata-only official API reads are allowed for this amendment.
- Confirmation member bodies and pixel decodes remain zero unless every
  development gate passes. Reserve member bodies remain unread in all cases.

## Frozen representation and explicit operator

All statistics use the central 90% crop and `eps=2^-16`.

1. Decode source photos and source/target environment maps as finite,
   nonnegative RGB float arrays with pinned OpenEXR 3.4.15.
2. Source descriptor: a 4x4 grid of per-channel median
   `log2(source_photo + eps)` values (48 scalars).
3. Identifying descriptor: real spherical-harmonic coefficients through degree
   two, evaluated at equirectangular pixel centres with solid-angle weights,
   for each RGB channel. Use target-minus-source coefficients normalized by
   the positive source degree-zero coefficient (27 scalars).
4. The candidate feature is bias + 48 source-grid + 27 environment-difference
   scalars. Its 48 outputs are a 4x4x3 log2-gain grid.
5. Fit one shared multi-output ridge model on training groups only. Standardize
   non-bias features from training rows only; `lambda=0.01`; solve in float64;
   no epoch, seed, capacity or hyperparameter search.
6. Clip predicted log2 gains to `[-4,4]`, bilinearly interpolate the 4x4 grid
   at pixel centres, and multiply the source photo. No target statistic,
   clipping, tone map, warp, postprocess or per-scene fit is allowed.

## Frozen controls

- **Equal-budget image-only ridge:** identical 76-by-48 parameter shape,
  training rows, standardization and ridge solve. Replace the 27 environment
  features with the first 3x3 orthonormal DCT coefficients per RGB channel of
  the 4x4 source log-grid.
- **Global environment gain:** exact P287 spherical per-channel mean ratio.
- **Identity:** unchanged source photo.
- **Cyclic identifying control:** unchanged trained candidate, but replace each
  target environment map with the next target-environment role in the same
  scene (`1->3`, `3->5`, `5->1`).
- **Target-exposed ceiling only:** per-row 4x4x3 median log-ratio grid. It may
  measure representation recovery but cannot build or authorize the candidate.

The strongest legitimate control is the lowest-error result among image-only,
global-gain and identity for each row.

## Target-unread and leakage rules

- Training targets may be read only for training-label construction.
- The shared candidate and image-only model bytes, normalization statistics,
  member roles, row order, features and all candidate/control output hashes
  must freeze before the first development target photo is decoded.
- For every development or confirmation row, build and hash candidate and
  controls from the source photo and permitted environment maps before decoding
  that row's target photo.
- Confirmation EXR requests/decodes are forbidden unless the complete
  development decision passes. No refit, recalibration or threshold update is
  allowed after development targets become visible.
- Scene overlap across train/development/confirmation/reserve must be zero.

## Frozen metric and gates

Score RGB-component RMSE in `log2(value + eps)` space on components where source
and target are finite and strictly positive. Report every row and scene slice.

Development and confirmation each require all of:

1. exact source/member/model/normalization/role/order/replay identities and
   zero scene overlap;
2. candidate beats the strongest legitimate control on at least 75% of rows,
   median reduction at least +10%, worst reduction at least -10%, and at least
   five of six scene-median reductions are positive;
3. candidate beats cyclic control on at least 75% of rows with median reduction
   at least +10%;
4. median recovery of target-exposed grid-oracle gain is at least 30%;
5. valid component fraction is at least 80% for every row;
6. finite output, source-zero preservation, no new exact 0/1 boundary, and
   candidate/source p95 spatial-gradient ratio no greater than 1.5;
7. candidate/model builds read no evaluation target, confirmation reads remain
   zero before development pass, reserve reads remain zero, and two fresh
   committed-head reports are byte-identical.

Any development failure closes this exact representation/model/control/role
family before confirmation access. Any confirmation failure closes it without
threshold, feature, ridge, grid, role, scene, fallback or postprocess rescue.

## Next boundary

A two-stage pass opens only a separately preregistered source-only uncertainty
and fallback/A4-A5 authorization leaf plus versioned private artifact/schema/
receipt work. Candidate 3 is not consumed by this contract alone.
