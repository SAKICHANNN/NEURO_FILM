# P287 — WildRelight environment-conditioned HDR D0 contract

Status: **prospectively frozen before any selected EXR request or pixel decode**  
Date: 2026-08-27  
Parent: P286 rights-clear paired-HDR source feasibility

## Question and claim ceiling

Test one materially new capture-time observation: does a paired HDR environment
map identify enough of the illumination change to improve an explicit global
RGB radiance operator between temporally aligned HDR photographs?

This is a one-scene private mechanism D0, not a relighting system, natural-scene
generalization, calibrated radiometry, product capability or automatic
single-reference candidate. Passing may open only a separately frozen,
group-disjoint multi-scene confirmation. Candidate 3 remains unconsumed.

## Frozen source and roles

- Official dataset: `Lez/wildrelight` at revision
  `90ab579145da9f3ea998d706c8defd0d4b87641f`, CC BY 4.0.
- P286 exact inventory: 30 scene groups and no published DNG bodies.
- Scene selection is the lexicographic minimum of
  `SHA256(revision + ":" + scene)` over the P286 scene list. The frozen result is
  `lake`, digest
  `18cfbdfa30f4cd7ecf2ccb2c37a2638c64375cd512ea227c5ab546a793da67b5`.
- Variant: `small-aligned` only. Selected payload is one 3,987-byte `meta.json`,
  six `photo/timeN_hdr.exr` files and six `envmap/timeN_envmap.exr` files,
  total 108,447,512 bytes. Every member size and SHA-256 is frozen in the
  execution config before acquisition.
- Rows are the ten adjacent directed pairs: `0→1` through `4→5` and their
  reverse directions. There is no row selection after pixels are visible.
- For each row, candidate and control arrays must be built and hashed from the
  source photo plus source/target environment maps before the target photo is
  decoded. A photo used as source in another row does not waive this row-local
  target-unread rule.

## Frozen decode and operator

- Decode with the existing pinned OpenEXR 3.4.15 CPython 3.12 Windows wheel.
- Require one non-deep RGB image per member, matching dimensions within each
  role, finite samples and nonnegative spherical environment-map support.
- Use equirectangular pixel-center `sin(theta)` row weights. Compute one
  spherical weighted arithmetic mean for each RGB channel.
- Candidate gain is `mean_rgb(target_envmap) / mean_rgb(source_envmap)` and the
  candidate is the source HDR photo multiplied channelwise by that gain. There
  is no clipping, learned fit, spatial warp, tone map, target-photo statistic or
  postprocessing.
- Legitimate controls are exact identity and one spherical BT.2020-luma gain.
  The strongest per-row control is the lower-error of those two.
- The capacity-identical identifying-information control uses the target
  environment map from the next cyclic time index rather than the true target
  index. The permutation is frozen and cannot be changed after scoring.

## Frozen score and gates

Evaluate the central 90% crop. A component is valid only where source and target
are finite and strictly greater than `2^-16`. Score RGB component RMSE in
`log2(value + 2^-16)` space. All ratios and improvements are computed from the
same valid mask.

All gates are conjunctive:

1. exact source/member hashes, scene-selection digest, metadata roles and
   row-order replay pass;
2. target-photo decodes before candidate/control freeze equal zero for every row;
3. minimum valid component fraction is at least `0.80`;
4. candidate beats the strongest legitimate control on at least `8/10` rows,
   median RMSE reduction is at least `+5%`, and worst reduction is at least
   `-5%`;
5. candidate beats the capacity-identical cyclic-envmap control on at least
   `8/10` rows with median RMSE reduction at least `+10%`;
6. candidate output is finite, its sign mask equals the source sign mask, and
   no clipping or new boundary is introduced;
7. paired photo/envmap capture-time separation is at most 60 seconds at every
   time point;
8. two fresh committed-head forward/reverse reports are byte-identical.

Any gate failure closes this exact scene/operator/statistic/metric family.
There is no scene, row, statistic, crop, epsilon, metric, threshold or
postprocessing rescue. A pass still does not authorize full-corpus acquisition;
the next leaf must freeze new scene groups before their pixels are read.
