# SF3.A0W NTIRE night geometry preflight preregistration

Date: 2026-08-21

SF3.A0W prospectively adopts the exact `as_shot_neutral` and `huawei_bounds`
key names observed only after SF3.A0V had already failed. It does not rewrite
or rescue SF3.A0V. The official MIT-licensed challenge baseline is pinned at
commit `3478fbb39449f5cba29483d4cbfe019013f8261f`; its projective matrix,
orientation, horizontal flip and `h_start,h_end,w_start,w_end` crop procedure
are copied as fixed facts in the contract before any RAW or Sony member is
read.

The first eight IDs from SF3.A0V's preregistered ordering are development-only
and can never enter candidate 2 fit/calibration/confirmation. Each RAW is
normalized, simply demosaiced and white-balanced only to create a
candidate-independent structural registration view. Official geometry then
maps it to 2000x2000. Fixed SIFT/RANSAC facts are compared against the correct
Sony row and a cyclic wrong-target control. All eight rows must satisfy the
absolute match, inlier, reprojection, near-identity and pair-separation gates.

Member payloads are streamed by bounded ZIP Range reads and are not persisted.
No colour operator is fitted and no target colour-quality metric is scored. A
pass opens only a separately preregistered fresh group-isolated candidate 2;
the bounded counter remains `1/3`. Any failure closes this source path before
candidate scoring, without parameter or row rescue.
