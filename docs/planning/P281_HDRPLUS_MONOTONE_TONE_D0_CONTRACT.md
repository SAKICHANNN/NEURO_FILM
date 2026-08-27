# P281 — HDR+ shared monotone tone representation D0

Status: **frozen before any target-dependent fit**  
Parent: P280 exact same-view paired-source eligibility  
Claim ceiling: one-burst private representation D0 only

## Roles and representation

- Development pair: P280 version 20161014.
- Confirmation pair: P280 version 20171023; its target remains unread for P281
  until the development transform is frozen in memory.
- Source representation: unchanged merged-DNG WorkingImage converted by the
  existing `working_image_to_srgb_float` path.
- Target: corresponding official `final.jpg` RGB divided by 255.
- Development fit/heldout partition: exact pixel checkerboard `(y+x)%2`, with
  fit parity 0 and heldout parity 1.
- Candidate: three independent, nondecreasing 17-knot piecewise-linear channel
  curves on fixed `linspace(0,1,17)`, fit using isotonic regression from only a
  fixed stride-8 subset of development-fit pixels.
- Same-budget control: three independent unconstrained 17-knot curves whose
  knot values are medians in the same fixed source bins and the same fit rows.

## Frozen metrics and gates

RMSE is computed in normalized encoded RGB on development heldout and full
confirmation. Candidate must:

1. improve RMSE over identity by at least 20% on both rows;
2. be noninferior to the same-budget control on both rows (`candidate <= 1.05x`);
3. have nondecreasing knots, finite `[0,1]` output and zero newly created exact
   0/1 components from strictly interior source components;
4. reproduce byte-exact reports under forward/reverse row enumeration.

The confirmation target is read only after fitting and hashing both development
curves. Fit/heldout overlap must be zero. No knot/grid/partition/metric/model/
threshold rescue is allowed.

## Boundary

Pass would show only that a tiny global monotone tone representation transfers
between two official result versions of one already-consumed burst. Failure
closes the exact family. Neither outcome proves HDR truth, independent-scene
generalization, colorimetric calibration, public/package capability, product
admission or candidate-3 change.
