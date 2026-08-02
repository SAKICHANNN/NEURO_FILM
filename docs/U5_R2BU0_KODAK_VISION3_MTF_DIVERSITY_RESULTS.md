# U5.R2BU0 Kodak VISION3 MTF diversity

## Decision

The exact first-party 50D, 250D and 500T MTF traces pass the frozen
source-domain diversity gate. All three stock pairs are material in at least
two of three layers, while the earlier characteristic-shape result remains at
zero material pairs. This opens only an explicit physical-prior compiler.

## Evidence

- two byte-identical formal reports: `fafed347...6020b`;
- stable evidence ID: `7056a688...29c6`;
- 50D/250D layer RMSE: `.182651/.168821/.087928`;
- 50D/500T: `.092136/.111298/.306616`;
- 250D/500T: `.274482/.066042/.269904`;
- all source, trace, axis, ink, range, monotonicity, coverage, uncertainty and
  parent-negative-control gates pass;
- all three source overlays were inspected and follow the correct solid curves.

An initial diagnostic run exposed a strict-float endpoint bug. The repair uses
the already-frozen two-pixel trace uncertainty in log-frequency; it changes no
source point, common frequency, metric or diversity gate.

## Boundary

This is a manufacturer source-domain spatial signature, not real-roll
calibration, colour response, scanner response, a digital-to-film look or a
product profile. Photographic rendering and the B0/AO6 default remain unchanged.
