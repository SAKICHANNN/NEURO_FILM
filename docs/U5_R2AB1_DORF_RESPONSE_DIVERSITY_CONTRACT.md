# U5.R2AB1 — DoRF explicit response diversity contract

## Question

Do exact, strict RGB triplets in the CAVE DoRF archive provide multiple
structurally safe and visibly non-basic **per-channel response priors**, after
controls that can explain ordinary tone and channel-gamma changes?

This is deliberately narrower than asking whether DoRF identifies a real
digital-to-film operator. It does not.

## DoR and fixed evidence

- AB0 decision must remain `limited_internal_research_source_pass`.
- Use the exact AB0-bound zip and parser.
- Use only complete, exact `Red/Green/Blue` suffix triplets whose three source
  scale labels agree and are `graph-log-log-neg` or `graph-log-log-pos`.
- Do not alias or repair Kodachrome-25 or any other incomplete group.
- Source names remain historical source labels, not verified current
  `film_stock_id` values.

## Fixed operator and population

For each candidate, apply its source red, green and blue normalized
irradiance-to-brightness curve independently to the corresponding encoded RGB
coordinate. Evaluate the Cartesian product of 11 frozen RGB levels
(`11^3 = 1,331` colours), 65 neutral levels and a 2,049-point derivative
domain from 0.02 to 0.98.

This treats source brightness as normalized display intensity only for a
synthetic Look Approximation. It does not assert scene-linear, scanner or
display calibration.

## Controls

For every candidate compare:

1. identity;
2. existing joint EV/WB/contrast/saturation basic fit;
3. one shared monotone curve equal to the triplet mean response;
4. independently fitted per-channel power curves;
5. exact per-channel monotone reproduction as a descriptive ceiling.

The last item must reproduce the candidate by construction and is evidence of
the representation limit: DoRF triplets contain no cross-channel interaction.
It is never a promotion baseline.

## Frozen gates

A candidate survives only when all of the following hold:

- median style Delta E76 from identity at least `7.0`;
- median residual from the shared curve at least `2.0`;
- median residual from per-channel power at least `1.0`;
- median residual from joint basic at least `4.9`;
- maximum neutral Lab chroma at most `25`;
- every channel derivative on the interior domain lies in `[0.01, 8.0]`;
- all raw outputs remain finite in `[0,1]`.

The bank additionally needs at least three survivors and minimum pairwise
survivor output RMSE at least `0.02`. Two complete evaluations must be exact.

## Branches

- No structural survivor: close the raw DoRF route without clamp, smoothing,
  neutral gauge, LUT, neural or image rescue.
- Basic collapse: retain only a tone/basic prior.
- Insufficient diversity: retain at most one descriptive response prior; do
  not create a stock bank or router.
- Full pass: open only a separately frozen AB2 internal Look-Approximation
  contract. It does not automatically authorize image rendering.

AB1 never opens fitting, training, LSM, calibrated stock claims or production
integration.
