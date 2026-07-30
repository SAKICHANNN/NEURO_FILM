# U5.R2BC1 AO6 Linear Density Grain Result

BC1 replaces the rejected encoded-RGB additive grain with one shared
optical-density field applied to all three linear-RGB channels. A per-pixel
analytical bright-end guard prevents cube escape; no output clip is used as a
safety mechanism. AO6 colour, the 16-image population, density sigma `.0225`
and optional simple-halation strength `.14` were frozen before rendering.

Two complete reports are byte-identical at SHA-256
`abd7b09a6f6d2bbd7ab4ed9b90bdffe5a8494df1af560966aad8ea6bcd982165`.
All automatic gates pass: median changed support is 79.09%, median
bright-guard use is 0.416%, worst linear chromaticity drift P99.9 is numerical
noise at `2.22e-16`, worst encoded change P99.9 is `.04063`, all 16 rows
support halation, and new raw clipping is zero.

The original contact-sheet order was visible in implementation and is excluded
from preference evidence. A second per-sample randomized mapping was committed
at SHA-256 `5885c8b0...a9e4d`; 27 label choices and severe observations were
committed before reveal. The three decoded preference counts are:

- round 1: grain `3`, grain+halation `3`, colour-only `3`;
- round 2: grain `2`, grain+halation `2`, colour-only `5`;
- round 3: grain `3`, grain+halation `4`, colour-only `2`.

No arm reaches the frozen six-vote threshold in any round. Full 1:1 crops show
no confirmed severe artifact, coloured speckle, posterization, banding,
geometry corruption or objectionable halo, but safety alone is not product
value. BC1 therefore closes without strength, threshold, mapping or additional
round rescue. The primitive remains mechanically useful research code only; it
is not measured film grain, a stock response, calibration or product
promotion.
