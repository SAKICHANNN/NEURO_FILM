# U5.R2BW0 — Fujifilm Velvia MTF revision signature

## Decision

Close the exact manufacturer-graph signature route. Two formal runs are
byte-identical at SHA-256 `4983bb1d...d40f5`, with stable evidence identity
`e19b113c...d5b7`, but the frozen source-calibration and all-stock separation
gates do not pass.

## Result

- Historical RVP and current RVP50 normalized log-response shapes are stable:
  RMSE `0.01465` against the frozen `0.03` maximum.
- Digitization uncertainty is bounded at `0.3510` of the separation gate.
- Current Velvia separates from VISION3 250D (`0.15273`) and 500T (`0.13658`),
  but not 50D (`0.05074 < 0.08`).
- The frozen global log-axis calibration does not fit the scanned chart axes:
  maximum residuals are `15.76 px` and `44.42 px`, above the `2 px` gate.
- Both Fujifilm revisions report diffuse RMS granularity `9` at a `48 µm`
  aperture; this is a stable scalar observation, not an NPS or grain renderer.

No piecewise-axis repair, threshold relaxation or same-source rescue is
allowed after this result. The exact traces remain first-party source priors,
not independent stock identification, calibrated physical response, RGB look
evidence or a product profile.
