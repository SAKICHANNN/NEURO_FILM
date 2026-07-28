# U5.R2AM1 SO(3) Coordinate-Curve Capacity Results

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AM1`  
Decision: **closed — analytic inversion passes, but capacity, cube and neutral
control do not**

## Integrity

- immutable contract hashes:
  - v1 `2f081922c090ce7dd6c24b2cbe570fed13a6085318fc527ba8d10ed9cb4b7e59`;
  - v2 `5c7a77eb6928a1761f51b0bf22b036cc861cf5911881c819c5ced44c19dac9be`;
  - executable v3
    `2ea192df0bf706edc62f9e5438d41340a61db93ed0d1fab168a78690e349746c`;
- formal fitting software commit:
  `bc049439599706c40c1930f61d71045420df1be3`;
- both fresh 33,345-byte reports are byte-identical at SHA-256
  `ed09300225f5d040f6073e8b8d32bef23e159689aadc901802608bd7138baf1a`;
- independently reconstructed repeat decision SHA-256:
  `f012f257ff76908c46ebb88073b5cafd840fd6b2e96c7dd19fc4e431cba5af7b`;
- persisted-report validation was corrected at commit
  `218b1fb` after canonical JSON key sorting exposed an order-only validator
  defect. The two reports, fits, metrics, thresholds and automatic checks were
  not changed;
- all three target prerequisites pass. No photograph, film pixel, external
  code, checkpoint, learned feature or new data was accessed.

## Result

The compact coordinate frame is not a consistent capacity improvement:

| Target | Candidate RMSE | Global curves | Gain vs global | Positive matrix + curves | Gain vs positive | K3 | Candidate/K3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| density cyan, s0.50 | 0.067568 | 0.070746 | 4.49% | 0.047605 | -41.93% | 0.033545 | 2.014x |
| positive warm, s0.35 | 0.021244 | 0.039534 | 46.27% | 0.024755 | 14.18% | 0.032525 | 0.653x |

Both targets miss the frozen `0.015` absolute ceiling. Density also fails all
three relative-capacity comparisons. Positive warm passes the relative
comparisons but cannot override its absolute failure.

The representation does preserve several useful properties:

- minimum Jacobian determinants are `0.10849` and `0.27529`, with zero sampled
  negative determinants;
- exact scalar inversion reaches maximum roundtrip errors
  `1.22e-15` and `9.99e-16`;
- SO(3), curve increments, replay, hue-neighbour, red-ramp and strength-path
  checks pass.

Those properties are insufficient. Both nonidentity fits leave the RGB cube:

- density: `[-0.003410, 1.017462]`;
- positive warm: `[-0.019694, 1.035173]`.

Both also fail the `0.03` neutral-axis spread gate at `0.11833` and `0.17040`.
No clamp or gamut projection was present.

## Interpretation

A right-handed orthogonal coordinate frame and three monotone scalar curves
provide an analytic inverse and positive orientation, but do not preserve the
axis-aligned RGB cube after nonidentity coordinate warping. One global frame
also lacks stable cross-channel capacity: it is only marginally better than
global curves for the density target and materially worse than the bounded
positive-matrix and K3 controls.

The branch closes without more rotations, curves, degree, restarts, target
changes, clamps, gamut projection, threshold changes or photographs. AM1
strengthens the broader conclusion that the current bottleneck is not the
absence of another compact explicit representation: O0 already supplies a
structurally safe higher-capacity family, while reusable film-style operator
identification remains unresolved.

## Claim ceiling

`repeat-exact clean-room synthetic negative representation-capacity and
structural-safety evidence for one bounded SO(3)-coordinate monotone-curve
operator`.

There is no photograph, film-pixel fitting, unpaired identification, stock,
calibration, preference, severe-safety or production claim.
