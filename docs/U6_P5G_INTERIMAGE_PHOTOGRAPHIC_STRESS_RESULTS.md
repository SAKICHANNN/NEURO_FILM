# U6.P5G interimage photographic stress

Decision: **automatic and autonomous severe-artifact pass; development only**.

Two full 18-image/nine-make runs from `7ebe9132` are byte-identical:

- report `521bebd6...3d969`;
- overview `4c11beb2...b1c2f`;
- one-to-one worst patches `c925beab...d9ea7`.

Maximum candidate-versus-P5C change is `0.002843`, population P95 is
`0.000157`, flat-region P99 is `0.000759`, and all 257/509-row outputs are
exact. There are zero isolated excursions and zero new hard boundaries.
Autonomous inspection found no severe coloured speckle, posterization,
secondary contour, edge echo, flat block, seam or geometry change.

The effect is visually weak. Safety does not establish value, preference or
authenticity. A separately frozen cost-and-visible-value audit must pass before
runtime integration; otherwise the simpler P5C path remains authoritative.
