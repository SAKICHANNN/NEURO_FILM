# U5.R2BO6 luma-conditioned positive matrix result

BO6 is closed. Two reports are byte-identical at SHA-256
`d528d9fb649130cf46f849f6fb2542d679416c68730b892a810e519b8fd25603`.
All fits converge and the deterministic positive row-stochastic construction
creates zero new cube-boundary pixels.

The representation does not improve held-scene colour transfer. Mean and
median gains over the BO3 curve-plus-matrix control are `-0.45%` and `-1.20%`;
the candidate wins 17/40 scenes. Every family has a negative median gain. P95
improves slightly to `0.975x`, but worst error grows to `1.103x`.

No knots, matrices, capacity, family router, or thresholds may be added as a
rescue, and the 12 confirmation pairs remain pixel-unread. The result supports
the earlier diagnosis that a small smooth conditional average is not enough
to recover the salient display-chain transform.
