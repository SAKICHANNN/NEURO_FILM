# U5.R2BM1 AO6 hard-case operator Oracle

BM1 passes its frozen automatic gate. Two independent full runs produce the
same report SHA-256, `ea903412c4887b143b192d4e868d8b8bc93f69d979a1a4c431d6af6a73e478e6`,
and stable evidence ID
`3d05baf021519940333d64921ebbc21c9a4c067967dfa22794190e558cb8a7b4`.

The test keeps operator and content spaces separate. Each development case is
only the density-domain safe-Lab mean/std used by the frozen AO6 t15/c35
renderer. The held query never contributes to that case identity. A fixed
17-cube output signature chooses one development medoid; the evaluator Oracle
then hard-selects one development case. A separate post-hoc control is allowed
to choose the best analytically in-gamut scalar strength along the medoid
residual. No content feature, router, dense mixture or new colour fit is used.

Across 17 held-camera sources, the hard case Oracle reduces median target RGB
RMSE by 28.85% versus the global medoid and by 19.03% versus the stronger
scalar-strength Oracle. Nine sources beat the strength control by at least 2%,
12 select a non-medoid case, and eight distinct case IDs are selected. Median
style retention is 0.9605, worst case is no worse than the medoid, and new
boundary fraction is zero. Exact self-context replay has maximum RGB RMSE
`4.42e-6`, validating the explicit context-rebinding evaluator.

This is a necessary Oracle gate, not retrieval success. It shows that this
deterministic control contains reusable case variation beyond a single
strength direction; it does not show that image content can predict the right
case. The eight selected IDs are not latent film modes, and AO6 remains a Look
Approximation rather than a film/stock response. BM2 may now compare frozen
colour-independent hard retrieval against the medoid, strength, shuffled and
BM1 Oracle controls. A retrieval failure will retain the Oracle descriptively
but close case routing.
