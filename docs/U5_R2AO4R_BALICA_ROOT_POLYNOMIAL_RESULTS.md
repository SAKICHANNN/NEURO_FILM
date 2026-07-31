# U5.R2AO4R root-polynomial baseline result

The paper-supported third-degree root-polynomial correction model does not
transfer into a usable forward film-look operator on the frozen 71-pair
Velvia display-proxy pool. Two formal runs are byte-identical at
`fe127107...8d0f`.

Against the bounded one-matrix control, combined RGB RMSE changes from
`.03352` to `.09412` (a `-180.82%` gain) and mean Delta E76 changes from
`6.60` to `10.65` (`-61.32%`). Chart-to-palette and palette-to-chart transfer
regress by `131.12%` and `133.72%`. The candidate also reaches `13.33%`
held-fold raw out-of-cube rows and a minimum cube Jacobian determinant of
`-.11519`.

This is directionality evidence: a mature model that removes film/capture
colour bias need not generate a transferable film appearance in the reverse
direction. No clipping, bounded-residual rescue, photograph rendering or
parameter retuning opens. AO6 remains the development-only Look Approximation
incumbent; the next colour leaf requires new controlled information or a
genuinely different bounded mechanism.
