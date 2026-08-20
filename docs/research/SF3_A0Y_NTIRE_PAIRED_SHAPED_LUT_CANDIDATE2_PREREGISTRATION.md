# SF3.A0Y paired shaped-LUT candidate 2 preregistration

Date: 2026-08-21

SF3.A0Y is candidate 2 of the bounded final three-candidate cycle. It uses the
new information qualified by SF3.A0X: synchronized, independently captured,
pair-specific natural RAW/Sony pixels with fixed official geometry. It is not
an after-only reference inversion, metadata-only WB rescue, content router or
per-scene operator.

Prescore amendment, still before any SF3.A0Y member or pixel read: the
bounded-logit-affine control uses ridge `0.01`, then the largest identity-to-fit
dose found by 30 fixed binary-search iterations subject to determinant
`>=0.01`, minimum singular value `>=0.05`, and condition number `<=20`. The
cyclic-wrong-target shaped-LUT control maps every fit source to the next fit ID
in the listed role order, with wraparound. These rules are frozen with the
other controls and are not selected from calibration results.

All 56 IDs are fresh relative to SF3.A0V/A0X and split before acquisition into
32 fit, 12 calibration and 12 sealed-confirmation scenes. The cache contains
only 256x256 aligned source/target arrays under the repo-relative P-backed data
root; full ZIP members are streamed and never persisted. Fit targets may be
read only for the fit role. The shared shaped 9-cube LUT, its nested monotone
shaper, bounded logit-affine and identity controls, and an equal-architecture
cyclic-wrong-target control must all be serialized before any calibration
target is read. Sealed targets remain unread unless every calibration gate
passes.

The primary comparison is the shared safe LUT against the strongest
legitimate control per scene. It must pass rate, median, worst-tail, absolute
OKLab error, gradient, boundary and Jacobian gates, and separately demonstrate
pair identification against the cyclic-wrong-target LUT. Calibration scoring
increments the bounded counter from `1/3` to `2/3` regardless of outcome. No
same-cohort parameter, capacity, role, threshold or routing rescue is allowed.
