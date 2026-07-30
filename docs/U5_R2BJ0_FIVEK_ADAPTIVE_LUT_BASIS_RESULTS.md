# U5.R2BJ0 — Adaptive explicit LUT-basis development

BJ0 is closed. Two complete 191-image runs are byte-identical at report
SHA-256 `509ba9f2...e82fc` and stable evidence ID
`17d82a6d...86252`.

The method is materially better than its global-LUT control on every
development population:

| Population | Mean improvement | Win fraction | P95 ratio | P95 safety-limited |
|---|---:|---:|---:|---:|
| AY0 camera-group crossfit | 14.43% | 79.69% | 0.835 | 22.07% |
| AY3 seen fresh | 13.60% | 65.08% | 0.983 | 17.55% |
| AY6 seen confirmation | 15.24% | 68.75% | 0.841 | 20.52% |

Oracle capacity, mean, win, P95, worst, style and zero-new-boundary gates all
pass. The single failure is decisive: every population exceeds the frozen
10% P95 fraction of pixels whose LUT residual must be analytically reduced.
Visual review is therefore forbidden.

This shows that a source-only model can predict useful coefficients for a
small explicit LUT bank without generating RGB, but the fixed BJ0 residual
basis demands too much gamut intervention in its tail. Grid/rank/
regularization/predictor/gate tuning and confirmation acquisition are
forbidden. A future leaf may test a mathematically different,
intrinsically cube-preserving residual basis.

Claim ceiling: development-only digital-retouch mechanism evidence; not film,
stock, calibration, preference or product promotion.
