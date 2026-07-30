# P162 — maximum 64-source batch scale evidence

## Result

Two fresh maximum-count transactions each process one reference and 64 ordered
one-megapixel sources. Peak process-tree RSS is `166,305,792 B` and
`166,797,312 B`; repeat ratio is `1.002956`. Worker wall time is `69.4451 s`
and `70.8665 s`.

All 64 source identities and all 64 output identities retain exact order and
reproduce across both runs. Recipe and normalized report are exact, all 128
render decisions are identity fallback, and staging/worker/worktree cleanup is
complete.

The ordered source-list and output-list canonical SHA-256 identities are
`9329c9b0...546b5` and `74bbb3ae...0f34`. The frozen config, runner and ignored
report hashes are `4d280c5b...8e850`, `e2580393...f3d30` and
`ace10e68...6fac0`.

## Interpretation

This closes the concrete product-count question: the real file transaction can
reach its strict 64-source ceiling without retaining prior encoded pixel
objects or reordering durable results. It complements P160/P161 rather than
replacing their 24 MP evidence.

## Boundary

Every source here is 1 MP. The result does not prove 24MP-by-64, arbitrary
sizes, RAW/HDR/video, native/device execution, non-identity quality or product
readiness.
