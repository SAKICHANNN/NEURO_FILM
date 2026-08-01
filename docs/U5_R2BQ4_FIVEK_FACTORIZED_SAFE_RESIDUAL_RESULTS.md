# U5.R2BQ4 factorized safe residual

BQ4 closes before confirmation, but it materially advances the algorithmic
evidence. Two formal runs are byte-identical at SHA-256
`5bf3cbc458eb3b25f27c4291b3066a32b27cf4ecf60be529dd592d8979d8bfa1`.
The model predicts only a normalized 14-parameter direction and a separately
bounded log magnitude from source pixels. A monotone triangular-logit operator
produces the candidate; one analytical source-to-candidate scale then keeps
each previously interior pixel strictly inside gamut without channel clipping.

Both development variants pass every accuracy, tail, control, OOD and safety
gate. Aligned/Filtered mean errors improve 11.48%/14.68% over the safe pooled
global, 9.51%/11.30% over its continuous strength Oracle, 5.96%/9.24% over
frozen nearest retrieval, and 6.88%/3.99% over direct BQ3. Group-bootstrap
lower bounds are strongly positive. New boundary fractions are exactly zero,
median image-mean residual scale is 1.0, and style magnitude is 1.25x/1.45x
the safe global rather than averaged toward identity.

The sole failed gate is absolute target-style retention: 45.58% for Aligned
Expert and 69.05% for Filtered, against the frozen 70% requirement. Filtered
is close but remains a formal failure; the threshold is not rounded or changed.
The confirmation population was not decoded or scored.

The next legal leaf is a development-only capacity diagnosis: measure the
style-retention ceiling of each fitted 14-parameter case operator and of the
error-minimizing evaluator Oracle. If those ceilings are also inadequate, the
next change must be a more expressive bounded explicit operator or sparse
residual bank, not a larger selector. FiveK remains digital-retouch control
evidence rather than film, stock, calibration, preference or product proof.
