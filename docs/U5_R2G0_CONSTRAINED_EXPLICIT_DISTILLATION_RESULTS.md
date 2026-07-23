# U5.R2G0 constrained explicit distillation results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2G0`  
Decision: **closed — no automatic survivor; bounded style evidence retained**

## Integrity

- config SHA-256:
  `8767a5762af8aa6d2ab4760228a369769bc294c51530374283c39b287adce9ce`
- software commit:
  `ecd44034ae31a2d503e6805bf5fdc6eac0482b93`
- two-pass manifest SHA-256:
  `2dc6fd9082f139f3710577051c6520bc7b46b53556f2f4fdb892d1557b52e48e`
- automatic report SHA-256:
  `70db7f59a9df7dee1caefeed7e9da1c563d4471f7f42d04088124fc1970545b9`
- each pass contains 81 independently fitted explicit operators and 81 PNG
  renders; both stderr logs are empty
- all 768 repository tests passed before the formal run

## Automatic result

No reference policy passes every gate, so the visual shortlist is empty and no
promotion review is allowed.

The result is nevertheless materially different from F2 uniform contraction:

- style passes: 9/9 references;
- non-basic residual passes: 6/9;
- reference sensitivity: median pairwise Delta E76 `10.0674`, pass;
- median style advantage over F2 cap100: `+4.7740`, pass;
- median style Delta E76: `11.0444`;
- median non-basic residual: `5.8490`;
- median synthetic target RMSE: `0.0789`.

The universal failure is output-boundary safety:

- clipping passes: 0/9;
- worst candidate new hard clipping: `16.3178%`;
- structure passes: 4/9;
- exact raw-range passes: 6/9.

All continuous operators have positive Jacobians; the minimum baked
tetrahedral determinant is `2.9171e-5`. Seven of 81 records fail the exact
structure/range boundary only because a convex row sum produces a floating
maximum of `1.0000000000000002`. More importantly, RGB8 rounding maps many
near-boundary fitted outputs to exact 0 or 255, causing the large empirical
new-clipping rates. These facts do not waive the frozen failure.

## Interpretation

Constrained refitting preserves far more style than uniform identity
contraction. This is positive evidence for the architecture in which ML
predicts a transform and a deterministic explicit renderer enforces the final
image contract. It is not evidence of film-stock learning: the immutable
teacher is a generic FiveK/Lightroom grading model and no film pixel was fitted.

G0 itself closes under its preregistered gates. A distinct, data-independent
output-quantization safety leaf may test one fixed interior-headroom map over
the immutable G0 operators. This is justified by the observed boundary
mechanism and adds no capacity, fit, image feature, or selected parameter. It
must be frozen before any headroom render and may not refit G0.

## Claim ceiling

`generic reference-conditioned external-ML transform approximated by a bounded
explicit operator on a synthetic colour grid`.

No stock, calibration, latent-mode, preference, severe-safety, production, or
authenticity claim opens. Ultimate remains active.
