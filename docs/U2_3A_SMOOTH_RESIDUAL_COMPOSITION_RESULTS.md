# U2.3A Smooth Residual Composition Results

**Decision:** identity composition passes; frozen nontrivial witness closes  
**Date:** 2026-07-24  
**Formal report:** ignored `outputs/u2_3a_smooth_residual_composition/formal/report.json`  
**Report SHA-256:** `9227e8a71cf2c1d86a82b24695211dffbe43e1261eb14037c1d299a5e50de0f4`

## Result

The new fail-closed wrapper composes the U2.2B sensitometry-print base with a
validated tetrahedral LUT. The 17-cube identity LUT preserves the base exactly
(`0.0` maximum error), and the vectorized interpolation matches the independent
scalar reference exactly. This closes the missing identity-composition
interface without changing the renderer or a profile.

The single frozen nontrivial cyclic witness is rejected before composition:

| Metric | Result | Frozen gate | Decision |
|---|---:|---:|---|
| Output range | `[0, 1]` | `[0, 1]` | pass |
| Residual amplitude | `0.0250000` | `<= 0.03` | pass |
| First-axis residual step | `0.0058594` | `<= 0.004` | **fail** |
| Second-axis difference | `0.00078125` | `<= 0.0005` | **fail** |
| Neutral-axis error | `0.0` | `<= 1e-12` | pass |
| Tetrahedral determinant | `0.9859..1.0066` | `>= 0.7` | pass |

The constraint failure is not an RGB-cube escape, foldover, replay defect or
interpolation error. It means this exact amplitude/function is less smooth on
the fixed grid than the preregistered residual budget permits. The candidate
is not weakened or retuned after inspection.

## Decision and propagation

- retain `SensitometryResidualLUTOperator` as a generic fail-closed research
  composition boundary and retain exact identity behavior;
- reject the nontrivial cyclic witness and do not open U2.3B from it;
- keep U2.2B as the current clean-room nonlinear algorithm witness and evaluate
  that base on real images only under a new frozen visual/artifact contract;
- keep U5.R2A as the standalone proof that accepted nontrivial tetrahedral LUTs
  are deterministic, while preserving its warning that numerical regularity
  is not semantic safety;
- do not fit film pixels, alter current data stops, integrate production, or
  claim a stock response.

## Verification

- 11 focused/adjacent tests pass;
- two formal audits have identical pre-provenance report and array hashes;
- all 803 local CPU tests pass;
- source preservation, invalid-input rejection, identity composition and
  independent tetrahedral-reference checks pass.

## Claim ceiling

Exact identity tetrahedral composition and a fail-closed bounded-residual
wrapper. There is no retained nontrivial U2.3A residual, visual-safety result,
identified digital-to-film operator, named-stock response, calibration or
production integration.
