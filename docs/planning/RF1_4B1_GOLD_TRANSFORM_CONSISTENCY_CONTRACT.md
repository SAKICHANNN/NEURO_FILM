# RF1.4B1 Gold display-transform consistency contract

Date: 2026-07-16

Status: frozen before colour fitting

## Question

Does a nontrivial, bounded, explicit colour operator learned from five Gold100
archive rolls predict the sixth roll's aligned display proxy better than
identity, per-channel affine colour, and single-wrong-roll operators?

The target is BlueNeg's pseudo-ground-truth print/scan display chain. Even a
pass is not isolated emulsion response or calibrated Kodak Gold authenticity.

## Split and sampling

- six leave-one-physical-roll-out folds over the 47 exact pairs;
- official bbox crops only; no estimated registration or resize;
- deterministic grid capped at 4,096 paired pixels per frame;
- frames are balanced within rolls and rolls within each training fold;
- no frame-random split, target leakage or per-test-image fitting.

All 47 source and proxy PNGs are unprofiled 8-bit RGB. Operator fitting is
therefore frozen in an explicit `unprofiled_srgb_encoded_8bit_compatibility`
domain. It must not be described as linear light, scene-referred RGB or
physical negative density. CIELAB metrics decode those values as sRGB only for
relative candidate comparison.

## Frozen operator ladder

1. identity;
2. bounded independent RGB affine gain/bias (simple cast/contrast control);
3. bounded ridge 3x3 affine plus bias;
4. 17-knot monotone separable LUT followed by bounded ridge 3x3 mixing.

The fourth operator is the primary nontrivial challenger. LUT outputs stay in
`[0,1]`, knots remain monotone, matrix coefficients are bounded, and the
mixing matrix is forced to retain a positive determinant by deterministic
identity shrinkage if needed. All fit statistics and weights come from the
training rolls only. The full-resolution renderer is global and deterministic. No spatial resampling,
local mask or final-RGB neural generator is allowed.

## Evaluation and controls

Primary score is physical-roll-balanced mean CIELAB Delta E76 to the aligned
proxy. RGB MAE, Delta E p95, clipping and rendered distance from identity are
reported separately. Metrics compare candidates, not calibration.

For every held-out roll, fit the same challenger separately on each available
single training roll. Their median held-out error is the wrong-roll/process/
print-scanner control. Improvement confidence uses a 10,000-resample physical-
roll cluster bootstrap fixed before results.

## Promotion gates

The SepLUT challenger must:

- reduce primary error by at least 5 percent versus per-channel affine;
- improve at least five of six held-out rolls;
- have cluster-bootstrap lower improvement bound above zero;
- beat the median wrong-roll operator on at least four rolls;
- render a median Delta E76 change from identity of at least 3;
- add no more than 1 percentage point of clipping;
- produce finite outputs and monotone LUTs.

Only after every metric gate passes are full-resolution identity/simple/SepLUT/
target contact sheets rendered for the severe-artifact veto and style audit.
Metric pass plus visual fail is rejection.

## Branch boundary

- simple affine ties/wins: there is no demonstrated nontrivial style value;
- wrong-roll ties/wins: the mapping is nuisance-specific, not transferable;
- confidence interval crosses zero: run only the smallest ambiguity diagnostic;
- all gates pass: open an RF2.S Gold *archive-display* explicit-operator
  candidate, never a calibrated or isolated-stock claim.
