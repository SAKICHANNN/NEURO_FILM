# U5.R2BO5 paired texture identifiability result

BO5 is closed. Two executions are byte-identical at SHA-256
`959b805624f5676a53e9649a6d876c02d9c16349ec05150e2bb137412456f174`.
The audit attempted all 40 BO3 development pairs, analyzed 31 with the frozen
flat-patch support, and did not load the 12 sealed confirmation pairs.

The result is not a stable cross-family texture signature. Median film/digital
log-luma high-pass energy is `0.8715`; the three family medians are `0.7631`,
`0.7296`, and `4.1384`, so the direction reverses by capture family. The pooled
film-minus-digital radial PSD has zero positive bins. Correctly aligned
predictive residual is effectively identical to the four-pixel shift control
(`0.99934` RMS ratio). Edge and JPEG-block controls themselves pass, but they
do not rescue identifiability.

No stochastic generator, texture profile, router, fitting, or confirmation run
opens. The evidence is consistent with camera/scanner/sharpening/resampling
nuisance dominating these web derivatives. It must not be called film grain,
emulsion structure, or a stock response.
