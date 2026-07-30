# U5.R2AY0 FiveK neutral-base parameter result

The frozen pilot fits only five bounded explicit photographic parameters per
retained RAW/Expert-C pair: contrast, logit exposure shift, red and blue
white-balance shifts, and saturation. A low-dimensional source descriptor
predicts those parameters under complete camera-model GroupKFold. It never
predicts or generates final RGB.

Two 64-pair runs are byte-identical at
`5fb8e8e74312fd2b71f5ce50f43893225fcccf61f48f5b59ff9a50c6a169af5d`.
The stable evidence identity is
`26353480f55cdd88c8b21f3d4200269693d0158013f7a679bd92bf8c80f4e3c4`.

| held-out result | value |
|---|---:|
| per-pair explicit-oracle median improvement over identity | 34.34% |
| ridge mean improvement over outer-train global median | 13.12% |
| ridge wins over outer-train global median | 75.00% |
| ridge/global P95 RMSE ratio | 0.8179 |
| bootstrap lower bound for mean improvement | 0.00690 |
| fit success | 100% |
| maximum new boundary fraction | 0 |

All preregistered gates pass. This is positive evidence for a
content-conditioned but explicit and bounded neutral photographic base. It is
not evidence that FiveK contains a film look, and Expert C is not film or stock
ground truth.

The only opened child keeps the AO6 `t15/c35` look completely fixed. It will
compare direct, global-control and held-out-ridge neutral bases after that same
look against the same look applied to the paired neutral target. This asks
whether the neutral-base advantage survives a strong deterministic style
operator; it does not fit AO6 or upgrade its Look Approximation claim.
