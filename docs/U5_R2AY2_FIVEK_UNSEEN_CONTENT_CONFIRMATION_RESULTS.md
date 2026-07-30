# U5.R2AY2 FiveK unseen-content confirmation result

The untouched rows 65-128 first pass two byte-identical source audits. All 64
source/target pairs are present, RGB16-decodable and dimension matched. There
are zero exact IDs, exact derived source/target hashes or raw-preview
`dHash <= 4` matches against rows 1-64. Camera metadata is unavailable for
this half, so it is an unseen-content test only.

The predictor is frozen before confirmation: the same 42 low-dimensional
statistics, five explicit bounded parameters, training-only standardization,
`Ridge(alpha=100)` and unchanged AO6 path. Two reports are byte-identical at
`b63dc5c422432708605d4109ef550a531bc475b57be68d55755db35355e3e678`.

The scientific mechanism transfers strongly:

| untouched-content metric | result |
|---|---:|
| neutral ridge mean improvement over global | 13.03% |
| neutral ridge wins | 60.94% |
| neutral ridge/global P95 ratio | 0.7428 |
| fixed-look ridge mean improvement over global | 13.81% |
| fixed-look ridge wins | 64.06% |
| fixed-look ridge/global P95 ratio | 0.7728 |
| per-pair oracle median improvement over identity | 35.81% |
| ridge/target median style-dose ratio | 1.0471 |

This answers an important part of the original ML question positively: a
small model can learn scene-conditioned, bounded, interpretable adjustment
parameters that generalize beyond the training contents, rather than merely
raising saturation or averaging all photographs into one look.

The formal decision is nevertheless **closed**. The frozen zero-new-boundary
gate fails: the worst method reaches 1.5648% and ridge reaches 0.4496%. The
automatic protocol therefore forbids visual review, threshold relaxation,
feature/alpha/capacity rescue and AO6 retuning.

Keep AY0-AY2 as positive mechanism evidence and a product-safety negative
result. A future separately preregistered operator family may use the lesson,
but rows 65-128 are now seen and cannot serve as its confirmation population.
Nothing here identifies film, a stock response, exposure, process or scanner
truth.
