# U5.R2BK23 — FiveK content-cell appearance control results

BK23 closes with a formal `K=1/identity` outcome.

Two complete CPU runs are byte-identical at
`709372fa5cf8d9db068abc0e29b4893684f4be4f11e1c20a4706b0ec1808b561`.
The 64 development and 64 confirmation FiveK identities are disjoint. Fitting
uses only pooled marginal samples with destroyed target order; exact photograph
identity is used only for confirmation cell scoring.

The selected sliced-quantile candidate looks excellent in aggregate: pooled
cross-objective improvement is 92.89%, and the two candidate operators differ
by only `.01669` RMSE. That aggregate is misleading. On the observed
source/target photograph cells, the worst improvement is `-6.6918`, only
64.06% of cells clear the frozen 10% threshold, and the policy returns
identity. The exact same target multiset under the frozen 17-cell permutation
also returns identity, with 46.88% of cells passing.

Cube, positive-Jacobian, norm, inverse and replay gates all pass. The failure is
not an unstable or malformed operator: it is lack of a single shared
cross-content appearance transform.

This is research-only digital-retouch evidence. It does not identify film,
stock, emulsion, scanner or a physical operator. It demonstrates why pooled
unpaired matching and canonicalizer agreement cannot substitute for
externally observed per-content validation.
