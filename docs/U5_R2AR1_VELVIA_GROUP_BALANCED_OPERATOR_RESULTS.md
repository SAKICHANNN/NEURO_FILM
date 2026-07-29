# U5.R2AR1 — Velvia proxy group-balanced operator results

Status: **closed by the frozen structural gate**.

The experiment compared the existing pooled 71-row bounded one-matrix fit with
one shared fit that gives the 24-row chart and 47-row stable-pigment palette
equal total least-squares weight. It does not fit or route separate experts.
The same fixed comparison was repeated in five deterministic cross-fit folds.
All rows are reused development display proxies, not untouched or RAW pairs.

Two formal runs are byte-identical at SHA-256
`bc88455d576390bbcfbcd1b2834c4ba29759b6bc9dcc4b10fa6cceed3aeddf2`.

| Measure | pooled | group-balanced | relative change |
|---|---:|---:|---:|
| full worst-domain RGB RMSE | 0.041212 | 0.037422 | 9.20% better |
| full combined RGB RMSE | 0.029410 | 0.030095 | 2.33% worse |
| cross-fit worst-domain RGB RMSE | 0.047764 | 0.045771 | 4.17% better |
| cross-fit combined RGB RMSE | 0.033157 | 0.034392 | 3.72% worse |

The candidate wins the per-fold worst-domain comparison in four of five folds.
It stays inside the RGB cube and its maximum sampled Jacobian norm is 1.5371.
However, its minimum sampled Jacobian determinant is only `0.001516`, below
the frozen `0.01` gate. A read-only diagnostic places the pooled AO5 control
at `0.002874` under the same newer grid, so equal weighting improves group
fairness but does not repair the near-degenerate volume response of this
one-matrix family.

Automatic promotion therefore fails and photographic rendering is forbidden.
No determinant threshold change, strength shrinkage, weight sweep, fold
change, extra capacity or router is permitted. AO6 `t15/c35` remains the
development colour champion.

Claim ceiling: negative, post-exploratory display-proxy development evidence;
not film-stock response, calibration, product evidence or untouched
confirmation.
