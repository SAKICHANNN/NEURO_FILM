# U5.R2AN2 Two-Stage Sparse-Rejection Results

Date: 2026-07-29

Two 92,893-byte reports are byte-identical at
`d5490ff406b7c26c0dcf56ab5957e4093b6e95cfb0892a6e62d880399ae0bcdb`.
The fixed label-blind policy fails 5 of 76 checks and closes.

The mechanism result is unusually clear. In every contaminated witness the
soft-L1 residual ranking rejects 26 of 27 true outliers (96.30% recall) with
72.22% precision while retaining all frozen patch, illuminant and exposure
support. Therefore the failure is not inability to locate corrupt rows.

The hard-delete plus linear-refit step harms recovery. For the critical
`cyan_shadow_warm_highlight_like` witness, confirmation RMSE rises from
`0.000437` at stage one to `0.000504`; gain over the all-row linear fit is
only 55.44%, below the unchanged 65% gate. It also loses 15.55% to stage one.
On clean-noise data, warm-highlight and cross-bias lose 14.43% and 19.18% to
stage one.

All structure and retained-support gates pass. The result points to estimator
variance or nonlinear parameter coupling after hard deletion, not clipping,
folding, group collapse or outlier-detection failure. The rejection fraction,
refit loss and gates are not retuned, and no proxy or photograph is opened.

The next distinct low-cost challenger may use a separately frozen
redescending soft-weight influence function at the existing loss scale. It is
still synthetic-only and must pass before any real proxy use.
