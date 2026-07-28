# U5.R2AN1 Robust Paired-Recovery Results

Date: 2026-07-28

Two 59,555-byte reports are exact at `ec5309e...bb08`. The fixed soft-L1
policy fails one of 31 automatic checks and therefore does not advance.

Across three truth operators, dual patch-mean noise gives clean held-group
RMSE `0.000342`--`0.000372`. With 3% sparse corrupted correspondences,
soft-L1 gives RMSE `0.000328`--`0.000437`, maximum error at most `0.00265`,
and improves over the robust one-matrix control by `96.56%`--`97.33%`.

The sole failure is the preregistered 65% relative gain over linear fitting:
`cyan_shadow_warm_highlight_like` improves by `61.44%`. The other two
witnesses improve by `78.24%` and `83.49%`.

The fixed soft-L1 promotion closes without changing its threshold or scale.
Absolute recovery remains strong enough to justify a separately frozen
two-stage sparse-outlier rejection challenger. No real-film, Velvia,
calibration or product claim opens.
