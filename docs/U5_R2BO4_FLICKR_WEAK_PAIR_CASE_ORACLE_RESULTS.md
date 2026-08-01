# U5.R2BO4 weak-pair case-bank Oracle

The same-capacity per-scene bank has real descriptive variation, but not enough
product value to justify a retriever. Two runs are byte-identical at
`ba33e56b...5c0e5a`; confirmation remains unread.

The Oracle lowers mean RMSE by 13.08% and wins 85% of development scenes, but
misses three frozen gates: median gain over family-global is **8.76% < 10%**,
median gain over train-selected medoid is **6.73% < 10%**, and p95 error ratio
is **0.943 > 0.90**. Thresholds and operators are unchanged.

Therefore no source-only similarity model, Top-1 selector or router is trained.
The 52 registered pairs remain useful noncommercial, one-author weak-pair
evidence for a future mechanism-distinct explicit operator experiment.
