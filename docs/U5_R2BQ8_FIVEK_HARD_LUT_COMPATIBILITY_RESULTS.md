# U5.R2BQ8 hard LUT compatibility

Two runs are byte-identical at SHA-256
`59d11fc2450724bda66c531a819acb464e4d3c64a5b571113e90b780cf9b4d87`.
The source-only ranker selects one intact BQ6 LUT or the global OOD fallback;
it never blends LUTs and confirmation remains unread.

Hard selection prevents blandness: median target-style retention is
121.08%/125.47%, with zero new boundary or out-of-cube samples. The learned
ranking is non-random, improving 6.14%/12.40% over random and 5.23%/12.58% over
the shuffled ranker. It nevertheless applies the wrong or excessive look too
often: mean error is 27.14%/21.68% worse than global, win rate is only
26.67%/32.59%, and the aligned tail fails.

No larger ranker, feature, loss or threshold rescue opens. BQ9 will hold the
selected case identity fixed and ask whether any style-eligible point on the
global-to-case LUT path has evaluator-Oracle value. This separates wrong
direction from wrong magnitude without training on confirmation.
