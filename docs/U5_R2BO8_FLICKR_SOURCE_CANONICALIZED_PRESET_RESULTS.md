# U5.R2BO8 source-canonicalized preset result

BO8 is closed. Two reports are byte-identical at SHA-256
`c51678e411b2de3f8afae469226c23dd3627fbf4d5065bd50545aff25a434732`.
The source adapter reads only application-source pixels and geometry; target
and reference access are false. Execution is invertible, bounded, repeatable,
and creates zero new cube-boundary pixels.

It does not identify useful source-specific adjustment. Mean gain over the
uncanonicalized control is `-0.25%`; median gain is `0.28%`, win rate is 50%,
and one family median is negative. Correct adapters improve only `0.048%`
over deliberately wrong same-family adapters. Worst error grows to `1.147x`.

The adapter statistic, cap, shared operator and capacity are closed, and all
12 confirmation pairs stay unread. This result must not be described as
learned exposure, white balance, illumination, scanner state, or a successful
NFRM canonicalization stage.
