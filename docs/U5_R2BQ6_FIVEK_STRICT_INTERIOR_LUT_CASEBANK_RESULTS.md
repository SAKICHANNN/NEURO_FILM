# U5.R2BQ6 strict-interior LUT case bank

Two formal runs are byte-identical at SHA-256
`04e410db3ce94870b027e1e13b19f2342c1fccc6d016890e780eded2676d441d`.
Both 381-case coefficient banks are also exact across runs. Confirmation pixels
remain unread.

The unchanged BJ2 4x4x4 residual LUT is the first BQ case representation to
pass both capacity and style. Self fits improve 49.78%/52.26% over identity
while retaining 78.19%/79.41% of target style. On five held-out camera groups,
the case Oracle improves 30.06%/34.25% over the fit-group global LUT and keeps
76.45%/82.65% style. Every held row has a style-eligible case; constraining the
Oracle to those cases costs only 2.62%/2.40% error. The construction introduces
zero new boundary or out-of-cube samples.

The active bottleneck is now source-only inference, not explicit-operator
capacity. BQ7 may compare a low-capacity factorized LUT predictor with the
already-frozen global and hard-retrieval baselines under camera-group holdout
and OOD fallback. This is paired FiveK digital-retouch control evidence, not a
film or stock result.
