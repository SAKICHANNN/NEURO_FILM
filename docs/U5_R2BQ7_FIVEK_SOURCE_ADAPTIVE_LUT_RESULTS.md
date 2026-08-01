# U5.R2BQ7 source-adaptive LUT

Two reports are byte-identical at SHA-256
`d23946e3cc158ebe8f55508720214dece592572b1733a50350ae6ee11a65f907`.
The predictor uses only fit-group source descriptors, rank-8 LUT coefficients,
grouped inner-CV Ridge and a fit-only OOD threshold. Confirmation is unread.

The learned signal is real but averaged. It improves 6.89%/8.98% over the
global LUT and 15.01%/18.31% over shuffled predictions, with positive grouped
bootstrap intervals, 5.93% fallback and zero new boundary/cube samples.
However, target-style retention is only 53.14%/57.91%, below the frozen 70%
floor. It beats a global strength Oracle in mean by 3.02%/4.19% but wins only
45.19% of individual rows.

No rank, alpha, descriptor, amplitude, OOD or threshold rescue is allowed and
automatic failure forbids visual review. BQ8 will keep one real BQ6 case LUT
intact and learn only source-to-case compatibility for hard Top-1 selection,
with global OOD fallback. This remains FiveK digital-retouch control evidence.
