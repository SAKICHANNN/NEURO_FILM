# U5.R2BQ5 projection-curve case bank

Two runs of the unchanged mature BN1 projection-curve fit are byte-identical at
SHA-256 `b53098b2205b656776963210379c62b5c844ec069586cbd0a64e180473a775a5`;
both coefficient banks are also byte-identical. Source-only fit eligibility
retains 358/381 rows and all 24 camera groups without changing the mature
minimum-system condition.

The representation has meaningful cross-case capacity. Group-held Oracle
error improves 24.46%/31.44% over a fit-group global operator, with median
style retention 70.22%/78.05%. A style-constrained Oracle costs only
3.89%/2.01% error.

The per-image fit nevertheless closes. After the frozen Jacobian safety dose,
self-fit style retention is only 27.59%/32.14%; median dose is 0.247/0.300 and
the worst dose is 0.094/0.107. Aligned style-eligible availability is also
89.92%, one held row below the 90% gate. Boundary, cube and sampled Jacobian
sign checks pass, but only because the dose removes most of the intended look.

No threshold or fit rescue is allowed and confirmation remained unread. The
next capacity baseline is the already-established BJ2 strict-interior 4x4x4
residual LUT, whose boundedness is intrinsic rather than obtained by collapsing
an unstable derivative. Only if that case bank passes may source prediction or
sparse retrieval reopen. This remains digital-retouch control evidence.
