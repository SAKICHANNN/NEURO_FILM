# U5.R2BJ2 — Strict-interior adaptive LUT basis

BJ2 is closed. Two 191-image reports are byte-identical at
`7f46c3cd...d5721`, stable evidence `98dc2a58...51ffc`.

The strict-interior construction succeeds at its safety objective:

- zero out-of-cube pixels;
- zero new epsilon-boundary pixels;
- no clipping or post-operator gamut scaling.

Source-adaptive mean improvement over the global LUT is 12.04% on AY0, 8.87%
on AY3 and 15.60% on AY6. Win, worst, style and Oracle gates pass everywhere.
AY3 P95 is nevertheless `1.003719x` global, violating the frozen `<=1.0`
tail gate. The 0.3719% miss is not rounded away.

This closes the BJ0--BJ2 adaptive neutral-base LUT family. It demonstrates
useful bounded explicit-parameter prediction and a successful analytical
safety construction, but not sufficiently uniform fresh-tail generalization
or product value.

Claim ceiling: development-only digital-retouch mechanism evidence; no film,
stock, calibration, preference or product promotion.
