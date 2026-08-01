# U5.R2BQ0S2 FiveK case-bank normalization result

BQ0S2 is closed and BQ1 v1 did not open. The complete 512-pair run retained
509 aligned pairs across 28 camera models and produced a target-blind
381/128 whole-camera development/confirmation split. Exact leakage is zero,
all alignment/support gates pass, and no temporary files remain.

The frozen `dHash <= 4` gate nevertheless found one candidate against the
prior 255-pair pool and one across the new development/confirmation split.
Therefore the formal report fails automatically at SHA-256
`f744cac82d604425736d4d8c2a0b9b1a6e91b28f2d0730ce786f933a54ab5885`.
A second full run was not executed because repeatability cannot repair an
automatic eligibility failure.

Direct source-preview review diagnoses both candidates as unrelated content:
a sunset beach versus a dark wooden door, and a contrail sky versus asphalt
pavement. This is post-result diagnostic evidence that the single 64-bit
dHash gate has low-information false positives; it does not override the
frozen result, authorize row substitution, or permit BQ1 operator fitting.

A new version may preregister a source-only multi-evidence duplicate decision
before re-splitting. Confirmation targets and all operator outcomes remain
uninspected. This remains a paired digital-retouch architecture control, not
film, stock, calibration, preference, or product evidence.
