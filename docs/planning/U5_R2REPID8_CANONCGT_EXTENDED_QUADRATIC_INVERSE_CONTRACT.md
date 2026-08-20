# U5.R2REPID8 — CanonCGT extended-domain quadratic inverse contract

Date: 2026-08-20

REPID7 established a strong same-image pseudo-neutral signal but failed its
all-reference contract because one canonical output slightly escaped the RGB
cube and one reference exceeded the held-out error gate. REPID8 does not clip,
project or relax either result. It changes the explicit operator
representation.

The unchanged pinned CanonCGT canonicalizer reads one already-consumed A0
reference. Its finite output is encoded with a fixed signed `asinh` coordinate
that remains defined outside `[0,1]`. Ten fixed quadratic basis terms map to
three target logits, giving exactly 30 coefficients; sigmoid execution makes
the rendered output intrinsically interior. Fit and held-out pixels use the
same deterministic disjoint partition as REPID7. Build reads no application
source.

Before Apply, all nine references must pass held-out median/p95 OKLab error,
fit/held gap, coefficient bound, extended-input support and a fixed 7-cubed
analytic-Jacobian grid over `[-.05,1.05]^3`. The minimum determinant and
singular value must remain positive and the maximum condition number bounded.
The new representation must also reduce median p95 error by at least 10%
against the frozen REPID7 logit-affine control and may not regress any
reference by more than `.01` absolute p95 error. Forward and reversed fresh
processes must produce byte-identical reports.

Any failure closes this exact basis, scale, ridge, split and grid without
projection, clipping, dose, reference exclusion or checkpoint rescue. A pass
would open only a separately frozen consumed-source application D0.

Execution amendment: the first formal invocation stopped before model or
reference access because the loader's direct parent was mistakenly set to the
REPID7 control rather than the F1 CanonCGT asset contract. The corrected config
binds F1 as `parent_contract` and REPID7 separately as
`matched_control_contract`; no scientific role, representation, parameter,
metric or gate changed, and formal execution restarts from zero.
