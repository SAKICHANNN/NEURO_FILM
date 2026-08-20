# U5.R2REPID7 — CanonCGT self-canonical inverse contract

Date: 2026-08-20

REPID6 proved that a fixed synthetic atlas can expose a stable reference grade
signal, but the resulting Restyler LUT is unsafe and mostly basic when applied
as a shared operator. REPID7 changes the identifying observation rather than
repairing that LUT.

For each already-consumed A0 reference, the unchanged pinned CanonCGT
canonicalizer produces a same-image pseudo-neutral observation. A bounded
12-parameter logit-affine operator is fitted from the canonicalized reference
back to the observed reference on one deterministic pixel partition and scored
on the disjoint partition. Build reads no application source. The sigmoid
executor is intrinsically interior and one fitted operator can later be frozen
and replayed without source conditioning.

Before any application source read, all nine references must pass raw
canonical range, held-out median/p95 OKLab error, fit/held gap, determinant,
minimum singular value and condition-number gates. Forward and reversed
reference enumeration must produce byte-identical reports in fresh processes.
Any failure closes this exact inverse representation without dose, projection,
ridge, split, threshold or checkpoint rescue.

A pass opens only a separately frozen consumed-source application D0. It is not
itself image-quality evidence or product admission.
