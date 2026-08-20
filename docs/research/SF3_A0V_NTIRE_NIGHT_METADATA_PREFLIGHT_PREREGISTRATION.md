# SF3.A0V NTIRE night metadata preflight preregistration

Date: 2026-08-21

SF3.A0V is a metadata-only child of the committed SF3.A0U source lock. It is
not candidate 2 of the bounded final scientific cycle. Its sole question is
whether a fixed, hash-ranked subset of the synchronous capture rows actually
contains machine-readable capture and target-crop facts sufficient to justify
a later pixel registration preflight.

The exact 24 numeric IDs, archive central-directory identity, accepted
normalized field-name families, byte budgets, gates and stop rules are frozen
in `configs/sf3_a0v_ntire_night_metadata_preflight_v1.json` before any member
payload is read. Field discovery is not allowed: each selected JSON object must
independently satisfy every predeclared family. Numeric values must be finite;
crop-associated numeric values must be bounded in magnitude. A missing row,
unknown schema or failed family closes the leaf without replacement.

The audit may read the exact central directory and only the selected JSON ZIP
members by bounded HTTP Range requests. It may not read a RAW PNG, Sony JPEG,
or any decoded pixel. It may not fit an operator, render, score quality, change
the `1/3` candidate counter, or make a product claim. A pass opens only another
separately frozen pixel/registration preflight.
