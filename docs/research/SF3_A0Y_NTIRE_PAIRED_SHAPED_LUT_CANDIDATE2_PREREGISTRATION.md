# SF3.A0Y paired shaped-LUT candidate 2 preregistration

Date: 2026-08-21

SF3.A0Y is candidate 2 of the bounded final three-candidate cycle. It uses the
new information qualified by SF3.A0X: synchronized, independently captured,
pair-specific natural RAW/Sony pixels with fixed official geometry. It is not
an after-only reference inversion, metadata-only WB rescue, content router or
per-scene operator.

Prescore amendment, still before any SF3.A0Y member or pixel read: the
bounded-logit-affine control uses ridge `0.01`, then the largest identity-to-fit
dose found by 30 fixed binary-search iterations subject to determinant
`>=0.01`, minimum singular value `>=0.05`, and condition number `<=20`. The
cyclic-wrong-target shaped-LUT control maps every fit source to the next fit ID
in the listed role order, with wraparound. These rules are frozen with the
other controls and are not selected from calibration results.

Storage-only execution amendment: the canonical P-backed exFAT volume reached
zero free bytes after four complete fit-cache rows. The interrupted attempt had
zero model fit/lock, calibration reads, sealed reads, or scientific scores. Its
partial cache was not retained. Formal execution therefore restarts from zero
through the repo-relative `.data_fallback` junction to the existing namespaced
`D:\_project_fallbacks\neuro_film_goal_019f4b76\data` root. No payload is placed
at the D drive root. Roles, pixels, candidate, controls, gates, and stop rules
remain unchanged; the fallback must be hash-verified back to P when durable P
space is restored.

Transport-only execution amendment: the default urllib path through the local
proxy advanced roughly 0.59 MB in five minutes, while `curl 8.21.0` returned
the same frozen 184,526-byte central directory in 1.67 seconds with the exact
expected SHA-256. Formal execution therefore uses an in-memory curl Range
reader that requires HTTP 206 and exact byte length; no full member is written
to disk. At amendment time only four fit cache rows existed and there were zero
model fits/locks, calibration or sealed reads, and scientific scores. All
scientific roles, models, controls, and gates remain unchanged.

Transport reliability correction: after reaching 27 complete fit-cache rows,
one later Range returned a non-206 response and stopped before model fitting.
The reader now permits four independent attempts with fresh in-memory buffers
and a fixed one-second delay. Every successful attempt still requires HTTP 206
and exact byte length; failed/partial bytes are discarded rather than joined.
At correction time model locks, calibration reads, sealed reads, and scores
remain zero.

All 56 IDs are fresh relative to SF3.A0V/A0X and split before acquisition into
32 fit, 12 calibration and 12 sealed-confirmation scenes. The cache contains
only 256x256 aligned source/target arrays under the repo-relative P-backed data
root; full ZIP members are streamed and never persisted. Fit targets may be
read only for the fit role. The shared shaped 9-cube LUT, its nested monotone
shaper, bounded logit-affine and identity controls, and an equal-architecture
cyclic-wrong-target control must all be serialized before any calibration
target is read. Sealed targets remain unread unless every calibration gate
passes.

The primary comparison is the shared safe LUT against the strongest
legitimate control per scene. It must pass rate, median, worst-tail, absolute
OKLab error, gradient, boundary and Jacobian gates, and separately demonstrate
pair identification against the cyclic-wrong-target LUT. Calibration scoring
increments the bounded counter from `1/3` to `2/3` regardless of outcome. No
same-cohort parameter, capacity, role, threshold or routing rescue is allowed.
