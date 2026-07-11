# FilmCase U4.3 Diagnostic Foundation

`src/filmcase/diagnostics.py` adds a diagnostic-only measure for new,
high-frequency chroma islands in already saturated source regions. It reports
candidate pixel coverage, connected-island counts and largest island size for
full-resolution review. It never returns a safety pass/fail result.

On the red bicycle/ColorChecker ID 11 counterfactual:

| Variant | Candidate pixels | Candidate coverage | Small islands |
|---|---:|---:|---:|
| #56 source gamut | 4,517 | 0.314% | 464 |
| #56 chroma + margin 4 | 3,206 | 0.223% | 608 |

The challenger reduces flagged coverage by about 29%, matching the observed
reduction in dense red speckling, but its island count rises. It is therefore
not an automatic winner: the metric identifies where an auditor should inspect
at original resolution and must be reported alongside the blind severe verdict.

The companion CLI writes only ignored JSON reports. U4.3 remains in progress
until face/text/edge/texture and effects diagnostics are similarly wired into
the final U4 evaluation report.
