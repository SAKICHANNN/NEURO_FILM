# RF0.3 FSA/OWI grouping-gate results

**Decision:** `grouped_phase_c_allowed`; training remains forbidden until the
full bounded corpus passes border, content and nuisance-identifiability gates.

The first preregistered grouping failed. Creator plus nearby LOC identifier
formed sixteen known-creator sequence groups, but only eleven had at least
three records versus the frozen minimum of twelve. The largest group contained
123 records. The failure is retained in the final report.

The recovery was frozen before rerun and uses the official LOC FSA/OWI
assignment tables: extract an explicit state or territory from the catalogued
caption, combine it with creator, and retain the sequence group as fallback.
It recovered location for 272 of 457 known-creator records (59.52%) and formed:

- 43 known-creator location guard groups;
- 30 groups with at least three records;
- a largest group of 65 records, or 14.22% of the learning-eligible lane;
- seven creator groups with at least eight records;
- 101 unknown-creator records reserved for stress testing only.

The report and grouped manifest rerun byte-identically.

| Artifact | SHA-256 |
|---|---|
| grouped manifest | `ec7f6f685a1b4d965e5179138a0676aa38f0e57a1324b668bbaf7ed110e258b5` |
| grouping report | `4bbf331383e2f1c376e2a95dbd24d3fe29c46fd29ea219d3e5fc84645f0498e9` |

Creator-out is the primary generalization test. Location-group-out is a
secondary nuisance guard, and the failed sequence-only result is a sensitivity
analysis. None of these groups may be called a physical roll, exact LOT,
development batch or scanner session. Scanner-out validation requires an
independent real-film source.

The next ready leaf is the bounded 558-image Phase C download. It must preserve
the grouped manifest, remain below 1 GiB, compute payload/perceptual hashes and
create deterministic interior masks before any colour feature or expert is
fit.
