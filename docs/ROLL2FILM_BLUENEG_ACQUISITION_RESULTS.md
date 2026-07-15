# Roll2Film BlueNeg bounded acquisition results

Date: 2026-07-15

Node: `ULT > U5.CT6`

The frozen BlueNeg pilot acquisition completed successfully.

| Check | Result |
|---|---:|
| Exact revision | `b038a1ae68f42067ff12b5e79ddbe62919b7af23` |
| Frozen files | 101 |
| Verified files | 101 |
| Frozen bytes | 118,929,719 |
| Verified bytes | 118,929,719 |
| LFS SHA-256 failures | 0 |
| Manifest-external lane files | 0 |
| Image payloads decoded | no |

The deterministic download report is
`outputs/roll2film/blueneg_v1/download_report.json`, SHA-256
`f20d83a4378ee5a4306ffa02ea04f7f2ead6c0c58c79aa36152e3bd199b934a7`.
It was produced at software commit
`14946319cb47588a02448006f3e3cb14ff40eec8` and reran byte-identically from
the local cache.

Source bytes are ignored under `data/raw/blueneg`. The downloader rejects path
traversal, paths outside the two frozen 8-bit lanes, acquisition/report hash
drift, byte-count drift, LFS hash drift and manifest-external lane files. A
corrupted in-scope local file is force-refetched and reverified.

This acquisition does not widen the scientific claim. Only four
`Kodak Gold 100-5` rolls form the matched core; one `Kodak Gold 100` roll is
exploratory. The full 955MB two-lane mirror and 290GB archive remain outside the
pilot.

The next gate is evaluator policy freeze before first pixel decode: fixed
support/query budgets, paired-blind support permutation, correct-roll and
same-film wrong-roll controls, pooled/shuffled controls, nuisance reporting,
bootstrap unit and pass/fail/ambiguity rules.
