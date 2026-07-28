# P156 Chunked Reference-Render Memory Evidence

## Result

The exact 128-row implementation passes every frozen P156 gate. It reduces
the median peak process-tree RSS of the 6 MP file path by `222,513,152 B`
(about 212.21 MiB), while preserving output, recipe, normalized report and
identity-fallback behavior exactly. P156 closes as
`frozen-memory-and-wall-gates-pass`.

The contract was frozen at
`dbb23519397bffcf8617758e6a0005c99b0b2b4b`; the implementation is
`240e6e6a81129c5bca298915d7caf9f013b74d15`; and the exact run binding is
`994c7dbf7744ca00a69d31005e766ed8d533e376`. The ignored raw report SHA-256
is `be2ca36f0374fd09ed43b6bc02f420d2fb08ab74cf7771fd49c24d79bbeee151`.
The durable decision is
`configs/reference_match_chunked_render_memory_decision_v1.json`.

## Measured matrix

The fixed order was baseline, candidate, candidate, baseline.

| Run | Revision | Peak process-tree RSS |
|---:|---|---:|
| 1 | baseline | 1,329,860,608 B |
| 2 | candidate | 1,106,051,072 B |
| 3 | candidate | 1,102,848,000 B |
| 4 | baseline | 1,324,064,768 B |

- baseline median peak: `1,326,962,688 B`;
- candidate median peak: `1,104,449,536 B`;
- reduction: `222,513,152 B`;
- candidate/baseline RSS ratio: `0.8323139346627959`;
- baseline/candidate median worker wall: `9.71432635 s` /
  `9.66415040 s`;
- candidate/baseline wall ratio: `0.9948348502808497`.

The frozen gates required at least 128 MiB reduction, RSS ratio no higher than
`0.88`, and wall ratio no higher than `1.10`. All pass without post-result
threshold changes.

## Exactness and boundary

All four fresh workers reproduced:

- output SHA-256
  `2fdb6f01c158c096f49d7e2108f61b6b706f0df873160ba17616068bf320c26d`;
- recipe SHA-256
  `70a357489958a32bf28f9e7c59a28dfe0bd6350b3cdca499cd75cfce00877bce`;
- normalized report SHA-256
  `ef0fa5a01c50808b837c7f2f1b7429526e06c1421884ea94018af76a3c6fabec`;
- default `identity-fallback`, zero staging residue and zero orphan workers.

Focused tests prove the chunked functions are float32-exact against their
former full-frame implementations for source and chroma recipes, including a
257-by-389 non-divisible height, an invalid final row and a maximum observed
chunk of 128 rows. All reference-match tests pass: `1339 passed, 5 skipped`.

Phase sampling explains the reduction: the baseline full-frame gamut phase
peaked around 1.32--1.35 GB, while the chunked candidate phase peaked around
417--420 MB. The remaining whole-process peak is about 1.10 GB and now lies in
reference fitting or later persistent-array stages. P156 therefore proves this
bounded render kernel only; it does not prove total high-resolution readiness.

This result changes no recipe schema, algorithm ID, guard policy, producer
contract, A1/A4/A5 decision, RAW/HDR/video support, FilmFX behavior, main
renderer, native/device parity or product admission.
