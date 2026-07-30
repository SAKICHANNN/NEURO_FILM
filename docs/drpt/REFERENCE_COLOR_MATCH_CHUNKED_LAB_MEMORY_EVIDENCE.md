# P157 Chunked Lab-Ingress Memory Evidence

## Result

P157 passes every frozen gate. Exact 128-row display-linear RGB-to-Lab
conversion, bounded gamut validation and reuse of the validated Lab reduce the
6 MP file-path median peak process-tree RSS by `166,363,136 B` (about
158.66 MiB). Output, recipe, normalized report and identity-fallback behavior
remain exact.

The contract is commit `0972628c07ca509e88558ec4cac55c80b6c2acb6`;
the functional candidate is
`6e1387b0f98a18ba7e89f9c68e47bc143205ac5b`; and the run binding is
`fdb5a46e37099275a36a52db19dba2502c0e42c8`. The ignored raw report has
SHA-256 `3838e43d977d99861ad8b2db229c14727f996b31a15f9094551086fc35a21160`.
The durable result is
`configs/reference_match_chunked_lab_memory_decision_v1.json`.

## Measured matrix

The fixed order was baseline, candidate, candidate, baseline.

| Run | Revision | Peak process-tree RSS | Worker wall |
|---:|---|---:|---:|
| 1 | baseline | 1,093,337,088 B | 10.8279180 s |
| 2 | candidate | 931,061,760 B | 8.7051019 s |
| 3 | candidate | 927,539,200 B | 8.6987255 s |
| 4 | baseline | 1,097,990,144 B | 9.3974098 s |

- baseline/candidate median peak:
  `1,095,663,616 B` / `929,300,480 B`;
- reduction: `166,363,136 B`;
- RSS ratio: `0.8481622155097647`;
- baseline/candidate median worker wall:
  `10.1126639 s` / `8.7019137 s`;
- wall ratio: `0.8604966788229699`.

The frozen gates were 64 MiB minimum reduction, RSS ratio at most `0.94` and
wall ratio at most `1.05`. All pass without threshold changes.

## Exactness and attribution

All four workers reproduce the same P154/P156 identities:

- output `2fdb6f01...20c26d`;
- recipe `70a35748...77bce`;
- normalized semantic report `ef0fa5a0...fabec`;
- default identity fallback, zero staging residue and zero orphan workers.

Both supported working spaces are bit-exact against full-frame conversion on a
257-by-389 non-divisible test. Fit and render each reuse their single validated
Lab result, and no conversion call observes more than 128 rows. The full
reference-match suite passes `1343 passed, 5 skipped`.

Independent 5 ms worker-phase sampling shows the intended mechanism:
reference-fit falls from about 1.09 GB to 0.372--0.374 GB. The highest remaining
candidate phase sample is the unmodified full-frame safe-Lab style stage at
about 0.993--1.010 GB. Phase samples have a different interval and scope than
the frozen 20 ms process-tree gate and are used only for attribution, not to
rewrite the measured gate.

## Boundary

P157 proves a local Windows/Python bounded Lab-ingress kernel. It does not
change safe-Lab style math, gamut compression, schemas, algorithm identity,
producer compatibility, A1/A4/A5, main rendering, native/device parity,
RAW/HDR/video support, visual quality or product admission. A later style
memory leaf must preserve the Gaussian luma-detail neighborhood exactly rather
than slicing it naively.
