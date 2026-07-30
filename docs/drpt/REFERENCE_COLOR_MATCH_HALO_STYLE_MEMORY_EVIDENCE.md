# P158 Halo-Aware Safe-Lab Style Memory Evidence

## Result

P158 passes all frozen exactness, memory and wall-time gates. Calling the
authoritative safe-Lab operator over 128-row cores expanded by five vertical
source rows reduces median peak process-tree RSS by `465,741,824 B` (about
444.17 MiB). Output, recipe, normalized report and fallback behavior remain
exact.

Contract, implementation and run binding are respectively `4dfe468`, `4741507`
and `4948c7e`. The ignored formal report SHA-256 is
`0eb6d27e2f15a40ed2e47a7f1282ea927319dcc94a0576f74e92ae8bbd731b05`;
the durable decision is
`configs/reference_match_halo_style_memory_decision_v1.json`.

## Measured matrix

| Run | Revision | Peak process-tree RSS | Worker wall |
|---:|---|---:|---:|
| 1 | baseline | 1,017,327,616 B | 9.2905085 s |
| 2 | candidate | 531,513,344 B | 8.5230345 s |
| 3 | candidate | 536,408,064 B | 8.5198528 s |
| 4 | baseline | 982,077,440 B | 8.8606430 s |

- baseline/candidate median peak:
  `999,702,528 B` / `533,960,704 B`;
- reduction: `465,741,824 B`;
- RSS ratio: `0.5341195896225642`;
- baseline/candidate median worker wall:
  `9.07557575 s` / `8.52144365 s`;
- wall ratio: `0.9389424852737781`.

The frozen 256 MiB / `0.70` RSS / `1.10` wall gates all pass without
post-result changes.

## Exactness and attribution

Focused tests cover both working spaces, image edges, non-divisible heights,
zero and nonzero luma-detail, and a hard maximum of 138 expanded rows. The
halo-row output is float32 bit-exact to full-frame safe-Lab. The full
reference-match suite passes `1348 passed, 5 skipped`.

All four real file workers reproduce output `2fdb6f01...20c26d`, recipe
`70a35748...77bce`, normalized report `ef0fa5a0...fabec`, default identity
fallback and zero cleanup residue.

At 5 ms worker-phase resolution, style application falls from about
0.985--1.017 GB to 0.345--0.347 GB. The next candidate phase is the already
row-bounded gamut step around 0.420--0.423 GB; the frozen whole-process 20 ms
monitor peaks around 0.534 GB. These different samplers are not substituted
for one another.

## Boundary

This is exact local Windows/Python evidence for this fixed safe-Lab operator,
not a generic image-tiling claim. It changes no schema, algorithm identity,
policy, producer compatibility, main renderer, native/device path, media rail,
A1/A4/A5, visual result or product admission. Further memory work must first
attribute the remaining whole-process load/encode boundary rather than
re-optimizing the now-subdominant style stage.
