# U5.R2BD1 Color Precision page-level identifiability result

## Decision

Close this source for stock learning and retain it only as an internal scanner
nuisance stress set.

The two exact runs are byte-identical at report SHA-256
`6a56d2849e71daab31cfa7e6fddefa52c4dbe13cbdebd31976690b5d7df541f7`.
They bind 191 unique full-photo PDF objects into 24 exposure-unassigned page
bags covering 12 stock/process variants and two scanners.

| Frozen view | Symmetric top-1 | Correct rank median | Same/wrong distance ratio | Permutation p | Gate |
|---|---:|---:|---:|---:|---|
| Raw | 20.83% | 4.5 | 0.8857 | .0386 | fail |
| Basic-normalized | 33.33% | 2.5 | 0.8092 | .0001 | fail |

The normalized view contains a detectable paired-distance residual, but it is
not reliable 12-way stock/process identification and misses the frozen top-1,
rank and distance-ratio gates. Scanner interpretation, exposure mixture and
unknown decoded PDF colour semantics remain material nuisance variables.

## Boundaries

- No embedded image was assigned to an advertised EV.
- Exact duplicate PDF objects were collapsed by object identity only.
- Poppler PNG output did not preserve a verifiable embedded ICC
  interpretation, so the evidence is same-pipeline display-proxy evidence.
- No learned representation, operator fitting, training, latent clustering or
  product profile extraction occurred.
- All Rights Reserved supplies no redistribution or training grant.

The authoritative decision is
`configs/u5_r2bd1_color_precision_page_identifiability_decision_v1.json`.
