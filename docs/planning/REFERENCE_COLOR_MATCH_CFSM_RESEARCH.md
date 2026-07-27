# Canonical Factorized Safe Match Research Contract

## Purpose

CFSM is the explicit-operator challenger for neuro-film's one-reference to
N-source matching mode. It is isolated under `src/color_match/research/` and
cannot enter default delivery until A1 reference identifiability, A4
photographic review and A5 batch consistency all pass.

## Executable v0 factorization

```text
reference pixels
  + image-independent canonical RGB prior
  -> orientation-preserving Gaussian transport estimate
  -> cube-face-pinned residual field
  -> maximum safe strength by deterministic bisection
  -> constrained tetrahedral 17^3 LUT
  -> one canonical candidate ID
  -> identical fixed operator for every source in the batch
```

The cube-face envelope is zero at every domain boundary. Projection additionally
requires bounded output, residual amplitude, first/second differences,
neutral-axis error and positive tetrahedron Jacobians. No clipping, direct
neural RGB generation, spatial resampling or per-source render fitting is
allowed.

The candidate payload freezes algorithm ID, projection policy, LUT, diagnostics
and constraint report into a typed canonical SHA-256 identity. JSON replay
re-audits the LUT and rejects payload, report or identity tampering.

## Frozen decision matrix

| Candidate | Improved | Median | Worst | New boundary | Photo | Context | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| safe-Lab v1 | 5/30 | -92.19% | -344.87% | 21.86% | 2/6 | 0/6 | rejected |
| CFSM fixed cube 0.04 | 19/30 | +3.851% | -9.649% | 0% | 6/6 | 6/6 | rejected |
| CFSM fixed cube 0.08 | 18/30 | +4.397% | -27.907% | 0% | 6/6 | 6/6 | rejected |
| CFSM fixed cube 0.12 | 15/30 | +0.353% | -27.907% | 0% | 6/6 | 6/6 | rejected |
| CFSM uploaded-source batch | 16/30 | +0.571% | -11.524% | 0% | 6/6 | 6/6 | rejected |

The frozen promotion floor is at least 75% improved rows, +10% median,
no worse than -10% tail, at most 5% new boundary pixels, all photographic and
context probes passing, followed by blind aesthetic review. Automated metrics
can only reject or request visual review.

## Interpretation

CFSM-v0 proves that a fixed projected LUT can recover much of the cross-content
signal while eliminating the old context drift and boundary failures. It does
not prove that a uniform cube is a photographic canonical prior. The remaining
A1 gap is prior/identification error, not a reason to increase style strength.

The source-batch experiment is also negative evidence: pooling the user's N
images as the canonical prior absorbs their content distribution and performs
worse. Batch images may later inform bounded nuisance confidence, but cannot
define the target style prior directly.

## P14A analytic-prior decision

Three versioned, data-free RGB priors were frozen before execution:
low-key/neutral, mid-key/neutral and wide-chroma. Operator slices are disjoint:
24 development operators, 12 validation operators and 24 unexecuted confirmation
operators across matrix-only, tone-only and combined families.

Uniform cube remains the winner:

| Split | Uniform | Best analytic | Decision |
|---|---:|---:|---|
| development | 43/48, median +9.70% | 25/48, median +0.47% | no selection |
| validation | 19/24, median +6.92% | 13/24, median +1.02% | confirmation closed |

The validation report repeats byte-exactly at ID `d08ff8b...53fe2`, SHA-256
`816ba59e...f6549`. No confirmation or real-matrix run is permitted under the
frozen contract. This closes hand-shaped first/second-moment priors, not learned
canonicalization.

## P14B direction

The next eligible head is:

```text
reference -> independently learned canonicalizer -> bounded explicit LUT
source_i  -> independently learned nuisance canonicalizer
shared reference intent + bounded source descriptor
         -> projected fixed/global operator plus optional low-amplitude grid
```

Training or importing that canonicalizer requires a stable main-task W1
evidence bundle, rights-cleared data, leakage controls and an explicit
teacher/student contract. CanonCGT/StatLUT/ColorFM may act as teachers or
comparators; production output must remain an audited explicit operator.

RAW/HDR decoding remains outside this node. A future D-PCT bridge must first
provide a versioned common MatchView with explicit scene/display and luminance
semantics.
