# U5.R2X0 Palette-Transfer Affine-Collapse Results

Date: 2026-07-27

Decision: **automatic global path closes as affine-only**

## Reproducibility

- software commit:
  `ec16c4b2bbbccc4f2ea686877030bb5533c74bfc`;
- config SHA-256:
  `1634bb5042939276aed93450aaa7f2a11f279fd099c9aea9c7b0e6d11eeb5793`;
- two formal reports are byte-identical at
  `13203413cf6606f326544a07bba0a6368b1fd50e8cbfb2164ea4a3cc15c7c763`;
- both stderr logs are empty;
- no image, external implementation or CPLEX process was accessed.

The official paper PDF is retained at SHA-256
`5d135d8cf58b0d04ff65f9317ef223d95de70aa626f0cf5e398d8526d882bc09`.

## Exact affine collapse

For a fixed source palette `U`, target palette `V` and valid transport `T`,
the paper's automatic global path evaluates:

```text
w(p) = (p - c) (X^T X)^-1 X^T + 1/n
p'   = w(p) T V
```

The second line is exactly:

```text
p' = p A + b
```

over all RGB inputs. On the frozen seven-colour palettes, valid transport and
10,000 query points:

| Check | Maximum error | Gate |
|---|---:|---:|
| Direct equation (5) versus collapsed affine | `9.99e-16` | `1e-12` |
| Weight sum versus one | `2.89e-15` | `1e-12` |
| Equation (3) source-colour reconstruction | `3.33e-16` | `1e-12` |
| Transport row/column sums | `0 / 0` | exact |

Palette extraction and the CPLEX transport optimization can choose `A` and
`b`; once chosen, they cannot make the automatic global pixel transform
nonlinear.

## Structural counterexamples

Both witnesses satisfy the paper's nonnegative transport and exact
row/column-sum constraints.

### Orientation

Using the standard RGB tetrahedron and a transport that swaps red and blue
gives:

```text
A = [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]]
b = [0, 0, 0]
det(A) = -1
```

The transport constraints do not guarantee positive orientation.

### RGB-cube range

Map the source tetrahedron with vertices at `.4 + .2 * RGB_tetrahedron` to
the standard RGB tetrahedron with identity transport. The exact operator is:

```text
p' = 5p - 2
```

The unit RGB cube maps from approximately `-2` to `3`. Negative generalized
barycentric coordinates therefore do not provide a global cube-range
guarantee.

These are existence counterexamples. They do not say every natural-image
result from the paper is folded or out of range.

## Project decision

The branch is `analytic_affine_only_close`.

The automatic global method is not a distinct Ultimate colour representation:

- it has no nonlinear operator capacity beyond a per-reference affine fit;
- its source/reference palettes are scene-content statistics, not identified
  stock evidence;
- it lacks the range and orientation guarantees already required of Ultimate
  explicit operators;
- its proprietary optimizer is unnecessary to establish these limitations.

No CPLEX reimplementation, image run, clipping/projection rescue or capacity
extension opens. The interactive local-region path is outside this theorem
and is not adjudicated here.

The result makes no film, stock, calibration or unpaired digital-to-film
operator claim.
