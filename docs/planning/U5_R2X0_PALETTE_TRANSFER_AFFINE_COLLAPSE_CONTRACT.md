# U5.R2X0 Palette-Transfer Affine-Collapse Contract

Date: 2026-07-27

Status: **preregistered implementation frozen; formal analytic run pending**

## Question

The 2025 *Palette-Based Color Transfer for Images and Videos* paper proposes
automatic reference-based colour transfer using affine generalized
barycentric coordinates and a constrained palette transport matrix.

U5.R2X0 asks:

> Does the automatic global path add a nonlinear colour operator beyond the
> affine baseline already present in Ultimate, and do its published
> constraints guarantee orientation and RGB-cube safety?

This is a method audit, not a film experiment.

## Exact scope

Included:

- the paper's equations (2), (3) and (5);
- fixed source palette, fixed target palette and fixed valid transport `T`;
- the automatic global recolouring path.

Excluded:

- palette extraction and clean-up;
- the proprietary CPLEX optimization that chooses `T`;
- interactive local-region constraints;
- real images and all project film pixels;
- claims about the full interactive method.

## Analytic theorem

Let source palette rows be `U`, source mean `c`, centered rows `X`, target
palette `V` and valid transport `T`. For a row-vector RGB colour `p`, the
published weights are:

```text
w(p) = (p - c) (X^T X)^-1 X^T + 1/n
```

The mapped palette is `TV`. Therefore:

```text
p' = w(p) TV
   = p A + b

A = (X^T X)^-1 X^T TV
b = mean(TV) - c A
```

For fixed palettes and `T`, the automatic global output is exactly one affine
RGB operator. Palette extraction and CPLEX may choose that affine operator,
but cannot add nonlinear response after the choice is fixed.

## Frozen witnesses

1. **Equivalence:** seven random source/target palette colours, a valid convex
   combination of two permutation transports and 10,000 RGB queries. Direct
   equation (5) and the collapsed affine operator must agree within `1e-12`.
2. **Orientation:** a valid red/blue permutation over the standard RGB
   tetrahedron has determinant `-1`. The paper's nonnegative row/column-sum
   constraints therefore do not guarantee positive orientation.
3. **Range:** mapping a valid source tetrahedron spanning `[.4,.6]` to the
   standard RGB tetrahedron yields `p' = 5p - 2`. The unit cube maps to
   `[-2,3]`. Negative barycentric extrapolation therefore has no global
   RGB-cube guarantee.

The witnesses prove insufficiency of the published structural constraints;
they do not claim that every natural-image result fails.

## Branches

- `analytic_affine_only_close`: theorem and both witnesses reproduce exactly;
  close as an Ultimate algorithm challenger.
- `published_equation_not_reproduced`: preserve discrepancy and investigate
  notation/implementation without changing the frozen witnesses.
- `witness_or_repeat_invalid`: close the audit as invalid.

No branch opens CPLEX, image execution, film/stock claims, a local-path claim
or production integration.
