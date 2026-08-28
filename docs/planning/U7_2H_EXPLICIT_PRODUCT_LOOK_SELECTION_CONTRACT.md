# U7.2H explicit product-look selection contract

## Question

Does the current product CLI silently select the Velvia 50 Look Approximation
when a caller opts into `safe-rich-product-v1` without naming a look?

The product contract requires the caller to choose a target look. Velvia 50 is
one display-proxy baseline and must not become the implicit identity of the
multi-look product. Historical CLI behavior outside the product profile remains
a compatibility boundary.

## Frozen change

- Record whether `--style` was present on the command line.
- Preserve `velvia_50` as the legacy default for the historical profile and
  the separate analytical research engine.
- After validating a render profile, reject `safe-rich-product-v1` when the
  caller did not explicitly provide `--style`.
- Keep the three explicit colour looks byte-identical and keep blocked
  `generic_bw` unavailable.
- Reject before input decode, recipe publication or output creation.

No colour parameters, profile assets, stock labels, recipes, algorithms,
separation evidence or defaults outside `safe-rich-product-v1` may change.

## Formal gates

1. omitted style plus exact product profile rejects with a stable diagnostic;
2. the rejection performs zero input decode and creates no output or recipe;
3. explicit Velvia 50, Portra 400 and Ektar 100 output/recipe bytes remain
   identical to the frozen pre-change oracle;
4. omitted style under the exact legacy profile remains byte-identical to
   explicit Velvia 50;
5. blocked generic B&W behavior is unchanged;
6. forward/reverse fresh-process reports are byte-identical and leave no owned
   residue.

## Claim ceiling and stop rule

A pass is only a product-selection truth repair. It does not improve or
validate any stock operator, repair Portra/Ektar separation, promote AO6,
calibrate film, or open new UI/wrapper work. A failure reverts the change; no
profile-ID, argument-parser or compatibility rescue is allowed in this leaf.
