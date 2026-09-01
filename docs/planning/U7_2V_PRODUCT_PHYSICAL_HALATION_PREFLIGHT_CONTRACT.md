# U7.2V product physical-halation preflight contract

## Question

Does the first-class `--product-look` entry reject the unbounded expert
physical-halation surface and every invalid locked-control value before it
reads an input image or creates a product artifact?

U7.2R bounds only the top-level grain, halation and dust strengths plus the
seed. U7.2U excludes the staged-density research model. The remaining
`physical` route still accepts expert controls and invalid locked-control
values through argument parsing; a zero-pixel witness with
`--halation-expert-controls --halation-background-gain nan` reaches
`load_working_image`. This is a product predecode and bounded-effect defect,
not a new halation experiment.

## Frozen change

- Apply the repair only when `--product-look` and
  `--halation-model physical` are both present.
- Require the existing `locked` physical-control mode; reject `expert` through
  the argument parser before profile loading, transaction setup, input decode
  or publication.
- Validate every explicitly supplied locked numeric override as finite and in
  its existing resolver domain: `amount` in `[0,2.4]`; `impact`,
  `anti_halation`, `source_selectivity`, `diffusion`, `warm_core` and
  `background_visibility` in `[0,1]`.
- Resolve the selected preset and the fully assembled locked controls through
  the existing strict resolver during product preflight, so unknown presets
  and invalid type/family/colour-response combinations reject predecode.
- Keep non-product expert/research routes, physical resolver arithmetic,
  presets, defaults, recipe schema, renderer and historical evidence
  unchanged.
- Do not change any product output pixel, catalog row or claim label.

## Frozen gates

1. each available product look rejects expert physical controls through the
   parser before reading a deliberately missing input;
2. NaN, positive/negative infinity, below-minimum and above-maximum values for
   every locked numeric override reject predecode and publish nothing;
3. an unknown preset and invalid strict categorical combinations reject
   predecode;
4. the domain endpoints, every existing preset, and the default locked
   physical selector cross the repaired preflight;
5. the non-product expert route crosses this new product-only preflight;
6. existing product `simple` behavior and the three U7.2O default product
   output/normalized-recipe identities remain unchanged;
7. forward/reverse reports are byte-identical, source locks pass and owned
   runtime residue is zero.

## Claim ceiling and stop rule

A pass establishes only that the existing private Look Approximation product
entry exposes one finite, domain-bounded locked physical-halation surface and
does not expose the expert surface. It does not validate physical-halation
realism, add an effect, change research behavior, or establish package,
installer, release, calibrated-stock or physical-film claims. Any gate failure
closes this exact repair without parameter rescue, algorithm changes or
adjacent effect-option expansion.
