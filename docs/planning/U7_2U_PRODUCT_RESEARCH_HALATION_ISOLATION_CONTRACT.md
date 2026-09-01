# U7.2U product/research halation isolation contract

## Question

Does the first-class `--product-look` entry reject the explicitly research-only
`--halation-model staged-density-research` route before it reads an input image
or creates a product artifact?

U1.6G4J admitted staged-density execution only as a private opt-in research
route and explicitly denied a product claim. The current CLI nevertheless
accepts it together with `--product-look` when its research controls are
satisfied, then reaches input decode while presenting the authoritative product
selector. This is a product/research isolation defect, not a halation experiment.

## Frozen change

- Apply the repair only when `--product-look` is present.
- Reject `--halation-model staged-density-research` through the argument parser
  before profile loading, transaction setup, input decode or publication.
- Use one stable message that names the research-only model.
- Keep product `simple` and `physical` model selection unchanged.
- Keep the existing non-product staged-density research route, its required
  controls, recipe schema, executor and evidence unchanged.
- Do not change any halation algorithm, parameter, default, profile, product
  catalog row, output pixel or claim label.

## Frozen gates

1. each available product look rejects the staged-density selector with parser
   status `2` before reading a deliberately missing input;
2. rejection remains predecode whether or not the caller also supplies the
   staged-density research control combination;
3. every rejection names `staged-density-research`, does not disclose/read the
   missing source and publishes no image, recipe, layers or metrics;
4. product `simple` and `physical` selectors cross this new preflight;
5. a non-product invocation with the exact valid U1.6G4J staged-density
   controls crosses this new preflight and retains its historical validation;
6. the three U7.2O product output identities and normalized recipes remain
   unchanged under parent-chain regression;
7. forward/reverse reports are byte-identical, source locks pass and owned
   runtime residue is zero.

## Claim ceiling and stop rule

A pass establishes only that the existing private Look Approximation product
entry cannot select one explicitly research-only halation route. It does not
validate simple or physical halation realism, add a product effect, change
staged-density research evidence, or establish package, installer, release,
calibrated-stock or physical-film claims. Any gate failure closes this exact
repair without deleting research functionality, changing algorithms or
expanding adjacent effect-option policy.
