# U7.2R product effect-argument preflight contract

## Question

Does the first-class `--product-look` CLI reject effect values that cannot be
represented by the existing deterministic recipe contract before it reads an
input image or creates any product artifact?

The product recipe already requires finite grain, halation and dust strengths
in `[0,1]` and a signed 32-bit seed. The public CLI currently parses wider
Python numeric values, reaches `load_working_image`, and only rejects some of
them after rendering when a recipe is requested. Negative or NaN strengths can
also silently behave as disabled effects. This is a product input-safety defect,
not a colour, stock, FilmFX or physical-halation experiment.

## Frozen change

- Apply this repair only when `--product-look` selects the existing
  `safe-rich-product-v1` path. Historical/research invocations retain their
  current behavior.
- Before profile loading, transaction preflight, input decode or publication,
  require `--grain`, `--halation` and `--dust` to be finite values in `[0,1]`.
- At the same boundary require `--seed` to fit the existing signed 32-bit
  recipe field (`-2147483648..2147483647`).
- Reject through the argument parser with a stable option-specific message.
  Do not clamp, normalize, reinterpret or silently disable invalid values.
- Do not change physical-halation expert controls, effect algorithms, recipe
  schema, stock catalog/profile, output transaction or legacy CLI semantics.

## Frozen gates

1. `NaN`, positive/negative infinity, a negative finite value and a value above
   one reject for each of grain, halation and dust before input decode;
2. seeds immediately below/above signed int32 reject at the same boundary;
3. all invalid cases return parser status `2`, name the exact option, do not
   mention/read the missing input sentinel and publish no output or auxiliary
   artifact;
4. strengths `0` and `1` and both signed-int32 seed endpoints are accepted;
5. the existing Ektar product render with grain `.05`, halation `.15`, dust
   `.02`, recipe, layers and metrics retains its exact U7.2O image/layer and
   normalized recipe/metrics identities;
6. all three default product looks retain their current image bytes;
7. legacy non-product invalid values retain their pre-change behavior;
8. forward/reverse fresh-process reports are byte-identical, source inputs are
   immutable and owned runtime residue is zero.

## Claim ceiling and stop rule

A pass establishes only predecode numeric-boundary enforcement for the existing
private first-class Look Approximation CLI. It does not validate physical film,
calibrated stock response, FilmFX realism, arbitrary expert controls, public
packaging, release readiness or cross-platform behavior. Any failure closes
this exact repair; do not rescue it by changing algorithms, schemas, profiles,
effect ranges, frozen output identities or research-mode semantics. Stop
adjacent argument-wrapper expansion after closure.
