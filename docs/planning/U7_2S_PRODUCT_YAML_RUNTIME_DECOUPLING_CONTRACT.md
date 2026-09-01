# U7.2S Product YAML runtime decoupling contract

## Parent and purpose

U7.2S is a product-delivery child of U7.2Q and the accepted U7.2R product CLI.
U7.2Q established that the exact binary-only product environment cannot be
installed because `omegaconf==2.3.0` requires source-only
`antlr4-python3-runtime==4.9.3`. U7.2S tests a materially different mechanism:
remove OmegaConf/ANTLR from the product import and configuration path while
preserving the exact deterministic Look Approximation renderer.

This is not a dependency-version rescue. The U7.2Q manifest and evidence stay
immutable. U7.2S uses a new versioned manifest and a narrow strict PyYAML
mapping loader for the one product profile document.

## Frozen observations before implementation

- `configs/color_rendering_profiles.yaml` contains no interpolation, YAML tag,
  anchor, alias, merge key, or non-finite literal.
- In the qualified project environment, OmegaConf
  `to_container(load(...), resolve=True)` and `yaml.safe_load(...)` produce
  equal Python objects and equal canonical JSON for that exact document.
- Importing `scripts.pipeline_color_baseline`, `src.inference.render_contract`,
  or `src.inference` currently imports both `omegaconf` and `antlr4`.
- Product use of OmegaConf is limited to profile loading in
  `scripts/pipeline_color_baseline.py` and legacy-profile migration in
  `src/inference/render_contract.py`; guardrails and recipes are JSON.

## Allowed implementation

- Add one private strict YAML mapping loader under `src/inference/` using
  `yaml.SafeLoader` semantics.
- Reject duplicate mapping keys, non-mapping roots, custom tags, malformed
  YAML, and unresolved `${...}` interpolation strings.
- Replace only the two product-path OmegaConf loads identified above.
- Add `requirements-product-v2.txt`, equal to the U7.2Q package set except
  that `omegaconf` and `antlr4-python3-runtime` are absent.
- Add focused tests and a dedicated formal audit/evidence bundle.

## Forbidden changes

- No edits to `requirements-product.txt` or U7.2Q evidence.
- No package-version substitutions, source-build fallback, editable install,
  dependency resolver override, or unpinned package.
- No change to product look values, stock labels, profile/config assets,
  recipe schemas, effect semantics, rendering algorithms, output encoding,
  product CLI arguments, or U7.2R argument gates.
- No expansion of the research/evaluation scripts that still use OmegaConf.
- No calibrated-stock, physical-film, package, installer, or public-release
  claim.

## Formal gates

1. The strict loader returns canonical JSON exactly equal to the frozen
   OmegaConf result for the tracked product YAML.
2. Duplicate keys, non-mapping roots, custom tags, malformed YAML, and
   unresolved interpolation strings all fail closed.
3. Fresh subprocess imports of `scripts.pipeline_color_baseline`,
   `src.inference.render_contract`, and `src.inference` load neither
   `omegaconf` nor `antlr4`.
4. The v2 requirements manifest is exact, pinned, binary-only installable,
   and passes `pip check` in two independently created Windows CPython 3.12.10
   environments under repo-relative project scratch.
5. Each fresh environment runs the unchanged accepted U7.2R audit from the
   committed tree. Its product images, recipes, metrics, auxiliary bundle,
   controls, and scientific report identity are exact to the frozen U7.2R
   baseline.
6. Current profile migration and focused/adjacent U7.2 product regressions pass.
7. Formal environment, report, media, and stage residue is zero after both
   runs; source inputs and foreign destinations remain immutable.
8. Forward/reverse complete formal reports are byte-exact.

Any failed gate produces `FAIL_CLOSED` without changing the manifest, loader,
profile, renderer, cohort, baseline, or gate. A pass supports only a private
Windows Python 3.12 Look Approximation runtime dependency boundary. It does not
establish an installer, package, release, calibrated stock response, physical
film reproduction, or a completed multi-stock system.
