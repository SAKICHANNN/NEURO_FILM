# U7.9D Private Runtime Source-Scope Binding Contract

## Parent and problem

U7.9A/U7.20F deliberately bind the private Windows runtime to one repository
commit. That protects execution identity, but it also makes the documented
runtime entrypoint unusable after an evidence-only or documentation-only
commit even when every executable product byte is unchanged.

U7.9D may change only the repository validation policy of the installed
launcher. It must not change rendering, recipes, look selection, output bytes,
dependencies, native argv handling, public packaging, or the Look
Approximation claim ceiling.

## Frozen runtime scope

The installed source commit remains the immutable comparison base. The exact
runtime scope is:

- `src/`
- `configs/`
- `scripts/render_film.py`
- `scripts/open_product_desktop.py`
- `requirements-product-v2.txt`

## Admission policy

Every launcher invocation, before importing or executing either product
entrypoint, must require all of the following:

1. the repository exists and the current `HEAD` is a descendant of the
   installed source commit;
2. all tracked repository files are clean;
3. `git diff <installed-commit> -- <runtime-scope>` is empty;
4. there are no untracked files inside the runtime scope;
5. the pinned requirements file still has its installed SHA-256 identity.

A committed change outside the runtime scope is the sole new positive case.
It may advance `HEAD`; the invoked renderer continues to record that actual
current commit in each recipe. The installed receipt must retain the original
source commit and publish the scope policy explicitly under a new receipt
schema.

## Frozen controls

- committed `docs/`-only descendant: accept;
- modified tracked documentation: reject;
- committed `src/` drift: reject;
- committed `configs/` drift: reject;
- committed product-entrypoint drift: reject;
- committed requirements drift: reject;
- untracked file within `src/` or `configs/`: reject;
- non-descendant repository history: reject;
- missing Git, requirements, root, or entrypoint: reject;
- existing/foreign destination and installation cleanup behavior: unchanged.

## Gates

- descendant-history enforcement passes;
- runtime-scope tracked drift enforcement passes;
- global tracked-clean enforcement passes;
- scoped untracked enforcement passes;
- requirements hash enforcement passes;
- committed docs-only launcher execution passes;
- receipt schema and scope disclosure pass;
- CLI and desktop launcher source share the exact policy;
- existing native argv, foreign-CWD, claim, and create-only controls remain
  unchanged;
- current Look Approximation catalog/output/recipe behavior remains exact.

## Stop rule and claim ceiling

Any failed frozen gate closes U7.9D without path-list reduction, dirty-tree
allowance, hash tolerance, recipe-schema change, installer overwrite, or
standalone-capsule rescue. A pass claims only private repository-bound runtime
usability across non-runtime descendant commits. It does not create a public
installer, standalone application, signed release, calibrated stock response,
or physical-film reproduction claim.
