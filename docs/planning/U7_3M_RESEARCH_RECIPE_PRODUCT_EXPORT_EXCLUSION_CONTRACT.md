# U7.3M Research Recipe Product-Export Exclusion Contract

## Question

Can every product-facing recipe export path reject a structurally valid but
explicitly research-only staged-density recipe before replay, while preserving
that recipe as read-only historical evidence and leaving ordinary product Look
Approximation exports byte exact?

## Trigger

A committed-head witness generated one valid v4 recipe whose halation model is
`staged-density-research`. `build_render_recipe_history` correctly retained the
recipe as historical evidence, but `build_recipe_export_request_set` also
published one request under the claim that every entry was an existing
film-inspired Look Approximation. Direct history export and the local browser
share that product-export route. Structural recipe validity is therefore being
mistaken for product-export eligibility.

## Frozen mechanism

- Keep the general recipe-history schema and default behavior unchanged. A
  valid research recipe remains visible as valid history.
- Add one explicit product-export-only validation mode. It reuses the strict
  recipe validator and rejects exactly the `staged-density-research` halation
  model with the stable invalid-row code `recipe_research_only`.
- Build offline export-request sets and perform direct history export through
  that product-export-only history view. The local browser inherits the same
  exclusion through its unchanged request-set builder.
- Do not delete, rewrite, relabel or weaken any existing research recipe,
  renderer, replay function, profile, output, evidence or scientific claim.

## Required gates

1. The default history view still admits the exact staged-density v4 recipe as
   structurally valid historical evidence.
2. The product-export-only history view reports that same recipe as invalid
   with `recipe_research_only`.
3. An offline product export-request set publishes zero requests for a
   research-only history root.
4. Direct history export rejects before calling replay and publishes no output.
5. The browser session refuses to open when research-only recipes are the only
   available rows.
6. One ordinary product Look Approximation recipe remains exportable and its
   request and replay output bytes remain unchanged.
7. Mixed history publishes only the ordinary product recipe; recipe/source/
   profile inputs remain immutable and owned scratch is removed.
8. Fresh-process forward/reverse scientific payloads are exact, and focused
   plus adjacent recipe-history/export/browser regressions pass.

## Stop rule and claim ceiling

This is one product-safety repair, not a new U7.3 wrapper. Do not add formats,
pages, recipe fields, renderers, effects, discovery rules, stock calibration,
physical-film claims, public APIs, packaging or release state. Stop after the
three existing product-export entry paths share the exact exclusion and their
ordinary successful bytes remain exact.

