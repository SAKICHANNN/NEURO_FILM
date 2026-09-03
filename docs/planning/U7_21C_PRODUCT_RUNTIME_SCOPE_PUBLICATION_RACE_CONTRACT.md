# U7.21C Product Runtime-Scope Publication Race Contract

Status: frozen before implementation
Parent: U7.21B desktop runtime-scope session binding
Claim ceiling: private deterministic `film-inspired / Look Approximation`

## Defect

The desktop validates the authoritative U7.9D runtime scope before starting a
render. The product CLI stages its output bundle transactionally, but it does
not revalidate the scope immediately before publication. An uncommitted source
or configuration change during a long render can therefore leave `HEAD`
unchanged while the staged result was produced under mixed runtime state.

## Frozen repair

For `--product-look` only:

1. load the exact U7.9D scope and bind the current committed `HEAD` before input
   inspection, pixel decode, rendering or staging;
2. revalidate the same scope and `HEAD` after all requested image, recipe,
   layer and metrics stages are complete but before bundle publication;
3. reject modified, staged, deleted or untracked scope members and any `HEAD`
   change;
4. let the existing product bundle transaction remove only its owned stages;
5. retain the existing successful image/recipe/layer/metrics bytes and public
   error boundary;
6. leave legacy/research invocations unchanged.

The scope list must continue to come from
`configs/u7_9d_private_runtime_source_scope_binding_v1.json`; no second list is
allowed.

## Gates

- predecode validation is called once for product renders;
- prepublication validation is called once after staging;
- an injected second-call rejection returns the safe CLI error code and leaves
  no output, recipe or stage residue;
- successful product output and strict replay remain exact;
- legacy non-product rendering does not acquire this product-only policy;
- U7.21B desktop session validation remains exact;
- no pixel, Look, strength, effect, format, recipe/receipt schema or claim
  change.

## Stop rules

Stop without rescue if the repair publishes any requested member after the
second validation fails, deletes a foreign entry, changes an accepted output,
weakens U7.21B/U7.2 transaction semantics, or expands into a new renderer,
format, updater, package or calibrated-stock claim.
