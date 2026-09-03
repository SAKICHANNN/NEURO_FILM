# U7.21E Desktop Batch Aggregate Publication Scope Contract

Status: frozen before implementation.

## Parent and defect

The desktop batch path validates U7.21B's complete runtime scope before each
child and again after the child loop. It then constructs and writes the
aggregate `batch.json` receipt before atomically renaming the staged directory
to its final destination. Runtime scope drift after the loop validation but
before directory publication can therefore publish a batch under the stale
desktop session identity.

## Frozen behavior

1. The existing full `_validate_session` check remains before each child and at
   the end of the child loop.
2. After the aggregate receipt is completely staged, the same full session
   validator must run once more immediately before the destination appearance
   check and directory rename.
3. Drift injected after `batch.json` creation must publish neither destination
   nor owned stage residue.
4. A successful two-image batch must remain byte/field exact to the current
   batch, child output, recipe and receipt semantics.
5. Single-photo export, preview, input binding, cancellation, late-foreign
   destination protection and existing U7.21B/C/D guards remain unchanged.

## Stop and claim ceiling

Any pixel, child recipe, batch receipt/schema, ordering, output-format,
create-only, cleanup or product-claim drift fails closed. No weaker scope,
timing tolerance or new transaction implementation is permitted.

Passing establishes only private desktop batch runtime-scope consistency at
aggregate publication. Outputs remain `film-inspired / Look Approximation`;
this is not calibrated stock response, physical-film reproduction, standalone
packaging or cross-platform parity.
