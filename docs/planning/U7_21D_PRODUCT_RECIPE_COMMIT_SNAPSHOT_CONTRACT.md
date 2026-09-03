# U7.21D Product Recipe Commit Snapshot Contract

Status: frozen before implementation.

## Parent and defect

U7.21C binds one repository HEAD and the exact U7.9D runtime scope at product
invocation start, then revalidates both immediately before product-bundle
publication. The renderer nevertheless performs a second independent
`git rev-parse HEAD` while building the strict recipe. A transient HEAD change
between the two scope checks can therefore be serialized into the recipe even
if the repository returns to the invocation snapshot before publication.

This is a product provenance defect. It does not imply a pixel or colour error.

## Frozen behavior

1. Both supported product entry routes bind one invocation-start commit:
   `--product-look` and the explicit `safe_rich_product_v1.json` profile.
2. Their strict recipe `software.commit` must reuse that exact already-bound
   value. Product recipe construction must not perform a second live HEAD read.
3. The U7.21C start and immediate-prepublication full-scope validations remain
   unchanged and must still reject persistent drift atomically.
4. Legacy non-product rendering retains its current live HEAD recipe behavior
   and does not acquire the product scope policy.
5. Successful product image and recipe semantics other than `software.commit`
   remain byte/field exact to their current oracles.

## Adversarial control

The regression injects a different value into the recipe-time live HEAD probe
while keeping the already-bound product snapshot stable. Before the repair, the
published product recipe records the injected value. After the repair, both
product routes record the invocation snapshot, while a legacy recipe continues
to record the injected live value.

## Stop and claim ceiling

Any pixel drift, product recipe semantic drift outside `software.commit`, loss
of either U7.21C validation, legacy-policy change, non-atomic drift behavior or
parent replay/transaction regression fails closed. No threshold, scope, recipe
schema, Look, strength/effect, format or claim rescue is permitted.

Passing establishes only private product recipe-to-runtime commit consistency.
All outputs remain `film-inspired / Look Approximation`; it is not calibrated
stock response, physical-film reproduction, public packaging or cross-platform
parity.
