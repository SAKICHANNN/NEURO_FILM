# U7.2G — Generic B&W severe-veto product contract

## Question

After BW2.D2 confirmed renderer-created highlight contours on three of sixteen
source-disjoint photographs, can product discovery preserve the historical
`generic_bw` identity while preventing new product execution and recipe replay
from presenting it as an available look?

## Frozen change

- Keep `generic_bw` discoverable so existing recipes and user interfaces can
  explain why the look is unavailable.
- Add an explicit `availability` field to every product-catalog row.
- Mark the three colour Look Approximation controls `available`.
- Mark `generic_bw` `blocked_severe_artifact` and bind its reason to BW2.D2.
- Reject `generic_bw` in the authoritative product dispatcher before runtime
  asset lookup or pixel rendering, including amount zero.
- Reject `generic_bw` in the product CLI before input decode or output/recipe
  creation.
- Preserve the historical `generic_bw` module and BW2.D1 recipes as mechanical
  replay evidence; do not alter their parameters or outputs.

## Gates

1. The catalog order remains Velvia 50, Portra 400, Ektar 100, generic B&W.
2. All three colour rows remain available and byte-exact to their existing
   dispatchers.
3. `generic_bw` discovery reports the severe-artifact block and the exact
   BW2.D2 evidence identity.
4. Direct product dispatch rejects `generic_bw` before calling its renderer.
5. CLI execution rejects `generic_bw` before loading the input image and leaves
   no output or recipe artifact.
6. Named HP5/Tri-X IDs remain unsupported.
7. Historical BW2.D1 direct replay tests remain unchanged.

## Claim ceiling

This is a product severe-veto propagation. It is not a repair, a new B&W look,
an HP5/Tri-X claim, a stock response, calibration evidence, or completion of
the multi-stock system.
