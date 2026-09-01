# U7.3N Preview Cache Index Create-Only Contract

## Question

Can the existing U7.3H three-stock preview cache publish
`preview-cache.json` without replacing a foreign file that appears after the
initial absence check, while preserving successful index bytes and every
U7.3H/U4.5C/U4.5E validation boundary?

## Trigger

`publish_three_stock_preview_cache_index` checks that the final index is
absent, validates the preview set, then calls `atomic_write_json`. That helper
finishes with `os.replace`, so a deterministic injected late winner at the
final path is overwritten. The cache index is the only direct final-directory
publication in this path; its sibling preview files and manifest are already
complete inputs.

## Scope

- Change only the private cache-index publisher in
  `src/inference/three_stock_preview_cache.py`.
- Encode the existing index with the exact historical canonical JSON bytes.
- Create one unique sibling stage with exclusive creation, then publish it
  through the existing same-volume create-only primitive.
- Clean up only a still-owned stage entry after failure.
- Preserve all cache schemas, fields, hashes, inspection behavior, receipt
  semantics and successful bytes.

## Required semantics

1. A final index that exists before publication rejects and remains byte exact.
2. A foreign regular file injected after validation but before the final
   publication wins; the publisher rejects and preserves it byte exact.
3. A publication failure without a final winner leaves zero owned stage
   residue.
4. Cleanup never removes a replacement whose filesystem identity differs from
   the owned stage identity.
5. Successful index bytes are byte-identical to the historical
   `indent=2`, `sort_keys=True`, UTF-8 plus final-newline encoding.
6. Existing full input/profile/output validation and receipt-bound inspection
   remain unchanged.
7. The repair is explicitly private Windows process-level ownership safety;
   it does not redefine the historical U7.3H latency result.

## Frozen gates

- existing destination preserved;
- late foreign destination preserved;
- owned residue zero after injected publication failure;
- successful index bytes exact;
- publish/inspect and receipt-bound parent behavior unchanged;
- U7.3H, U4.5C and U4.5E adjacent regressions pass;
- committed-head forward/reverse audit reports and scientific payload are
  byte exact.

## Non-goals and stop rule

No cache hashing, latency, session, receipt, renderer, preview-media, browser,
wrapper, public API, packaging, release, calibrated stock or physical-film
claim changes. Perform one prospective repair and one committed-head audit;
then stop adjacent cache work.
