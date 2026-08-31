# U7.2N product auxiliary-output transaction contract

## Question

Can the authoritative safe-rich-product-v1 CLI publish the primary image and
every explicitly requested recipe, layer and metrics artifact as one
create-only process-level bundle, without overwriting any destination or
leaving owned partial output after failure, while preserving the frozen Look
Approximation bytes and semantics?

U7.2L made the primary image create-only and U7.2M transacted the image plus
recipe. The write-layers and write-metrics paths still run after that pair is
published: layer images use ordinary saving and metrics uses write_text.
They can overwrite foreign entries, and an auxiliary failure can return
nonzero after leaving a successful image/recipe pair. This is a product
delivery-integrity defect, not a colour, film-stock or schema question.

## Frozen change

- Apply the bundle transaction only to safe-rich-product-v1 when at least one
  of write-recipe, effective write-layers, or write-metrics is requested.
  Effective layers means that an enabled effect will actually produce at least
  one layer. Preserve U7.2L for product image-only calls and every legacy path.
- Derive final paths exactly as today: the requested image, stem.recipe.json,
  stem.metrics.json, and stem_layers/.
- Before load_working_image, require the input and every requested final entry
  to be distinct normalized absolute paths. Every final entry must be absent;
  files, directories, links, broken links and reparse entries count as existing.
- Stage the image, optional canonical recipe, every layer image and optional
  metrics bytes in unique same-parent entries. Bind each regular file by
  filesystem identity, byte length and SHA-256. Bind the layer stage root and
  its complete sorted relative-file manifest; reject links, nested directories,
  missing files or content/identity drift.
- Build the recipe against staged image bytes while recording the canonical
  final image path. Build metrics from the unchanged render facts and staged
  recipe SHA when requested. Preserve current JSON bytes.
- Only after every requested artifact is sealed, publish image, recipe, layer
  directory contents, and metrics through create-only operations. Create the
  layer root exclusively and publish each file create-only.
- On publication failure, roll back earlier bundle entries in reverse order
  only while captured identities still match. Remove the layer root only when
  its identity still matches and it is empty. Preserve every foreign entry.
- Remove all still-owned stages after success or failure. Never replace a
  destination and never delete an identity-mismatched entry.

This is process-level all-requested-artifacts completeness after return. It
does not claim simultaneous directory visibility or power-loss atomicity.

## Frozen gates

1. existing or aliased requested image/recipe/layer-root/metrics entries reject
   before input decode and remain unchanged;
2. the three U7.2M image and normalized-recipe identities remain exact;
3. the frozen Ektar effects image, normalized recipe, normalized metrics and
   five layer-file identities remain exact;
4. injected image, recipe, layer encode, layer publish, metrics encode or
   metrics publish failure leaves no owned final or stage artifact;
5. foreign destinations injected after preflight are preserved while all
   still-owned earlier publications roll back;
6. foreign replacements and foreign additions inside an owned layer root are
   preserved; only matching owned children may be removed;
7. in-place and identity-changing mutation of every stage class rejects before
   publication;
8. two concurrent identical calls yield exactly one complete bundle winner;
9. source bytes remain exact in every success and failure case;
10. product image-only, product image+recipe, and legacy behavior retain their
    prior identities and semantics;
11. forward/reverse fresh-process reports are byte-identical and owned residue
    is zero.

## Claim ceiling and stop rule

A pass establishes only private create-only process-level publication and
identity-safe rollback for requested artifacts of existing deterministic
film-inspired / Look Approximation renders. It does not establish simultaneous
visibility, crash/power-loss atomicity, calibrated stock response, physical-film
reproduction, public packaging, installer or release readiness. No frozen gate
may be rescued by overwriting, retrying, changing paths/content/schema/effects,
weakening identity checks, or deleting an identity-mismatched destination.
