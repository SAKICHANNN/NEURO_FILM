# U7.2M product image + recipe transaction contract

## Question

Can the authoritative `safe-rich-product-v1 --write-recipe` CLI publish one
rendered image and its derived recipe without overwriting either destination,
without leaving an owned partial pair after a process-level failure, and
without changing the three frozen Look Approximation image bytes or recipe
semantics?

U7.2L made the product primary image create-only, but explicitly left recipe
publication outside that transaction. The current CLI publishes the image
first and then writes `<output-stem>.recipe.json` through `os.replace`. It can
therefore overwrite a pre-existing recipe and can leave a published image plus
a temporary recipe after recipe failure. This is a delivery-integrity defect,
not a colour, stock-calibration or recipe-schema question.

## Frozen change

- Apply the pair transaction only when the loaded profile is
  `safe-rich-product-v1` and `--write-recipe` is present.
- Derive the recipe path exactly as today with
  `output.with_suffix(".recipe.json")`.
- Before `load_working_image`, require input, final image and final recipe to
  be three distinct normalized absolute paths and require both destinations to
  have no directory entry. Regular files, hard links, symbolic links, broken
  links and Windows reparse entries all count as existing.
- Encode the image to a unique sibling stage whose terminal suffix remains the
  requested image suffix. Build and validate the unchanged recipe against the
  staged bytes, but record the canonical final image path.
- Encode the recipe to a unique sibling stage. Publish the image and then the
  recipe using the existing same-volume create-only primitive.
- If recipe publication fails, remove the image only while its captured
  filesystem identity still matches. Preserve any foreign replacement.
- Remove all owned stages after success or failure. Never retry by replacing a
  destination and never remove an identity-mismatched entry.
- Preserve U7.2L behavior for product renders without `--write-recipe` and all
  legacy-profile behavior.

This is process-level failure rollback and complete-pair state after return.
Two independent directory entries are not claimed to become simultaneously
visible or power-loss atomic.

## Formal gates

1. existing or aliased image/recipe destinations reject before input decode;
2. pre-existing regular, hard-link, symbolic-link, broken-link and Windows
   reparse recipe entries remain unchanged;
3. Velvia 50, Portra 400 and Ektar 100 image bytes equal the frozen U7.2L
   hashes and normalized recipe semantics equal the pre-change hashes;
4. injected image encode or recipe encode/write/publish failure leaves no
   owned final image, recipe or stage;
5. a recipe destination injected after preflight is preserved and the owned
   image is rolled back;
6. if the published image is replaced by a foreign entry before rollback, the
   foreign image is preserved;
7. two concurrent calls to one absent pair yield exactly one complete winner;
8. source bytes remain exact in every success and failure case;
9. product-without-recipe and legacy behavior retain their U7.2L identities;
10. forward/reverse fresh-process reports are byte-identical and residue-free.

## Claim ceiling and stop rule

A pass establishes only private create-only image/recipe pair publication with
identity-safe process-level rollback for existing deterministic
`film-inspired / Look Approximation` outputs. It does not establish power-loss
atomicity, calibrated stock response, physical-film reproduction, recipe
schema expansion, public packaging or installer readiness. No frozen gate may
be rescued by overwriting, retrying, clipping, changing paths or schema,
weakening identity checks, or deleting an identity-mismatched destination.
