# U7.2L product create-only render transaction contract

## Question

Can the authoritative `safe-rich-product-v1` CLI publish its primary rendered
image without ever replacing an existing filesystem entry or the input image,
while preserving the three frozen colour Look Approximation bytes and every
legacy-profile behavior?

The current encoder writes a fixed sibling `.tmp` and finishes with
`os.replace`. A product invocation can therefore overwrite an existing target,
including the source image when input and output identify the same file. Two
concurrent invocations can also contend for the same fixed stage. This is a
publication-contract defect, not a colour or stock-model question.

## Frozen change

- Apply this contract only when the loaded render profile has
  `profile_id == "safe-rich-product-v1"`.
- Before `load_working_image`, reject an output path that already has any
  directory entry, including a regular file, hard-link alias, symbolic link,
  junction/reparse alias or broken symbolic link.
- Before `load_working_image`, reject input/output paths whose normalized
  absolute spellings resolve to the same path.
- Encode to a unique, exclusively-created sibling stage and publish it through
  the existing same-volume create-only primitive. A destination created after
  preflight must still win the race and remain unchanged.
- Remove every owned stage after success or failure. Never remove a foreign
  destination and never modify the input.
- Preserve the existing fixed-stage/replace behavior for non-product legacy
  profiles in this leaf.

The authoritative transaction is the primary rendered image only. Optional
recipe, metrics and layer artifacts retain their existing independently
published compatibility semantics; this leaf does not claim an all-or-nothing
multi-file transaction.

## Formal gates

1. same lexical input/output rejects before input decode;
2. pre-existing regular, hard-link, symbolic-link, broken-link and Windows
   reparse destinations reject before input decode;
3. a destination injected between preflight and publication is preserved and
   the render leaves zero owned stages;
4. two concurrent product renders to one absent target yield exactly one
   success, one fail-closed result and the winning canonical bytes;
5. injected encoder failure leaves no target or owned stage;
6. Velvia 50, Portra 400 and Ektar 100 primary output bytes remain exactly the
   frozen pre-change hashes;
7. the input bytes remain exact in every success and failure case;
8. the legacy profile continues to replace an existing target and retains its
   frozen Velvia output bytes;
9. forward/reverse fresh-process reports are byte-identical and leave no owned
   runtime residue.

## Claim ceiling and stop rule

A pass establishes only private product primary-image create-only publication
for the existing deterministic `film-inspired / Look Approximation` CLI. It
does not make optional sidecars transactional, calibrate a film stock, improve
stock separation, change colour math, open public release or authorize
installer/format/wrapper expansion. A failed frozen gate closes this exact
implementation; no relaxation of alias, predecode, create-only or byte-identity
requirements is allowed as rescue.
