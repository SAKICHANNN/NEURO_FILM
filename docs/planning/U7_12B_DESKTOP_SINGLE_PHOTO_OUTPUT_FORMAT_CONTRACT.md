# U7.12B Desktop Single-Photo Output Format Contract

## Role

`U7.12B` closes a concrete delivery gap in the native Look Approximation
desktop: the public product CLI already encodes PNG16, TIFF16 and JPEG8, while
the one-photo desktop hard-codes PNG16. This leaf exposes those existing
encoders through one explicit desktop choice. It does not add a codec, change
look pixels, reopen the closed installed-runtime batch family, or change the
U7.11A PNG16 batch contract.

## Parent state and readiness

- Frozen source parent: commit
  `3c1c079692991a11e768784d808a86de6fe3f930`.
- Existing direct encoders remain `scripts/render_film.py` with
  `--product-look`, bounded `--look-amount`, `--write-recipe`, tile size 256
  and one tile worker.
- U7.2T already freezes the supported 8-bit product suffix set and U7.3L
  freezes strict replay create-only publication. U7.11A freezes batch output
  as PNG16.
- The separately owned U7.12A foreground-worker close-safety repair is a hard
  implementation dependency. Before the first U7.12B product render or formal
  run, an additive execution lock must bind its exact committed core, tests and
  evidence. U7.12B must not edit the shared desktop files while U7.12A owns
  them.

## User workflow

For exactly one selected input and one explicit available Look Approximation,
the desktop presents one visible output-format choice:

| Choice | Canonical extension | Accepted extensions | Recipe format | Bit depth |
| --- | --- | --- | --- | ---: |
| PNG16 | `.png` | `.png` | `PNG` | 16 |
| TIFF16 | `.tiff` | `.tif`, `.tiff` | `TIFF` | 16 |
| JPEG8 | `.jpg` | `.jpg`, `.jpeg` | `JPEG` | 8 |

PNG16 is the default and preserves the exact existing desktop command. TIFF16
and JPEG8 use only the existing CLI encoder selected by extension and bit
depth. No JPEG quality, chroma-subsampling, TIFF compression, profile, look,
strength, seed, tiling or colour-math control is added or changed.

## Frozen product boundary

- The selection is explicit, visible beside the one-photo export action,
  keyboard reachable and disabled while a foreground operation is active.
- Changing output format does not invalidate already hash-bound previews: it
  changes publication encoding only, not the selected look, amount or preview
  pixels.
- For two or more inputs the selector is disabled and visibly states that the
  atomic U7.11A batch remains PNG16. Batch commands, output names, recipes and
  `batch.json` must remain byte-exact with the current path.
- The core receives an explicit format ID. The chosen ID, destination suffix,
  CLI bit depth and resulting strict-recipe `format`/`bit_depth` must agree.
  Unknown IDs, missing suffixes and mismatches reject before `command_runner`
  and before output or recipe publication.
- PNG16 alone receives the existing explicit PNG compression value. TIFF16 and
  JPEG8 commands must not carry `--png-compression`.
- The output and `.recipe.json` destinations remain absent, outside the
  preview workspace and protected by the existing create-only/late-foreign
  transaction. Source, session assets and HEAD are revalidated unchanged.
- Successful output is fully verified by the strict recipe and strict replay
  must reproduce the output byte-for-byte for each format.
- Every named look remains `film-inspired / Look Approximation`; the format
  selector does not imply calibrated stock response, physical-film
  reproduction, stock distinguishability or wide-gamut/HDR publication.

## Success gates

1. PNG16 remains the default and its workflow command is byte-for-byte equal to
   the pre-U7.12B single-photo command.
2. PNG16, TIFF16 and JPEG8 each match a direct CLI render at the same final
   path in image bytes and raw strict-recipe bytes.
3. Strict replay reproduces each of the three desktop outputs byte-for-byte.
4. Recipe format, bit depth, output path/hash, explicit look/amount, software
   commit and Look Approximation claim are exact for every format.
5. Unknown format, suffix mismatch, missing suffix, export-before-preview and
   batch format mutation reject before the renderer and publication.
6. Existing and late-created foreign output/recipe destinations are preserved;
   failure leaves zero owned output, recipe, stage, worker or workspace
   residue.
7. The U7.12A close-during-preview/export controls and ordinary foreground
   success/failure behavior remain passing for every one-photo format.
8. U7.11A batch artifacts and command tuples remain exact; U7.2T, U7.3L,
   U7.10A and product CLI/recipe parent regressions pass behaviorally.

## Stop rule and claim ceiling

Any formal gate failure closes this exact format-selection composition. Do not
rescue it by changing encoder quality/compression, accepting a mismatched
suffix, normalizing output or recipe bytes, weakening replay/create-only
checks, changing the renderer or look parameters, or expanding batch/runtime
wrappers. A pass establishes only private native single-photo SDR export
mechanics for existing Look Approximations. It does not establish a public
release, installer, cross-platform GUI, arbitrary media/ICC/HDR support,
calibrated stock response, physical film reproduction, stock
distinguishability or preference.

## Verification, commits and rollback

1. Commit this contract/config before implementation.
2. After U7.12A releases the shared files, commit one additive execution lock
   binding its exact committed artifacts before the first product render.
3. Commit core/UI/tests separately from the formal controller and evidence.
4. Use deterministic small RGB input for format/control tests, then exact
   direct-CLI and strict-replay comparison for all three formats in
   forward/reverse order.
5. Run targeted U7.12A/B plus behavioral U7.11A, U7.10A, U7.3L, U7.2T and
   product CLI/recipe regressions. Historical hash evidence remains immutable.
6. Propagate only after evidence is committed. Each scoped commit is locally
   revertible; no push is authorized.

