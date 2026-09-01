# U7.11A Desktop Single-Look Batch Contract

## Role

`U7.11A` upgrades the existing native `U7.10A` desktop from a one-photo
export surface to one coherent multi-photo product workflow. It is not a
rerun or rescue of the closed `U7.8E` real-scale transaction and it does not
normalize or bypass the native-dialog coordinate failure that closed
`U7.10C`.

## Question

Can one Windows user complete this workflow in the existing native desktop
without preparing a manifest or typing renderer arguments?

1. choose between one and 100 existing local photographs;
2. render the existing three bounded previews for the canonical first input;
3. explicitly select one available Look Approximation and one bounded amount;
4. publish one new directory containing one RGB16 PNG and strict recipe per
   input, plus a deterministic aggregate receipt;
5. see bounded progress and either complete atomically or stop without a
   partial destination or orphan renderer.

## Frozen implementation boundary

- The existing `scripts/open_product_desktop.py` and
  `src/inference.product_desktop_ui.ProductDesktopApp` remain the one native
  product surface. A separate renderer, stock algorithm, preview path, batch
  launcher, or installer is forbidden in this leaf.
- Preview pixels still come only from
  `render_three_stock_previews_to_directory`; final pixels still come only
  from the existing isolated
  `scripts/render_film.py --product-look ... --write-recipe` entry point.
- One input preserves the existing single-file Save workflow exactly. Two or
  more inputs use the new batch path. The batch applies exactly one explicit
  look and one amount to every input; it must not silently render all three
  looks per input or route looks by content.
- Available looks remain exactly `velvia_50`, `portra_400`, and `ektar_100`.
  Every visible named-stock output is `film-inspired / Look Approximation`,
  never a calibrated stock response, physical-film reproduction, or evidence
  of stock distinguishability.
- Input paths are resolved, unique by normalized path and filesystem identity,
  canonically ordered, bounded to 100, and fully SHA-256 bound before preview
  publication. The first canonical input is visibly identified as the preview
  representative. Every input is revalidated before its child render and the
  complete set is revalidated before directory publication.
- Look amount is finite in `[0,1]`. Input-set, amount, look, product assets and
  source-commit drift invalidate the batch before final publication.
- Output names use a canonical ordinal, a lowercase ASCII `[a-z0-9_-]` stem
  normalized to at most 48 characters, and the explicit look ID. User
  basenames never become directory components without that normalization. The
  aggregate receipt exposes relative output/recipe paths, input basenames and
  hashes, but no absolute input or scratch paths.
- The selected destination directory must be absent with an existing parent.
  Every child is first rendered into one identity-owned sibling stage. Each
  recipe is rewritten before publication to bind its future final output path,
  then structurally and input-cryptographically revalidated while staged.
  Only the complete stage is renamed to the final directory; strict recipe
  file verification runs again after publication and any failure rolls back
  only still-owned published members.
- Cleanup removes only still-owned, hash-bound stage files and an unchanged
  empty stage root. A child failure, cancellation, source/config/HEAD drift,
  existing destination, or late foreign destination publishes no final batch
  and preserves foreign filesystem entries.
- The batch worker is a tracked non-daemon thread and reports completed/total
  progress through Tk `after` polling. While a batch is active, all
  input/look/amount/export controls are disabled. Cancel or window-close
  requests stop after the current child, clean owned staging, and do not leave
  an orphan worker or renderer. The Tk main thread never blocks in `join`; the
  window is destroyed only after polling observes that the worker stopped.

- Batch rendering uses the same current desktop session values as single-file
  export: PNG compression 6, tile size 256 and one tile worker. It must not
  inherit U7.8's compression-0/tile-512 research-batch settings.

## Success gates

1. A two-input batch produces exactly two distinct RGB16 PNGs, two strict
   recipes and one aggregate receipt using one explicit look/amount.
2. Every child output and recipe is byte-exact with the unchanged direct
   product CLI at the same final paths; strict recipe replay reproduces each
   PNG exactly.
3. Forward/reverse input selection produces the same canonical order, child
   identities, aggregate receipt and stable scientific identity.
4. Duplicate normalized paths, hardlink aliases, more than 100 inputs,
   invalid/nonfinite amount, export-before-preview, unknown look, source drift,
   session-asset/HEAD drift and unsafe destination names fail before final
   publication.
5. Injected child failure, cancellation after one completed child, late foreign
   destination and window-close-during-batch leave no owned final/stage
   residue, preserve foreign entries and leave no active worker/renderer.
6. The UI visibly distinguishes one-photo and N-photo states, names the first
   preview representative, freezes the selected look and amount while busy,
   reports deterministic progress and covers success, error, cancellation and
   close states without exposing absolute paths.
7. The existing single-photo desktop export, installed-runtime launcher,
   product CLI/catalog/recipe, U7.8 batch transaction and historical evidence
   semantics remain behaviorally unchanged.

## Stop rule and claim ceiling

Any formal failure closes this exact desktop-batch composition. Do not rescue
it by weakening path/hash/identity/atomicity/cancellation/report gates, by
normalizing post-result fields, by changing the renderer or look parameters,
or by rendering all three looks and discarding two. A pass opens only a
separately frozen installed-runtime batch confirmation. It does not establish
calibrated stock response, physical film reproduction, stock
distinguishability, arbitrary media support, cross-platform GUI, public
installer/package, release, or product preference.

## Verification and rollback

- Commit this contract/config before implementation.
- Targeted core/UI tests use deterministic small inputs, then direct-CLI and
  strict-replay parity, injected failure/cancel/late-foreign controls and a
  real Tk state test.
- Run behavioral U7.10A/B, product CLI/recipe and U7.8A/B/G parent regressions;
  historical hash-binding tests may retain immutable old hashes and must not be
  rewritten to pretend source files never evolved.
- Implementation/tests and formal evidence/propagation are separate scoped
  local commits. Each can be reverted independently; no push is authorized.
