# U7.10A Product Desktop Input Workflow Contract

## Role

`U7.10A` is the first repo-local desktop workflow that starts from a newly
selected user image.  It is separate from the closed `U7.3` browser family,
which can only inspect and replay already-existing recipes and outputs.

## Question

Can one Windows user complete this bounded workflow without typing renderer
arguments?

1. choose one existing local input image;
2. choose an explicit bounded look amount;
3. render and compare the three available product Look Approximation previews;
4. explicitly select one look;
5. publish one new RGB16 PNG plus its strict recipe through the existing
   create-only product transaction.

## Frozen implementation boundary

- The interface is a single native `tkinter` window launched by
  `scripts/open_product_desktop.py`.
- Preview pixels come only from
  `src.inference.three_stock_preview.render_three_stock_previews_to_directory`.
- Final pixels come only from the existing isolated
  `scripts/render_film.py --product-look ... --write-recipe` entry point.
- The available looks remain exactly `velvia_50`, `portra_400`, and
  `ektar_100`.  Every visible named-stock label must also say
  `film-inspired / Look Approximation`; the workflow must not claim stock
  calibration, physical-film reproduction, or stock distinguishability.
- Look amount is finite and bounded to `[0, 1]`.  Preview state is invalidated
  whenever input or amount changes.
- The selected input is SHA-256 bound at preview time and rechecked before
  export.  Input drift closes export.
- The only U7.10A final format is RGB16 PNG with a strict recipe.  Existing
  output or recipe destinations must remain untouched.
- Preview scratch is created only below repository-relative `tmp/`, which is
  the project-owned generated-artifact junction.  Cleanup may remove only the
  still-owned directory and still-bound preview members.
- Rendering occurs on a worker thread; the native event loop remains able to
  paint and report busy, success, and error states.

## Success gates

1. A new input can produce three distinct deterministic previews and a
   manifest at no more than 1,000,000 pixels each.
2. Preview rows and visible controls expose exactly the three authoritative
   available looks and the claim boundary above.
3. Preview input hash and look amount are revalidated before export.
4. The selected final RGB16 PNG and strict recipe equal a direct invocation of
   the unchanged product CLI.
5. Missing input, invalid/nonfinite amount, export-before-preview, input drift,
   unknown look, existing output, existing recipe, renderer failure, and
   replaced preview workspace fail closed.
6. Forward/reverse executions are scientifically identical and leave zero
   owned preview scratch.
7. The native window covers initial, ready, busy, success, error, and close
   states, keeps primary/secondary actions distinct, and exposes keyboard
   focus labels.

## Stop rule and claim ceiling

Failure closes this exact workflow without changing the renderer, stock
catalog, preview dimensions, transaction semantics, or evidence gates.  Pass
opens only a later separately frozen installed-runtime launcher integration.
It does not open calibrated stock, physical-film, public installer/package,
cross-platform GUI, batch, cache, or release claims.

## Verification and rollback

- Targeted core/UI-state tests and direct-render parity run first.
- The behavioral U7.2 product and U7.3 preview/export parent tests run next.
- A real Windows `tkinter` smoke and visual inspection are required before
  evidence closure.
- The contract/config, implementation/tests, and evidence/propagation are
  separate local commits.  Each commit can be reverted independently.
