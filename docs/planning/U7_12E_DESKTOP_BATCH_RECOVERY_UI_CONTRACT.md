# U7.12E Desktop Batch Recovery UI Contract

## Product question

Can the existing private desktop deliberately expose the proven U7.12D
restart-safe one-look batch transaction, so cancellation, window close, or a
later process can resume already verified photos without changing U7.11A
output bytes, recipes, receipt, or legacy core semantics?

This is a narrow UI integration leaf. It is not another recovery algorithm,
colour model, wrapper family, installer, public API, or stock-calibration
claim.

## Frozen dependencies and ownership

- U7.12D terminal evidence commit: `4ad662a6f`.
- U7.12D evidence SHA-256:
  `f052be6957f95f2029728c12d7946d29263d031b0216c3f9c80b7188400db58c`.
- U7.12D implementation commit: `3cf53cc7a`.
- U7.11A remains authoritative for final PNG16/recipe/batch receipt bytes.
- U7.12A remains authoritative for non-daemon worker and close safety.
- This leaf may change only `product_desktop_ui.py` plus its own new contract,
  config, tests, audit and evidence. It must not change `product_desktop.py`,
  U7.12D transaction semantics, renderer/look math, recipes or final receipt.

## Deliberate UI route

For two or more bound inputs, the existing button copy becomes **Export /
resume N PNG16 + recipes**. One-photo export and its format selector are
unchanged. After the user chooses an absent final folder path, the UI derives
one deterministic sibling workspace:

`.kmcfm-u7-12e-<first 32 lowercase hex digits of SHA-256(normalized absolute destination path)>.recovery`

Normalization is exactly `normcase(abspath(fspath(resolved_destination)))`
encoded as UTF-8. The workspace path is never user-selectable, never placed
inside the destination, never included in the final receipt, and never reused
for a different destination identity. The UI calls the existing versioned
U7.12D entry point explicitly with the current canonical bound inputs,
selected look, cancellation event, progress callback, deterministic workspace,
and chosen destination.

## Pause, resume and terminal behavior

- A final `DesktopBatchReceipt` follows the existing completion dialog/status.
- A `DesktopBatchProgressReceipt` is a truthful successful pause, not an
  error. The UI returns to an enabled preview state and reports exact completed
  and remaining counts plus the instruction to select the same inputs, look,
  strength and destination to resume.
- Cancel says **Pausing safely after the current photo…** and sets only the
  existing event. It does not delete verified children.
- Close during a batch sets the same event, waits for the foreground batch
  worker, suppresses dialogs, allows U7.12D to retain the workspace, then
  cleans preview state and destroys Tk exactly once.
- A foreign/mismatched/tampered workspace remains a fail-closed error from
  U7.12D. The UI neither deletes it nor offers a reset/overwrite rescue.
- Once the final destination validates, U7.12D monotonic-success behavior and
  the ordinary completion dialog remain authoritative.

## Frozen verification

Success requires:

1. one-photo labels, format routing and export remain exact;
2. multi-photo button/status copy explicitly says export/resume;
3. deterministic workspace derivation is order/case normalization stable,
   sibling-only, fixed-length and destination-distinct;
4. the batch UI calls U7.12D, never legacy `export_batch`, with exact bound
   inputs/look/workspace/destination/cancel/progress;
5. a paused receipt restores controls without error dialog and reports exact
   counts/instructions;
6. a final receipt follows unchanged completion behavior;
7. cancel and close preserve the workspace and non-daemon wait semantics;
8. mismatched workspace/foreign destination errors remain fail-closed and are
   not deleted or normalized;
9. one real three-input integration pauses after one child, resumes in a fresh
   UI/workflow session and matches unchanged U7.11A bytes and receipt;
10. focused U7.11A/U7.12A-D parents, Ruff, compile, JSON and diff pass.

## Stop rule and claim ceiling

Any formal gate failure closes this exact UI integration without changing the
workspace name, cancel semantics, messages, cohort, thresholds, or U7.12D
core. Do not add workspace browsers, reset/delete buttons, installed-runtime
variants, or adjacent recovery wrappers.

A pass establishes only private Windows/Python UI invocation of the existing
deterministic film-inspired / Look Approximation recovery core. It does not
establish hard-power-loss durability during a child, arbitrary-input or 24MP
performance, calibrated stock response, physical-film reproduction, stock
distinguishability, public API, installer, release readiness, or product value.
