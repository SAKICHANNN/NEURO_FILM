# U7.3K Recipe Directory Create-Only Repair Contract

## Question

Can the existing U7.3E offline workspace and U7.3F export-request set preserve
every foreign destination entry when another actor wins or mutates the
destination after preflight, while keeping successful directory bytes exact?

## Trigger

Both materializers currently preflight an absent destination, call
`Path.mkdir`, and then recursively delete whatever occupies that path when any
later operation raises. A deterministic injected race reproduced deletion of a
foreign directory and its `foreign.bin` member in both paths. Their member
writes also use replacement-capable `write_bytes`.

## Scope

- Add one private inference helper for a flat create-only directory.
- Route only `materialize_offline_recipe_workspace` and
  `materialize_recipe_export_request_set` through it.
- Preserve the existing builders, schemas, filenames, receipts, HTML, recipe
  replay, CLI arguments, and successful bytes.

## Required semantics

1. The final directory is claimed only by one atomic `mkdir`.
2. Every intended member is opened with exclusive creation.
3. Owned directory and file identities are captured from the filesystem.
4. Success requires the exact intended member set, identities, lengths and
   SHA-256 payloads.
5. Failure cleanup removes only still-matching owned files, then removes the
   still-matching owned directory only when empty.
6. A foreign directory that wins the initial race, a foreign member added to
   an owned directory, and a foreign replacement of an owned member all remain
   byte exact.
7. No recursive deletion is used by either repaired materializer.

## Non-goals and claim ceiling

This repair does not add a browser, wrapper, renderer, stock response,
calibration, installer, release, crash/power-loss transaction, public API, or
new output format. It is private process-level ownership safety for the
existing Look Approximation desktop/export files only.

## Verification

- focused success and race controls for both U7.3E and U7.3F;
- parent U7.3A-J regression tests;
- forward/reverse committed-head audit reports with exact payload identity;
- Ruff, compile, scoped diff and tracked-worktree ownership review.
