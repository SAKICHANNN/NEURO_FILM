# U7.3L Recipe Replay Create-Only Repair Contract

## Question

Can strict recipe replay preserve a foreign output that appears after its
absent-path preflight, while retaining the existing successful output bytes and
removing only a still-owned publication when a later identity check fails?

## Trigger

`replay_style_safe_recipe_to_file` currently calls replacement-capable SDR
encoders and unconditionally unlinks the final output on every exception. A
deterministic injected encoder race reproduced deletion of a foreign file that
appeared after preflight. The defect reaches recipe-history export, portable
recipe replay, and the existing local browser workflow.

## Scope

- Add one private create-only SDR encoder publication result that includes the
  exact filesystem identity returned by the existing same-volume publisher.
- Route only strict recipe replay through the create-only publication path.
- Preserve the existing renderer, recipe schema, supported formats,
  compression, ICC payloads, successful bytes, output hashes, CLI arguments,
  and callers.

## Required semantics

1. Existing files and dangling symlinks reject before render or encode.
2. A destination that appears during encoding is never replaced or removed.
3. An encoder failure before publication leaves no owned output or temporary
   residue and preserves any late foreign destination byte exact.
4. A publication that is replaced by another actor before verification is not
   removed by failure cleanup.
5. A still-owned publication whose format or SHA-256 differs from the recipe is
   removed by filesystem identity, not by path alone.
6. Successful PNG8, PNG16, TIFF8 and TIFF16 replay bytes remain exact to the
   pre-repair encoders.
7. Recipe inputs, profile inputs and all foreign destinations remain immutable.

## Non-goals and claim ceiling

This repair adds no renderer, format, recipe field, browser, wrapper, installer,
release, stock calibration, physical-film response, public API, or crash/power
loss transaction. It is private process-level output ownership safety for the
existing deterministic Look Approximation replay path only.

## Verification

- focused late-foreign, replacement, symlink, owned-mismatch and success tests;
- parent replay/history/portable/browser regression tests;
- committed-head forward/reverse reports with exact scientific identity;
- Ruff, compile, JSON, scoped diff and tracked-worktree ownership review.

