# U7.12F Desktop Error Selection Reset Contract

## Product defect

The native Look Approximation desktop correctly invalidates `preview_ready`
after an operation error, but it retains the previously selected `style`.
When the user renders a new preview set after a failed or cancelled export,
the ordinary preview-completion path sees that stale style ID and immediately
re-enables Export.  No new radio action is required for the new preview set.

That violates the U7.10A product invariant: every valid preview set begins
without an active look, and publication opens only after a real, explicit
selection on that preview set.  A failed operation must not silently carry an
old look choice into a newly rendered preview authority.

## Frozen parent and scope

- Source parent at freeze: `c9ad2c18b5a3b51602bae13d919bd6686d8f7e13`.
- U7.10A remains authoritative for explicit look selection.
- U7.11A remains authoritative for batch export and cancellation.
- U7.12A remains authoritative for foreground-worker close safety.
- U7.12B/C remain authoritative for output formats and display-native
  previews.
- U7.12D remains a private recovery core; U7.12E remains fail-closed and
  withdrawn.  This leaf does not reopen recovery UI integration.
- The only existing implementation file that may change is
  `src/inference/product_desktop_ui.py`.  New U7.12F contract, config, tests,
  audit and evidence files may be added.  Renderer, look math,
  `product_desktop.py`, recipes, output bytes, batch transactions, cache,
  installer and recovery code are forbidden.

## Frozen intervention

- Every UI error path that marks the current previews invalid must also clear
  the selected look before controls are re-enabled.
- The reset occurs on the Tk thread and changes only the UI selection state.
- Existing error copy/dialog behavior, preview workspace preservation, owned
  cleanup, worker state and destination safety remain unchanged.
- A new preview set rendered after an error must display all three previews
  with no selected radio and Export disabled.  Only a subsequent real radio
  action may enable Export.
- Successful preview and successful export behavior remain unchanged.  A
  successful export may retain its valid preview and current selection for an
  intentional second create-only destination.
- Input changes and strength changes retain their existing explicit reset.
- Output-format changes retain the selected look because they do not change
  preview pixels or look authority.
- Close-request suppression remains unchanged: a terminal worker result after
  close does not dispatch an error or selection reset into a destroyed UI.

## Success gates

1. Reproduce the defect on the frozen parent: selected look plus an export
   error leaves the stale style populated, and a new preview completion enables
   Export without a radio action.
2. After the repair, single-photo export error followed by a new preview set
   has an empty style, disabled Export and all look radios enabled for a fresh
   choice.
3. Batch export error/cancellation followed by a new preview set obeys the same
   reset and requires a new radio action.
4. A real radio invocation after the new preview set enables Export and binds
   exactly the selected available look.
5. Successful preview/export, output-format changes, input/strength resets,
   ordinary error dialogs, batch progress/cancel and close-safety behavior
   remain passing.
6. Direct product output, strict recipe, replay and U7.11A batch bytes are
   unchanged; no renderer or publication command changes.
7. Targeted U7.10A/U7.11A/U7.12A-C behavioral regressions, Ruff, compile, JSON
   and diff checks pass.  Immutable historical current-file hash assertions
   are not rewritten.

## Stop rule and claim ceiling

Any formal gate failure closes this exact state-machine repair.  Do not rescue
it by preserving a stale style, automatically selecting a default look,
changing preview validity semantics, retaining export enablement after error,
rewriting parent evidence, altering error/cancel copy, or adding recovery,
cache, installer or wrapper variants.

A pass establishes only private Windows Tk explicit-selection safety for the
existing deterministic `film-inspired / Look Approximation` desktop.  It does
not establish calibrated stock response, physical-film reproduction, stock
distinguishability, arbitrary media/device behavior, public release,
cross-platform GUI, installer readiness or product preference.

## Verification, commits and rollback

1. Commit this contract/config before implementation.
2. Add a focused regression that demonstrates the frozen-parent trigger and
   proves the repaired state transition for single and batch errors.
3. Commit the one-line state repair plus focused tests separately.
4. Run targeted behavioral parents before formal forward/reverse audit and
   evidence.  Do not rescue a frozen formal failure.
5. Propagate tracker and agent log only after evidence is committed.  Every
   commit is local and independently revertible; no push is authorized.
