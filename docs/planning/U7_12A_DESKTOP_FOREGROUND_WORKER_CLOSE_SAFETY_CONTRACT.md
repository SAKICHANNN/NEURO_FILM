# U7.12A Desktop Foreground Worker Close-Safety Contract

## Role

`U7.12A` repairs one user-visible lifecycle defect in the existing native
Look Approximation desktop. Preview and one-photo export currently run in an
anonymous daemon thread, while window close synchronously calls the workflow
cleanup that waits on the same workflow lock. Closing during either operation
can therefore freeze the Tk main thread and races queued completion callbacks
against root destruction.

This leaf changes no renderer, look, preview pixels, output bytes, recipe,
batch transaction, installer or native-dialog automation.

## Frozen intervention

- Track at most one preview or one-photo-export worker explicitly.
- The worker is non-daemon and writes only a terminal result or error. It must
  not call Tk from the worker thread.
- Tk polls the worker. Ordinary terminal success/error is dispatched on the Tk
  thread exactly once.
- Closing during an active foreground operation marks the application closing,
  disables controls, returns without joining or taking the workflow lock, and
  waits asynchronously for the current operation to finish.
- Once that worker terminates, close performs workflow cleanup and destroys the
  Tk root exactly once. It does not show a success/error dialog or invoke the
  operation completion callback after close was requested.
- Preview and one-photo export remain non-cancellable; the UI accurately says
  that it is closing after the current operation. Batch cancel/close behavior
  remains unchanged.
- A second foreground operation is rejected while one is tracked.

## Gates

1. Preview-close and one-photo-export-close both return promptly from the Tk
   callback, retain a live non-daemon worker until release, then clean and close
   exactly once.
2. No success callback, error dialog or other Tk completion callback executes
   after close was requested.
3. Ordinary preview/export success and failure clear tracked state and preserve
   existing UI behavior.
4. A concurrent foreground start rejects without replacing the first worker.
5. Existing batch non-daemon, cancellation, progress and close behavior remain
   exact.
6. Existing U7.10A/U7.11A output, recipe, selection and workspace ownership
   behavior remains passing; foreign entries are never deleted.

## Stop rule and claim ceiling

Any gate failure closes this exact repair. Do not rescue by making preview or
single export cancellable, killing renderer children, weakening workspace or
destination ownership, adding UI automation hooks, or changing renderer/
installer/output-format behavior. A pass establishes only private Windows Tk
foreground-operation close safety for the existing film-inspired / Look
Approximation desktop. It is not a calibrated stock, physical-film, public
release, cross-platform GUI or installer claim.

