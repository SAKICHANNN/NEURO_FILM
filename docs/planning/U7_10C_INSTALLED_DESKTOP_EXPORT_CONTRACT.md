# U7.10C Installed-Desktop Physical Export Contract

## Role

`U7.10C` closes one user-visible gap left by passed `U7.10B`: the installed
desktop was physically opened, previewed and selected, but the installed GUI's
real save dialog, PNG16 plus recipe export, and strict replay were not exercised
as one end-to-end path. This is not permission to add another launcher,
installer, UI, renderer, dependency, output format or product look.

## Question

Can one fresh exact `U7.10B` private runtime, started from a foreign current
directory, receive native preview, look-selection and Export actions, publish a
new PNG16 plus strict recipe through the real Windows save dialog, and reproduce
that output through the existing strict replay path exactly?

## Frozen implementation boundary

- The existing `install_product_runtime.py`, receipt-v2 launchers,
  `open_product_desktop.py`, desktop core/UI, renderer and replay core are
  unchanged. U7.10C may add only a committed audit, tests, evidence and minimal
  user documentation.
- The audit uses a deterministic 61x43 RGB8 input, default look amount `1.0`,
  explicit `velvia_50`, the existing three-preview path, PNG16 output and strict
  recipe. It adds no product-only automation flag or callback.
- The real installed `kmcfm-desktop.cmd` must create the Tk mainloop. External
  Win32 input clicks **Render three previews**, the Velvia radio and **Export
  PNG16 + recipe**. The actual `Export Look Approximation` common dialog receives
  one absent audit-owned destination and completes through its normal accept
  action. The real completion message is dismissed normally.
- Before the physical run, the installed CLI produces the direct oracle at the
  exact same destination spelling; its PNG and recipe bytes are retained in
  memory, then only those audit-owned oracle files are removed. The physical GUI
  must reproduce both byte-for-byte. The existing strict replay API then writes
  a distinct create-only file whose bytes match the GUI PNG exactly.
- All persistent formal reports and screenshots remain under repo-relative
  `outputs/...`; installation, input, direct oracle, GUI export, recipe, replay
  and scratch remain under one identity-owned repo-relative `tmp/...` formal
  root and are removed before report publication.
- Source commit, materialized Git objects, requirements, receipt, launcher,
  input, output and recipe identities are bound. U7.10C consumes the exact
  U7.10A export-pair preservation test and U7.10B installer existing/late-
  foreign gate as immutable parent evidence; it does not mislabel the latter
  as a new native-dialog race. Existing/foreign files remain protected, and no
  global Git, Python or OS configuration is changed.

## Success gates

1. A fresh exact receipt-v2 runtime installs from the existing 12-wheel /
   111,666,287-byte wheelhouse and passes `pip check`.
2. The desktop starts from a foreign current directory with hostile mixed-case
   `PYTHON*`; injected startup code does not execute.
3. Exactly one product window and one real save dialog appear; the three native
   click actions and dialog accept complete without direct UI method invocation.
4. Three previews become ready, the visible Velvia radio is selected and the
   final status names the created PNG and recipe.
5. Installed direct-CLI, installed-GUI and strict-replay PNG bytes are exact;
   direct and GUI strict recipe bytes are exact.
6. The PNG is RGB16, the recipe binds input/output/profile/software, style
   `velvia_50`, look amount `1.0`, and `look-approximation` with calibrated
   reference disabled.
7. Source/materialized Git objects, all three parent-evidence files and their
   frozen SHA-256 identities remain exact; the commit stays fixed and the
   tracked tree remains clean. The recipe software commit equals this exact
   source commit and its profile id/version/SHA-256 equals the exact product
   profile.
8. Forward/reverse committed-head reports and accepted screenshots are
   byte-exact; formal root, stage, window and owned process residue are zero.
9. U7.10A/B, U7.9A/C, recipe replay and create-only behavioral parents remain
   green or are verified through immutable Git-object evidence. The screenshot
   of the completed main window is frozen before formal execution and must be
   byte-exact in both committed-head runs, after manual confirmation that its
   visible status names the PNG and recipe.

## Stop rule and claim ceiling

Any failure closes this exact installed-export composition without adding UI
automation hooks, bypassing the native save dialog, changing file-dialog or
renderer semantics, relaxing byte identity, retuning a look, changing
dependencies, or opening another launcher/package variant. Pass establishes
only private repository-bound Windows installed GUI export mechanics for one
deterministic `film-inspired / Look Approximation` fixture. It is not a
standalone/public installer, signed app, arbitrary-input acceptance,
cross-platform GUI, calibrated stock response, physical-film reproduction,
stock distinguishability, preference or release approval.

## Verification and rollback

- Commit this contract/config before the first native save-dialog execution.
- Commit audit/tests separately and run behavioral parents before formal.
- Run formal forward/reverse only from a tracked-clean committed HEAD.
- Bind accepted reports, screenshots and parent evidence in a separate evidence
  commit, then propagate only the minimal user command/tracker/log facts.
- Every code change is additive and independently revertible; no push.
