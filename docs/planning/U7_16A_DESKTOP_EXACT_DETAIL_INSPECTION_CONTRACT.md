# U7.16A Desktop Exact Detail Inspection Contract

## Purpose

`U7.16A` fixes a concrete product-inspection gap in the private native desktop.
The current window shows one truthful input-basis thumbnail and three bounded
`300x260` Look Approximation cards, but it provides no way to inspect faces,
text, edges, grain-like detail, or local clipping at native output sampling.
Enlarging those card pixels would create a false detail claim, and rendering an
isolated source crop would recompute safe-Lab whole-image context on the wrong
population.

The repair adds one explicit user-selected detail point and an on-demand
spatial `1:1` crop.  The crop must come from a temporary full-resolution result
created by the unchanged product export command, strict recipe validation, and
currently selected output format.  The temporary image and recipe are removed
after the crop is bound.  The main export still performs its existing
create-only render and remains byte-identical.

## Parent, ownership, and prior evidence

- Frozen parent HEAD: `e76cd5501a267e8ec63c9404b1d5d96ed60038c1`.
- U7.15C evidence SHA-256:
  `d6191cd954a21e387aa115f3cc910294b4ef8336009798c51111fc4d30daf374`.
- Parent product-core SHA-256:
  `6e7f9701044ee515cf08d9f65ab6cf91b8a75050e7df2c6ce44a2df9156af059`.
- Parent desktop-UI SHA-256:
  `19b1acd90830038fb6a60a3319543f1e4c782d59be49fe56de34f91d8b27819b`.
- Root is the sole writer and integration owner for U7.16A.  Producer and
  consumer tasks have released the desktop/core/shared paths.
- Existing `.codex/` and `tmp/` trees are foreign/untracked.  The feature may
  create only a unique owned child below the existing repo-relative product
  scratch root and must remove it after every successful observation.

## Frozen product behavior

1. No preview card is upscaled and no source crop is rendered independently.
2. A successful ordinary preview plus one explicit available Look selection is
   required before detail rendering.  Batch sessions inspect only the already
   bound representative and label it as such.
3. The detail point defaults to the image centre.  Clicking inside a rendered
   Look card selects that Look and records an exact normalized point in
   `[0,1]x[0,1]`; card padding is not a valid point.  Keyboard-only users retain
   the centre point through the existing Look radio buttons.
4. The current single-photo output choice (`PNG16`, `TIFF16`, or `JPEG8`) is
   reused.  A bounded helper creates one unique temporary destination, invokes
   the unchanged `ProductDesktopWorkflow.export`, validates its receipt, opens
   the exact published output, and returns one at-most `512x512` RGB8 viewer
   crop plus full-image geometry, crop bounds, output hash, recipe hash, Look,
   amount, input hash, and format identity.
5. `1:1` is a spatial statement: one displayed crop pixel corresponds to one
   exported image pixel.  PNG16/TIFF16 are converted to the existing desktop
   RGB8 viewer representation; the window must not claim sample-code identity,
   HDR, wide-gamut, or calibrated display accuracy.
6. The temporary output and recipe must match their receipt identities before
   removal.  Unknown, replaced, tampered, late-created, or extra members are
   preserved and the operation fails closed rather than claiming cleanup.
7. Input, representative, strength, Look, output format, operation failure, or
   application close invalidates and closes the old detail window.  A running
   detail render uses the existing non-daemon foreground worker and close-waits
   contract.
8. The ordinary preview images, final output bytes, strict final recipe,
   batch order/receipt, look/profile assets, renderer math, schemas, and claim
   labels do not change.

## Frozen formal roles and controls

The primary photographic role is the already-consumed exact U7.12C portrait
JPEG `a4593-kme_0276.jpeg`, 4032x6048, SHA-256
`7ea3d1ed37b7df518c0466c1379f08aa34ea5ae00eb7dc8208fcff2cfe68d1c6`.
It is a product/runtime fixture, not film truth.  The primary arm uses Ektar
100 Look Approximation, visible amount `0.65`, PNG16, point `(0.73,0.31)`,
crop limit `512`, tile size `256`, one worker, and the unchanged product
profile/assets.  One helper render and one ordinary final export must have the
same decoded dimensions and full output SHA-256; the returned crop must equal
an independent crop from the final output at the frozen bounds.

Small deterministic RGB fixtures cover centre/edge/corner point clamping,
portrait/landscape/smaller-than-limit geometry, all three output formats,
duplicate basenames, batch representative identity, and the following rejects:
no preview, unavailable Look, bool/nonfinite/out-of-range point, zero/negative/
bool crop size, missing or mismatched suffix, changed source/session, foreign
temporary member, replaced output or recipe, and cleanup ownership drift.

## Frozen success gates

1. The parent demonstrably has no detail action or `1:1` output inspection.
2. The feature never derives a detail crop from the `300x260` preview bytes and
   never invokes a crop-only look render.
3. The primary temporary and ordinary final exports have exact full output
   SHA-256, dimensions, Look, amount, input identity, and output-format parity.
4. The primary returned RGB8 crop and bounds equal the independent final-output
   oracle exactly, including spatial one-pixel correspondence.
5. All three output formats pass deterministic small-fixture crop parity; the
   crop is at most `512x512` and never upscales a smaller image.
6. Default centre and explicit card-point selection are exact; padding clicks
   reject, same-name labels reveal no parent path, and the batch representative
   remains the inspected source without changing canonical output order.
7. Changing input, representative, amount, Look or output format closes the
   old detail authority before another render or export.
8. Temporary output/recipe cleanup is identity-bound, unknown/tampered members
   are preserved, successful residue is zero, and source files are immutable.
9. U7.10A, U7.11A, U7.12B/C/G, U7.14A, U7.15A/B/C behavioral suites pass in
   isolated fresh processes.  Historical evidence remains immutable.
10. Forward/reverse committed-head scientific payloads are exact; Ruff,
    formatting, compile, JSON, diff and tracked-clean gates pass.

## Stop rule and claim ceiling

Any formal gate failure closes this exact mechanism.  Do not rescue by
upscaling low-resolution previews, lowering the crop size, changing the point,
rendering the crop independently, switching format, changing the source/look/
amount/tile parameters, tolerating output drift, retaining temp exports as a
cache, or weakening cleanup/identity gates.  A failure does not open automatic
face/text detection, aesthetic point selection, zoom pyramids, history,
side-by-side widgets, or another preview family.

A pass establishes only private Windows/Python native-desktop spatial detail
inspection for the existing deterministic `film-inspired / Look Approximation`
product chain.  It is not calibrated stock response, physical-film
reproduction, stock distinguishability, severe-artifact population evidence,
sample-code/display calibration, HDR/wide-gamut support, arbitrary-media
quality, public release, installer, or cross-platform GUI evidence.  AO6
remains only the Velvia 50 display-proxy Look Approximation baseline.

## Execution and rollback

1. Commit this contract/config before implementation or primary-role rendering.
2. Commit the bounded helper, UI integration and focused tests separately.
3. Commit the formal controller before clean committed-head execution.
4. Run forward/reverse formal reports, isolated parent regressions, static
   checks, source immutability and residue checks.
5. Commit evidence and binding test, then minimally propagate tracker, README
   only if user-facing behavior needs it, and AGENT_LOG.  Do not push.

