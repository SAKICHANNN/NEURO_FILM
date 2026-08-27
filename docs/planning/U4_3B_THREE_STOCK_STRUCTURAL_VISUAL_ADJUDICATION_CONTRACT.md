# U4.3B — Three-stock structural visual adjudication contract

## Question

Do the U4.3A queue leaders show confirmed unintended structural artifacts at
the frozen RF3.D0 raster resolution, or are their diagnostic tails visually
consistent with acceptable strong colour stylization?

## Frozen inputs and selection

- Bind the exact U4.3A report and its scientific identity.
- Select the union of the first three rows from each of its seven independent
  queues before any image review. This produces 11 source/arm pairs; no row may
  be added, removed or substituted after inspection.
- Reuse only the exact source and output PNG bytes already bound by U4.3A.
  Rendering, network access, fitting and new data acquisition are forbidden.

## Review material

For each selected pair, produce one deterministic review sheet containing:

1. source and output fit-to-panel views for scene context;
2. exact-pixel crops around the strongest flat-region new-high-frequency
   location, edge-direction disagreement and edge-gain tail;
3. the same crop coordinates for source and output, without enhancement in the
   decision panels;
4. optional labelled diagnostic maps that may guide attention but may never be
   used as visual evidence by themselves.

Review sheets are ignored artifacts. Their hashes and the underlying source and
output hashes are recorded so the adjudication remains replayable.

## Decision rule

A pair is a confirmed severe failure only when original-pixel source/output
comparison shows an unintended visible defect such as banding, colour blocks,
posterization, isolated chroma speckles, objectionable halos/ringing, or loss of
recognizable content structure. Strong saturation, contrast, intended colour
separation, ordinary sensor noise, or a large diagnostic value alone is not a
failure. Ambiguous rows remain `REVIEW_UNRESOLVED`.

The leaf may close only when every frozen row has one adjudication and a short
visible rationale. It must report the rows individually; no scalar score or
majority vote is permitted.

## Claim ceiling

Autonomous visual evidence on 11 preselected RF3.D0 raster pairs only. It is not
population preference, stock identification, calibrated film response, a
universal safety proof, multi-stock completion or product promotion.
