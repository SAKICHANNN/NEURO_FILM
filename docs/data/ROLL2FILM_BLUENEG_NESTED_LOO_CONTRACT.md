# RF1.2 BlueNeg nested leave-one-frame-out diagnostic contract

## Question

The frozen two-roll confirmatory test changed sign: `19960817H` lost by
0.2201 Delta-E while `19970620C` gained 0.3095. RF1.2 asks whether that
heterogeneity is consistent with physical-roll information, content retrieval,
archive-wide restoration/scanner behaviour, arbitrary grouping, or a small
number of frames. It is a retrospective diagnosis, not a second confirmatory
attempt.

No model family or capacity is added. The only fitted operator remains the
previously selected explicit Lab mean/std transform.

## Frozen cohort and access

Use only the four Kodak Gold 100-5 rolls already decoded by CT6. There are 43
locally complete aligned preview/pseudo-GT frames: 13, 6, 12 and 12 in
`19960816G`, `19960816H`, `19960817H` and `19970620C`. The extra source-only
frame in `19970620C` is excluded because it cannot be an aligned query.

Every complete frame is a query exactly once. Its pixels and label are excluded
from every support arm in that fold. Three support frames and equal per-frame
pixel budgets are used in every learned arm. Source and target support pixels
remain independently sampled, so aligned correspondence is never exposed.

All upstream metadata, acquisition, transformation and CT6 result hashes are
bound in `configs/roll2film_blueneg_nested_loo.json`.

## Fixed arms

For each query:

1. `correct_roll`: three non-query frames from the same physical roll.
2. `wrong_roll_<id>`: three frames from each other physical roll separately.
3. `pooled_wrong`: three frames from the union of all other rolls; this is the
   archive/restoration-common fixed-budget control.
4. `content_retrieval_wrong`: three other-roll frames nearest in a frozen
   source-only global descriptor.
5. `shuffled_roll`: a fixed count-preserving permutation of roll labels,
   followed by the same three-frame selection.
6. `identity`.

All hash selections are deterministic. The content descriptor uses no target
pixels. It consists of linear-RGB channel means and standard deviations, Lab L
10/50/90 percentiles, and Lab chroma mean/std, robustly standardized over the
43 source frames.

## Matched-style control

Raw comparison can reward an operator simply for being weak. Therefore every
arm is also evaluated after a source-only scalar style match. Along the vector
`input + alpha * (candidate - input)`, alpha is selected in `[0, 1.5]` so the
candidate's mean Delta-E from the query input matches the correct-roll output.
The hidden query target is forbidden during this strength selection.

## Statistics and interpretation

Report query-level target Delta-E, style strength, each physical-roll mean,
each wrong-roll comparison, pooled/retrieval/shuffle comparisons, and
associations with source-descriptor distance, location and scene property.
Bootstrap queries within physical roll and physical rolls as the outer cluster;
with only four rolls, the interval is descriptive and cannot support a broad
claim.

The group-information hypothesis is worth reopening only if correct-roll gain
is positive on all four rolls and survives every fixed control after style
matching. A retrieval/pooled/shuffle win is nuisance evidence. Mixed roll signs
or a cluster interval containing zero remains `heterogeneous_or_unidentified`.
Regardless of outcome, RF1.2 cannot promote named-stock, calibration,
digital-to-film or new confirmatory claims.
