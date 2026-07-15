# RF1.1 FILM-R signal/content/nuisance audit contract

## Purpose

Before fitting any film-look expert, test whether FILM-R can separate a reusable
colour signal from content, damage and restoration nuisance. The filename
families remain hints, not authoritative stock labels.

All 44 original/restored pairs were visually reviewed before this contract.
The broad content label for every pair is frozen in
`configs/real_film_filmr_signal_audit.json` and bound to both review-sheet
hashes. Categories deliberately remain coarse: product/equipment, urban/
architecture, landscape/nature and other object/interior.

## Stage-zero identifiability gate

The family-by-content support matrix is evaluated before new feature pixels are
decoded. A family cell is supported only with at least two independent pairs.
An eligible family needs at least three total pairs and two supported content
cells. Pairwise family comparison needs at least one shared supported content
cell. At least three mutually comparable families are required for a
multiclass signal test.

If these conditions fail, family colour cannot be separated from content in
this dataset. RF1.1 stops at `structurally_unidentified`; a high random-split
classifier score is forbidden as evidence.

## Conditional feature audit

Only if stage zero passes, compare four fixed descriptors:

1. colour-only Lab distribution statistics;
2. grayscale structure/content features without colour;
3. damage-only paired restoration-change statistics;
4. their combination.

Original/restored siblings stay in one group. Evaluation uses pair-grouped
leave-one-out plus leave-one-content-category-out, fixed nearest-centroid and
regularized linear classifiers, content-stratified permutations, and
original-to-restored perturbation checks. No architecture or hyperparameter is
selected on held-out results.

Promotion requires colour-only signal to survive content and damage controls.
If structure-only or damage-only is as strong, or leave-one-content performance
collapses, FILM-R remains a real-film style/artifact stress set rather than a
training source for family experts.

## Claim boundary

Even a pass establishes only exploratory FILM-R filename-family signal. It
cannot establish a named stock response, physical-roll generalization,
scanner independence, calibrated authenticity or digital-to-film supervision.
