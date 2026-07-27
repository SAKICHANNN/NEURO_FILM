# U5.R2W2F0 FilmSet Recipe Global-Explainability Contract

Date: 2026-07-27

Status: **preregistered implementation frozen; formal run pending**

## Parent and question

U5.R2W1 repeats exactly with branch `paired_upper_bound_only_passes`.
Output-only fixed reference recovery is closed, but aligned neutral/styled
pairs show that the bounded O0 family has enough synthetic headroom.

W2F0 asks only:

> Is each FilmSet Capture One recipe coherent enough across held-out contents
> to be described by one shared bounded global colour operator?

`Cinema`, `ClassNeg` and `Velvia` are recipe-domain names. They are not
physical stocks, processes, scanners or calibrated responses.

## Frozen pixel boundary

The existing metadata preflight partitions 465 internal identities into
256 development, 128 confirmatory and 81 stress identities using
`sha256(seed:pool:content_id)` with seed `2026072701`.

W2F0 reads only:

- the first 24 development identities;
- the first 16 confirmatory identities;
- input plus the three aligned recipe outputs for those identities.

All remaining internal identities, target-only/source-only pools and the
final 628 stay unread. Every payload is hash-checked and decoded through the
existing `WorkingImage` path. Any path escape, hash mismatch, alignment
mismatch, unsupported colour state, ICC failure or non-finite value fails
closed.

Each aligned image uses a deterministic `4 x 4` spatial grid with 32 unique
fit and 32 unique evaluation pixels per cell. Coordinates are identical
across input and recipe outputs. Fit and evaluation seeds are separate.
No full-raster output or visual shortlist is generated.

## Frozen operators

Per recipe compare:

1. identity;
2. one shared joint exposure/WB/contrast/saturation basic operator fitted on
   development pairs;
3. one shared `4 x 4 x 4` bounded O0 flow fitted on development pairs;
4. one per-pair O0 evaluator Oracle for each confirmatory aligned pair.

The per-pair operator is never a product input. It measures how much of the
recipe is globally explainable per image and how far one shared operator
lags behind image-specific global fits.

All O0 fits use 16 integration steps, coefficient-vector norm cap 2.0,
80 deterministic float32 Adam steps and the local CUDA device when available.
The final renderer remains the explicit O0 flow.

## Frozen evidence and gates

Report per recipe:

- confirmatory linear-RGB RMSE for identity, shared basic, shared O0 and
  per-pair O0;
- shared improvement over identity and basic;
- shared normalized regret to the per-pair Oracle;
- per-pair velocity-grid dispersion around the shared O0;
- `4 x 4` between-cell residual fraction after shared O0;
- output range, Jacobian determinant/norm, inverse, coefficient and replay
  evidence;
- exact report repeat.

The branch gates are frozen in
`configs/u5_r2w2f0_filmset_recipe_global_explainability_v1.json`.

- `global_operator_coherent`: shared O0 improves identity by at least 25% and
  basic by at least 5%, normalized Oracle regret is at most `.15`, grid
  dispersion at most `.08`, spatial residual fraction at most `.25`, and all
  structural gates pass.
- `basic_only`: basic improves identity by at least 25%, while shared O0 adds
  no more than 5%.
- `adaptive_or_spatial_recipe`: per-pair O0 is informative but shared regret,
  grid dispersion or spatial residual exceeds its gate.
- `operator_unidentified`: even per-pair O0 fails to improve identity by 25%
  or evidence otherwise does not support a global interpretation.
- invalid colour/alignment or structural/repeat failure closes the affected
  route.

These are engineering development gates, not population or calibration
claims. Confirmatory metrics may not be used to retune them.

## Branch consequences

W2F1 output-only reference recovery remains closed regardless of W2F0 because
W1's practical information regime failed.

- coherent recipe: retain a recipe-global explicit champion as mechanism
  evidence;
- basic-only recipe: retain basic control, close CFSM complexity;
- adaptive/spatial recipe: record global insufficiency; do not add a local
  model without a new preregistered residual hypothesis;
- unidentified or invalid: close without capacity rescue.

No result opens current real-film pixels, stock learning, latent modes,
calibration, production integration or authenticity claims.
