# U5.R2W2F FilmSet Reference-Look Bridge

Date: 2026-07-27

Parent: `ULT > U5.R2 > U5.R2W0/W1`

Status: **W2F0 paired/global explainability ready; W2F1 output-only closed**

## Purpose

Test whether an output-only reference-look mechanism that survives generated
known-operator controls transfers to aligned real rasters produced by repeated
Capture One recipes.

This is a method bridge, not film evidence. `Cinema`, `ClassNeg` and `Velvia`
are FilmSet recipe-domain names. They are not physical film stocks, measured
emulsions, processes, scanners or calibrated responses.

## Existing evidence and capability boundary

The passed FilmSet evidence freeze exposes:

| Pool | Identities | Allowed W2F role |
|---|---:|---|
| `source_train` | 2,096 input-only | unpaired source-distribution control |
| `target_train` | 2,096 x three recipe outputs | output-only reference bank |
| `internal_dev_lockbox` | 465 x input plus three targets | evaluator-only development/held-out split |
| `final_628_lockbox` | 628 x input plus three targets | forbidden to W2F |

The manifests have zero content-ID, duplicate-cluster, exact, dHash or frozen
embedding leakage across source, target and internal-development pools. Their
frozen SHA-256 values are:

- source: `4788ecffcc0d05df9baf614e159fd5e2bd6f5e5ca76e2bd3109ed993aca6f865`;
- target: `740019507339bb156a7b73061571209145447a3774f1c5319dbc4f9555c0db4f`;
- internal development:
  `fc78bb22e9a36972c405f52fd22d7b505fd179155af294c3718a5b693b61b0a2`;
- final 628:
  `7bba03a6f9ae4a70c4f0f128bf12b14116918ac7898dc656f324113110d9b0b2`.

W2F must not decode, select on, refit from or re-adjudicate the final 628.
Its historical CT8 result remains frozen and is not a fresh confirmation set.

The metadata-only commit-bound preflight ran twice byte-identically on
software commit `77df1b9004aa0d155ce1b662202db476ec5f677e` at SHA-256
`AB435E2433373D9FB001C13D20AF7B408F0346D1D1F2CF2F989DB080C86103C3`.
It confirms zero content-ID and duplicate-cluster intersection and freezes:

- 512 target-only reference-bank identities plus 1,584 reserve;
- 256 source-only probe identities plus 1,840 reserve;
- 256 internal development identities;
- 128 internal confirmatory identities;
- 81 internal stress identities.

The final manifest was hash-checked but not parsed. Image payload read/decode
counts are both zero. The repeated W1 decision is now
`paired_upper_bound_only_passes`. This opens W2F0 paired/global recipe
explainability only. It does not open W2F1 output-only reference recovery.

## Why this source precedes RTD

FilmSet is already local, already integrity-frozen and already allowed for
internal recipe research. It supplies the exact output-only topology needed by
CFSM without accepting a new click-through agreement or disclosing contact
information.

INRetouch RTD remains the stronger recipe-diversity benchmark, with 167
recipes instead of three, but its pixel leaf is gated and non-commercial.
W2F therefore answers the smaller mechanism question first. It cannot support
unseen-recipe generalization or replace RTD if the mechanism survives.

## DoR

Execution opens only if all of the following hold:

1. U5.R2W1 has two byte-identical formal development reports.
2. Its branch explicitly permits real-raster mechanism transfer. A failed
   fixed descriptor cannot be rescued on FilmSet with a larger encoder.
3. W1's frozen feature, operator family and information regimes are reused;
   FilmSet pixels cannot select a new descriptor.
4. Every FilmSet manifest and payload used by W2F matches the evidence freeze.
5. A hash-only internal-development/held-out partition is frozen before W2F
   pair pixels are measured.
6. Input/output colour-state and alignment validation are frozen.
7. The exact compute, sample and artifact-review budgets are recorded.

If only W1's paired upper bound passes, W2F may run the paired/global
explainability stage but not claim output-only reference recovery.

## W2F0 — recipe global-explainability

Before judging CFSM, determine what each recipe ID means in explicit-operator
space:

1. fit the same bounded explicit O0 family independently to each aligned
   input/target pair in the development partition;
2. fit one shared O0 operator per recipe using development pairs only;
3. compare identity, best basic, shared O0 and per-pair O0;
4. measure within-recipe coefficient/grid dispersion;
5. measure aligned residual spatial structure;
6. repeat on the untouched internal held-out partition without refitting.

Each recipe is classified as one of:

- `global_operator_coherent`;
- `basic_only`;
- `adaptive_or_spatial_recipe`;
- `operator_unidentified`;
- `alignment_or_colour_state_invalid`.

Only `global_operator_coherent` recipes may score output-only global CFSM.
Adaptive/spatial recipes remain a later local-residual stress class.

## W2F1 — output-only reference recovery

For every eligible recipe, construct references only from `target_train`
outputs. No matching input image for a target reference may be exposed.

Compare:

1. identity and best basic;
2. frozen recipe-global champion from W2F0;
3. W1 output-only single reference;
4. W1 symmetric multi-reference aggregation;
5. hard reference medoid;
6. within-recipe content-matched hard reference;
7. shuffled-recipe and wrong-recipe negatives;
8. paired per-pair O0 as an evaluator Oracle, never a product input.

Look/operator inference and content retrieval remain separate. Content
matching occurs only after the recipe domain is fixed and may use bounded
luma, dynamic-range, grayscale structure and colour-state proxies. It cannot
change recipe identity, enter the look descriptor or densely average recipe
operators.

All outputs use the deterministic bounded O0 renderer. No network may emit
RGB, no generated-image teacher is allowed, and no FilmSet pixel may train a
new high-capacity representation after W1 is viewed.

## Evidence and branches

Report per recipe and jointly:

- correct/wrong/shuffled operator error;
- output-only single/multi/content-matched regret to shared and per-pair
  Oracles;
- held-out content improvement over identity, basic and recipe-global
  champions;
- reference-count curve and effective-reference count;
- coefficient/grid stability;
- range, positive-Jacobian, norm, inverse, strength-zero and replay gates;
- clipping, banding, posterization, speckle, face/text/object and full-
  resolution severe-artifact evidence.

Branches:

- W2F0 incoherent: close global CFSM for that recipe; do not add capacity.
- W2F0 coherent but W2F1 loses to global: retain the global recipe champion.
- sparse content matching wins: freeze hard retrieval before any bounded
  parameter router.
- dense averaging wins only by weakening style: reject it.
- wrong/shuffled control is competitive: operator identity is unresolved.
- any severe artifact: reject the candidate regardless of style strength.
- all three recipes pass: open a separately frozen RTD decision; do not call
  the result film learning.

## DoD and claim ceiling

This leaf is complete only after repeated reports, exact configuration/code/
manifest hashes, frozen group splits, structural and severe-artifact evidence,
branch propagation and scoped commit/push.

Maximum claim:

> output-only reference-look recovery on three repeated Capture One recipe
> domains under an internal research-only, content-disjoint FilmSet protocol.

Forbidden claims include real film, Velvia stock response, unpaired
digital-to-film identification, professional preference, calibration,
commercial rights, unseen-recipe generalization and production readiness.
