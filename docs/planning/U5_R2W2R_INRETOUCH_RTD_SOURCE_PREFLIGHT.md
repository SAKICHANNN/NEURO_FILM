# U5.R2W2R INRetouch RTD Source Preflight

Date: 2026-07-27

Status: **conditional source candidate; no gated data accessed**

Parent: `ULT > U5 > U5.R2 > U5.R2W0/W1D`

Decision record:
`configs/u5_r2w2r_inretouch_rtd_source_preflight_decision_v1.json`

## Question

If the generated W1 experiment finds a useful bounded reference-look
mechanism, can a rights-isolated, real-raster same-preset benchmark determine
whether that mechanism survives real image content, Lightroom rendering and
compression?

This node does not ask whether Lightroom presets are film, whether the presets
are aesthetically good or whether an unpaired final photograph identifies its
true grading operator.

## Observed public facts

The official INRetouch project and dataset card report:

- 569 selected MIT-Adobe FiveK contents;
- 167 Lightroom presets applied across those contents;
- 508 development and 61 benchmark contents;
- 145 development and 22 benchmark presets;
- paired `natural` and preset-rendered directories;
- about 100,000 images / 18 GB;
- CC BY-NC-SA 4.0 and gated access requiring contact-information disclosure.

The later NTIRE 2026 challenge adds a useful, independently described
evaluation topology: 18 unseen development presets, 12 unseen automatic-test
presets and a separate subjective set. It requires a before/after reference
pair, not only a final styled reference. Its most relevant retrieval
submission represents an aligned pair by a 57-dimensional RGB-residual
descriptor, retrieves preset-specific initializations and then performs
test-time fitting. This is evidence for the paired-reference upper-bound lane,
not permission to weaken W1's output-only claim or use direct-RGB INRs.

A read-only Hugging Face API audit at repository revision
`3e100e1fa896d9ed023cd1545890400edd67f949` lists 96,164 files:

| Partition | Files |
|---|---:|
| Train | 74,168 |
| Validation | 8,906 |
| Benchmark/Test | 1,403 |
| Benchmark/Test_References | 11,684 |

The same public path names were joined to the existing local FiveK
`freeze_v1` manifest without opening RTD pixels:

| Identity set | Count |
|---|---:|
| RTD natural content IDs | 569 |
| local `freeze_v1` identities | 128 |
| exact `source_name` overlap | 15 |
| overlap in RTD Train | 13 |
| overlap in RTD Validation | 2 |
| overlap in RTD Benchmark/Test | 2 |
| overlap in RTD Benchmark/Test_References | 13 |

This establishes an identity-key bridge only. The local freeze contains its
own RAW/Expert-C/project-derived representations; none is an RTD preset render
or a substitute for the absent benchmark pixels.

The role counts deliberately overlap: RTD reuses the 508 Train contents as
Test_References for held-out presets, and the 61 Validation contents as
Benchmark/Test inputs. The join has 15 unique identities, not 30.

The public repository-API snapshot is retained under ignored source
reconnaissance storage at 6,293,126 bytes / SHA-256
`3ACFAC6F57EC469E04670DEE500E64050D51FF401D3E43998FEDA7AA1A407E2F`.
It contains no gated image payload.

The offline topology audit checks more than aggregate file arithmetic. For
every partition and every preset, its exact content-ID set must equal that
partition's natural-image content-ID set. This prevents a missing natural
identity plus an unrelated extra identity from cancelling in the total count.
All 334 partition/preset matrices pass, and the two reports are byte-identical
at SHA-256
`9F9C0935D2F20EB0AF20E386B4F6D29A0724EBA876A03FA6E703DCF0FDB8DC35`.

The public method repository was inspected at commit
`cbf0db19487222c21d63116c320357764438360d`. Its coordinate/RGB-conditioned
INR directly predicts final RGB and is not an admissible project renderer.
Dataset feasibility and method admissibility are separate decisions.

## Evidence and claim class

Allowed future label:

`real-raster/same-known-preset/reference-look-control`

Forbidden labels:

- real film or film stock;
- calibrated grading;
- professional per-image edit;
- identified digital-to-film operator;
- commercial training or production-weight clearance.

The same preset rendered over many contents gives operator-group supervision.
It is stronger than unpaired final-image marginals but remains a software
preset control. The paper explicitly reports that one preset can transform
different contents differently. `preset_id` is therefore a shared recipe
label, not presumed global-operator truth.

## Definition of Ready

Every item must pass before gated access:

1. W1 has an exact repeated decision and identifies the smallest surviving
   method or closes all practical methods.
2. A human explicitly accepts the provider's contact disclosure and
   click-through terms. General download authorization does not substitute for
   this external identity disclosure.
3. The project records a non-commercial/share-alike isolation decision that
   prevents RTD pixels, derived assets or weights from entering a future
   commercial/released lane.
4. The MIT-Adobe FiveK origin and the project's partial FiveK freeze are
   reconciled by source ID. The current public-name join covers 15 identities
   across paired RTD roles, including two Benchmark/Test identities; no local
   partial asset is silently treated as the complete source or an RTD render.
5. A byte budget, target partition and exact provider revision are frozen.
6. Development/confirmation methods, thresholds and group splits are
   preregistered before pixels are decoded.
7. A global-explainability pilot is frozen before any preset is admitted as a
   shared global look.

If any item fails, this node remains `not_ready`; it does not justify a larger
model or a licence assumption.

## Bounded acquisition order

After DoR, acquire the smallest useful official partition:

1. licence/card snapshot and exact repository tree;
2. `Benchmark/references_file.txt`;
3. only `Benchmark/Test/natural` and the 22 benchmark preset outputs;
4. only the fixed reference rows named by the benchmark file;
5. expand to additional `Test_References` rows only if the frozen
   multi-reference regime requires them;
6. do not acquire Train/Validation or the complete 18 GB corpus unless a
   later evidence-backed child explicitly needs them.

For every retained file record:

- provider revision and path;
- byte count and SHA-256;
- FiveK content ID;
- preset ID;
- natural/styled role;
- partition;
- derivation and duplicate group;
- licence snapshot/date;
- allowed-use and redistribution prohibition.

Exact/perceptual duplicates and source-content overlap across evaluation roles
must be zero after grouping.

## Candidate evaluation

Only methods frozen by the W1 decision may enter. Minimum controls are:

1. identity;
2. global mean operator;
3. raw histogram/statistics negative;
4. smallest surviving W1 single-reference method;
5. the corresponding multi-reference method;
6. paired natural/styled explicit O0 fit as an information upper bound;
7. shuffled preset IDs and content-matched wrong-preset negatives.

The released INRetouch direct-RGB INR is a comparison only and cannot become a
Style-safe candidate.

Before that comparison, each preset needs a truth-class audit:

1. fit a bounded global O0 separately to each aligned natural/styled pair;
2. measure within-preset dispersion of the per-image operators;
3. fit a shared preset operator on development contents and evaluate held-out
   contents;
4. compare global residual with an identity/basic baseline;
5. localize aligned residual energy without fitting a local model.

The outcome for each preset is one of:

- `global_operator_coherent`;
- `basic_only`;
- `adaptive_or_spatial_recipe`;
- `operator_unidentified`;
- `alignment_or_colour_state_invalid`.

Only the first class may score global CFSM operator recovery. The adaptive
class remains a later local-residual stress set; it does not justify a direct
RGB model or count as failure of a global method.

Metrics and numeric promotion thresholds are intentionally not frozen here:
they depend on the W1 measurement scale and must be preregistered by a child
after W1, before RTD pixels are opened. Required categories are:

- held-out-preset explicit-operator error;
- held-out-content same-look consistency;
- correct versus shuffled preset;
- single versus multiple references;
- identity false positive;
- strength/style retention where a legal strength path exists;
- range, Jacobian, norm, inverse and replay;
- full-resolution severe artifact and clipping;
- content/source/nuisance probe.

## Branches

- W1 practical methods all fail: keep RTD as a future benchmark; do not
  download it to rescue the representation.
- W1 only paired upper bound passes: RTD may test paired explicit fitting but
  cannot reopen output-only reference matching.
- W1 multi-reference only passes: acquire only the references needed by the
  frozen multi-reference contract.
- W1 single-reference passes: compare single and multi regimes without
  changing the method after RTD is viewed.
- content/shuffled/source control fails: close real-raster transfer without
  capacity rescue.
- preset global-explainability fails: reclassify it as adaptive/local; do not
  average its per-content operators into false global truth.
- structural or severe-artifact gate fails: reject the candidate regardless
  of style strength.
- all gates pass: retain `reference-look` evidence only; a separate
  rights-cleared film/stock programme is still required.

## Definition of Done

This source-preflight node is complete when the public topology, access
boundary, DoR, bounded acquisition sequence, evidence class and branch rules
are recorded. It does not become an executable pixel leaf until all DoR items
pass.
