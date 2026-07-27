# U5.R2Z1 — Asymmetric Query/Operator Applicability Contract

Status: **FROZEN BEFORE IMPLEMENTATION OR FRESH CONFIRMATORY ACCESS**

Node: `ULT > U5 > U5.R2 > U5.R2Z1`

DRPT level/mode: L2 / Mode A, exactly one primary writer.

## 1. Question

U5.R2Z0 established one narrow fact inside the paired FilmSet software-recipe
control: a fixed bank of 24 ClassNeg per-case bounded O0 operators has
confirmatory Evaluator Oracle value, while both frozen input-only photometric
nearest-neighbour selectors are worse than the shared O0 operator.

Z1 asks a distinct question:

> Can a small development-only asymmetric model of query content and explicit
> operator signature select one fixed ClassNeg operator for an unseen input,
> close a useful fraction of the evaluator Oracle gap, and fall back to shared
> O0 outside development support?

This is an applicability-ranking experiment. It is not output-only reference
recovery, a generic router, a direct RGB model, a film-stock experiment or
unpaired operator identification.

## 2. Parent evidence and activation

The experiment activates only if all immutable parent checks pass:

- Z0 decision SHA-256 is
  `d7c75d12a1952d810528ef9131b9cc8523cf771b9a842b27c7bf647256a3504e`;
- both Z0 formal reports are byte-identical at
  `ffec19f4bbc5068014deef448bf272c10d279b4e869818d831a34be699d7c6dc`;
- ClassNeg branch is `case_bank_oracle_only`;
- ClassNeg structure is passing;
- Velvia remains closed and is not loaded by Z1;
- W2F1 output-only recovery remains forbidden.

Z0 gates, descriptors, results and selected identities are immutable controls.

## 3. Data boundary

Source: the already local FilmSet paired software-recipe control.

Allowed domains:

- `input`;
- `classneg`.

Forbidden domains and data:

- Velvia and Cinema targets;
- final-628 manifest payload rows;
- current community/stock pixels;
- any new user image, label, preference vote or film/digital pair.

The existing internal partition seed and ordering remain unchanged.

- Development: the same first 24 identities of the 256-identity internal
  development partition used by Z0.
- Fresh confirmatory: identities 17--32 of the frozen 128-identity internal
  confirmatory partition. Z0 used identities 1--16; Z1 must not reuse them.
- Stress and all other partition members remain unread.
- Expected selected payloads: `(24 + 16) * 2 = 80`.

Fresh confirmatory targets may be decoded only after the automatic development
gate passes. A development failure must emit a complete decision without
opening those files.

Fit and evaluation pixel coordinates remain disjoint. Coordinates are identical
across aligned input/ClassNeg rasters. No full-raster output is generated.

## 4. Fixed explicit operator bank

Z1 refits, from the frozen development fit pixels only:

- one shared ClassNeg O0 cube-diffeomorphic flow;
- 24 per-case ClassNeg O0 flows.

The O0 topology, bounds, optimiser, seeds and structural tests are copied
unchanged from Z0. A case remains eligible only if its self-improvement and all
range, Jacobian, inverse, coefficient and replay gates pass.

The bank is fixed before a fresh query is scored. The applicability model never
changes an operator and never produces RGB, curve, LUT or grid parameters.

## 5. Strictly separated representations

### 5.1 Query-content space

The query representation uses the existing input-only spatial photometric
descriptor from Z0. It contains global moments/quantiles and fixed 4x4
low-frequency cell summaries of the display-linear input.

It does not use:

- the ClassNeg target;
- operator output;
- recipe, stock, uploader or source identity;
- CLIP, semantic labels, geometry identifiers or filenames.

### 5.2 Operator-signature space

Each bank member is represented only by its explicit O0 operator:

- flatten the immutable 4x4x4x3 stationary velocity grid;
- subtract the shared O0 velocity grid;
- append, in order, the three signed per-output-channel means, the three
  per-output-channel root-mean-square values, the global root-mean-square value
  and the maximum per-grid-node vector norm of that same residual grid.

The signature contains no source-image descriptor, target pixels, content ID or
filename.

### 5.3 Development-only transforms

Query and operator spaces have independent z-score and PCA transforms. Every
normalisation and PCA basis is fit on development cases only.

For each leave-one-case development fold, transforms are refit after removing
the held-out case from both spaces. Confirmatory data never affects a mean,
scale, component, rank, threshold or tie break.

## 6. Simplest asymmetric reranker

For query vector `q` and operator vector `s`, the predicted evaluation loss is
a ridge regression over:

```text
[1, q, s, vec(q outer s)]
```

This is asymmetric because query and operator occupy different feature spaces.
It predicts one scalar loss for each fixed operator, then applies hard Top-1
argmin. It cannot average operators or directly render pixels.

Fixed capacity ladder:

1. additive/operator-only control (no `q outer s`);
2. rank-2 query PCA x rank-2 operator PCA interaction;
3. rank-4 query PCA x rank-4 operator PCA interaction.

All models use z-scored pair features, a fixed ridge penalty of `0.1`, an
unpenalised intercept and float64 closed-form fitting. The first interaction
rank that passes every development gate is selected. If rank 2 passes, rank 4
is not eligible for confirmation. If neither passes, the branch closes without
fresh confirmatory access. No neural, kernel, local, semantic or rank>4 rescue
is allowed.

## 7. Leave-one-query-and-operator-out development protocol

Let development case `j` be the held-out query.

- Remove every training pair whose query is `j`.
- Remove operator `j` from model fitting and from the candidate bank.
- Fit all transforms and ridge coefficients on the remaining off-diagonal
  query/operator loss pairs.
- Score the held-out input against the remaining 23 operators.
- Break exact score ties by stable ascending content ID.

This prevents the same paired case from serving as both query evidence and its
own candidate operator.

After a rank passes, refit its transforms/model on all 24 development cases
using only the off-diagonal 24x23 loss pairs. The resulting policy is sealed
before fresh confirmatory target access.

## 8. OOD fallback

Query support is measured in the independently standardised query descriptor
space. In each development fold, the support statistic is distance to the
nearest remaining development query.

The frozen deployment threshold is the development 95th percentile of those
leave-one-out distances. A fresh query above this threshold falls back to the
shared O0. No confirmatory-dependent confidence or coverage tuning is allowed.

The hard-reranker policy is therefore:

```text
if query support distance > frozen development threshold:
    shared O0
else:
    one hard Top-1 bank operator
```

## 9. Controls

Every development and eligible fresh-confirmatory report includes:

- shared O0;
- case-bank Evaluator Oracle;
- fixed best development operator (operator-only medoid);
- Z0 global photometric nearest neighbour;
- Z0 spatial photometric nearest neighbour;
- exact random-bank expected loss;
- 32 fixed operator-signature permutation controls;
- additive/operator-only loss model;
- rank-2 and, only if needed, rank-4 interaction models;
- OOD-aware hard policy and its fallback coverage.

Shuffle seeds and all tie rules are frozen in the config. A shuffled control
must preserve query values, actual operator outputs and loss targets while
breaking the operator-to-signature association.

## 10. Frozen gates

### 10.1 Development gate

Before fresh confirmatory access, the selected interaction rank must:

- retain at least 12 structurally eligible operators;
- have case-bank Oracle improvement over shared O0 >=10%;
- have case-bank Oracle win fraction >=75%;
- have a positive 95% case-bootstrap Oracle-improvement lower bound;
- close at least 25% of the development Oracle gap;
- improve mean RMSE over shared O0 by at least 2%;
- improve over the fixed operator medoid by at least 2%;
- improve over spatial photometric retrieval by at least 2%;
- improve over exact random expectation by at least 2%;
- improve over mean shuffled-signature policy by at least 2%;
- beat shared O0 on at least 62.5% of development queries;
- have a positive 95% case-bootstrap lower bound for mean improvement over
  shared O0;
- use shared fallback for no more than 25% of development queries;
- pass every selected-output structural and replay gate.

Failure closes Z1 without accessing fresh confirmatory targets.

### 10.2 Fresh-confirmatory gate

The fresh 16-query set first has to replicate bank value:

- Oracle improvement over shared O0 >=10%;
- Oracle win fraction >=75%;
- positive 95% case-bootstrap lower bound.

If the fresh Oracle fails, applicability is not evaluated as a promotion claim.

If the Oracle passes, the sealed OOD-aware policy must:

- close at least 25% of the fresh Oracle gap;
- improve mean RMSE over shared O0 by at least 2%;
- improve over the fixed operator medoid by at least 2%;
- improve over spatial photometric retrieval by at least 2%;
- improve over exact random expectation by at least 2%;
- improve over mean shuffled-signature policy by at least 2%;
- beat shared O0 on at least 62.5% of fresh queries;
- have a positive 95% case-bootstrap lower bound;
- fall back on no more than 25% of fresh queries;
- pass all output range, Jacobian, inverse, coefficient and exact replay gates.

Two commit-bound reports must be byte-identical.

## 11. Branches

| Branch | Meaning | Next action |
|---|---|---|
| `development_applicability_fail` | no rank<=4 passes development | close Z1; do not access fresh targets or add capacity |
| `fresh_case_bank_oracle_fail` | development passes but fresh bank value does not replicate | close case routing for this recipe control |
| `asymmetric_applicability_fail` | fresh Oracle exists but sealed selector fails | retain evaluator Oracle only; no router/visual/capacity rescue |
| `asymmetric_applicability_pass` | sealed hard policy passes every fresh gate | open only a separately frozen repeat/full-resolution safety and product-value audit |
| `structure_or_repeat_fail` | explicit operator, selected output or exact-repeat invariant fails | reject policy and diagnose engineering fault |
| `invalid` | lineage, partition or activation evidence drifts | fail closed before fitting |

A closed Z1 does not stop Ultimate. It returns the parent Goal to another
evidence-authorised explicit algorithm or product leaf.

## 12. Forbidden actions and claim ceiling

Forbidden:

- inspect fresh confirmatory targets before development gates are frozen and
  passed;
- reuse Z0 confirmatory identities as Z1 confirmation;
- select thresholds, PCA rank, ridge strength, shuffle policy or OOD coverage
  after confirmatory results;
- rank>4, kernel, neural, semantic, local or direct-RGB rescue;
- average operators or tune operator parameters per query;
- access final 628, current stock pixels or gated external datasets;
- generate a visual shortlist under this contract;
- call ClassNeg a physical film stock or the learned score a film mode;
- claim real unpaired digital-to-film operator identification.

Maximum claim:

> Repeatable paired-software-recipe evidence that a small development-only
> asymmetric query/operator loss model can or cannot recover a fixed ClassNeg
> case-bank evaluator gap on fresh unrelated contents while preserving an
> explicit hard operator and OOD fallback.
