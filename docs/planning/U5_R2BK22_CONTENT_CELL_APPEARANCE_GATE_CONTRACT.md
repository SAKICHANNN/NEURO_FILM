# U5.R2BK22 Content-Cell Appearance Gate Contract

Date: 2026-07-31  
Status: frozen before implementation

## Purpose

BK21 showed that two colour-distribution objectives can agree while jointly
following a content shift. BK22 does not add a loss, flow coefficient or third
canonicalizer. It tests one new authorization primitive: worst-case validation
across content cells whose identity is supplied independently of colour.

## Exact counterfactual

Four generated latent scene cells each have a distinct colour population. One
bounded shared look is applied to independent target samples.

- Valid view: source cell `i` is evaluated against styled target cell `i`.
- Confounded view: the same four target arrays are permuted by `[1,2,3,0]`.

The pooled source bytes and pooled target bytes are identical between views.
Only the externally supplied cell correspondence changes. Consequently any
pooled histogram, MMD, SWD or whole-reference appearance score sees the same
evidence.

## Fixed policy

The two BK21 canonicalizers fit one shared O0 operator each on pooled fit
cells. Pooled validation hard-selects one candidate; it is authorized only if:

- the two operators disagree by no more than `0.06` RGB RMSE;
- pooled cross-objective improvement is at least 25%;
- every content cell improves by at least 10% under the mean of both frozen
  objectives;
- existing range, Jacobian, inverse and replay gates pass.

Otherwise the output is exact identity. No per-cell expert and no blending is
allowed.

The valid correspondence must authorize one hard candidate. The byte-identical
pooled but shuffled correspondence must return identity. Any failure closes
this policy without threshold or capacity rescue.

## Epistemic boundary

Generated cell identity is an evaluator-only oracle. It is not inferred from
colour or content embeddings. A synthetic pass would show only that such
external connectivity is useful when it exists. It would not establish that
current film data supplies valid cells, identify an operator, or open stock,
mode, training or product claims.
