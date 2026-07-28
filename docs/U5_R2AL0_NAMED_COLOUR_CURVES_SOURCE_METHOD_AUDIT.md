# U5.R2AL0 Named-Colour Curves Source and Method Audit

Date: 2026-07-28

## Decision

Retain one narrow representation hypothesis:

> A fixed, clean-room, analytic partition of colour space may blend a small
> bank of endpoint-fixed monotone RGB curves into a continuous, deterministic
> explicit colour operator.

Do not reuse or execute NamedCurves/NamedCurves+ code, checkpoints, learned
colour-naming assets, canonicalizers, attention or transformer fusion. Do not
use MIT-Adobe FiveK or PPR10K as film evidence. The next leaf may ask only a
frozen synthetic capacity and structural-safety question.

This distinction matters. The potentially useful idea is colour-selective
curve capacity. The published systems are paired expert-retouching networks,
not identified film operators, and their learned spatial components are
incompatible with the project's deterministic Style-safe renderer boundary.

## Primary-source freeze

The audit used the following current primary sources:

- [ECCV 2024 official landing page](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8963_ECCV_2024_paper.php);
- [ECCV 2024 paper PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08963.pdf),
  4,900,159 bytes, SHA-256
  `5fdada4bf7434eea8b46ea0240ada5918c1e5900f491feaae306216665076254`;
- [NamedCurves+ arXiv record](https://arxiv.org/abs/2607.08185), submitted
  2026-07-09, linked to TPAMI DOI `10.1109/TPAMI.2026.3690637`; its paper
  record links CC BY 4.0;
- [official NamedCurves repository](https://github.com/davidserra9/NamedCurves)
  at commit `35af8380137807d83881e9198b35b567af2c96c1`.

The repository audit found no root `LICENSE`, `COPYING` or `NOTICE`, and the
GitHub repository API reports no detected SPDX licence. Its tree contains the
learned `models/joost_color_naming.mat` asset. The README says pretrained
models are released, while NamedCurves+ code remains a TODO. Public paper
licensing does not license the repository code, checkpoints or third-party
colour-naming asset. All of those artifacts are therefore forbidden inputs.

No repository clone, checkpoint, model asset, training image or inference
output was downloaded. The only retained external artifact is the ignored
paper PDF used for source verification.

## What the published methods actually do

NamedCurves first sends an input through a UNet-like backbone to a learned
canonical latent image. It then applies a learned 11-name colour classifier and
groups its probability maps into six branches:

1. orange/brown/yellow;
2. achromatic white/grey/black;
3. pink/purple;
4. red;
5. green;
6. blue.

Each branch predicts three endpoint-fixed Bezier curves, one per RGB channel.
Positive control-point increments are normalized to make the curves
monotonically increasing and to place the last endpoint at `(1, 1)`. Six
globally adjusted images are then combined through learned spatial attention.
The original paper thresholds colour-name probabilities below `0.2` before
renormalizing them.

NamedCurves+ retains the learned backbone, colour-name decomposition and
Bezier curves, but replaces the fusion stage with a transformer. Its paper
explicitly reports that the earlier design can create halos and inconsistencies
where colour-name probabilities overlap or change abruptly.

Both systems learn from paired aesthetic-retouching datasets. NamedCurves uses
MIT-Adobe FiveK and PPR10K; NamedCurves+ adds exposure and tone-mapping
benchmarks. None provides `film_stock_id`, roll, process, scanner or a
digital-to-film operator.

## What does not transfer

### Canonicalization is not neutral evidence

The learned backbone is the largest contributor in the reported ablations and
changes RGB before curve selection. It can absorb camera, illumination,
content and editing style. It is therefore neither an identified neutralizer
nor a safe source of stock supervision.

### Learned colour names and spatial fusion reintroduce shortcuts

The learned probability maps, attention and transformer can use content and
spatial context. That violates the current no-content-shortcut boundary and
would allow the model to change final RGB spatially. A hard probability
threshold is also discontinuous and conflicts with the project's banding,
seam and full-resolution artifact discipline.

### Monotone component curves do not prove a safe 3D map

Each one-dimensional curve can be monotone while input-dependent blending
still creates a negative or excessive 3D Jacobian through derivatives of the
colour weights. Endpoint-fixed curves and convex RGB output prevent simple
range overflow, but they do not prove orientation, inverse stability or
absence of boundary artifacts. Those properties require independent project
gates.

### Paired retouching is not film learning

FiveK and PPR10K establish performance against particular human retouches.
They do not establish film appearance, a reusable stock response, latent stock
modes or unpaired operator identification. Current project film pixels remain
closed for fitting and training.

## Clean-room representation hypothesis

The allowed successor is deliberately not an implementation of the paper:

- use the project's existing D65 colour math, not the learned naming asset;
- define one analytic achromatic weight plus five smooth chromatic hue-sector
  weights whose sum is exactly one;
- make chromatic weight vanish smoothly on the neutral axis;
- use one shared achromatic curve to preserve exact neutral RGB;
- use fixed endpoint monotone curves with positive normalized increments;
- blend only the resulting explicit curve outputs;
- use continuous strength and no hard mask threshold;
- keep all partition constants fixed by the contract, not learned from images;
- allow a future ML model, only after separate evidence gates, to predict
  bounded curve parameters rather than pixels.

The neutral-axis, partition-of-unity, range, continuity, Jacobian, inverse,
serialization, tiled/full and strength-path checks must be tested directly.
The `53/55/56` near-collinear strength path and ID11 red
speckle/posterization observation are mandatory negative controls before any
photographic frontier.

## AL1 definition of ready

AL1 may open only after a separate committed contract freezes:

1. analytic colour space, sector centres, bandwidth and chroma gate;
2. exact curve count, control count, positive-increment floor and parameter
   budget;
3. two or more synthetic colour-selective truths not derived by fitting
   photographs;
4. global-curve and existing explicit-operator controls;
5. development/confirmation split and deterministic optimizer;
6. identity, neutral, range, continuity, Jacobian, inverse, replay, partition
   and relative-capacity gates;
7. zero photograph, film-pixel, external-code, checkpoint and asset access;
8. the branch rule that failure closes the fixed representation without
   capacity or threshold rescue.

No new data is required for AL1. A synthetic pass would only establish
representation capacity and safety. It would not authorize film fitting,
training, photographic rendering, stock claims or product integration.

## Claim ceiling

`primary-source method/licence audit and clean-room analytic
colour-selective-curve hypothesis only`.

No external implementation, learned asset, dataset result, film operator,
stock calibration, preference, photographic-safety or production claim is
made.
