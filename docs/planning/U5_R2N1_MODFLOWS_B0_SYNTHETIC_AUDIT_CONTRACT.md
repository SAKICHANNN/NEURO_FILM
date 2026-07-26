# U5.R2N1 — ModFlows B0 Synthetic Audit Contract

**Status:** frozen before checkpoint download, deserialization or output
inspection

**DRPT level:** L2

**Parent:** U5.R2N0 external source/method feasibility

**Primary writer:** current Codex Goal session

## Question

Can the pinned small ModFlows checkpoint be interpreted from the published
architecture without copying unlicensed source, produce a repeatable
palette-sensitive rather than geometry-sensitive embedding, and drive a
well-behaved explicit RGB ODE on synthetic colour data?

N1 is a checkpoint and identifiability audit. It does not render real images
or film references.

## Frozen asset boundary

Only
`modflows_color_encoder_B0_dim_515.pt` may be downloaded:

- exact revision `f7b70779...83635`;
- exact bytes `18,970,914`;
- exact SHA-256 `124f7b42...d3eacb`;
- destination `data/external/modflows_b0_v1/`;
- deserialization must use `weights_only=True`.

B6, datasets and official source are forbidden. The GitHub repository has no
licence and cannot be copied to solve incompatibilities.

## Frozen clean-room interpretation

The encoder is torchvision EfficientNet-B0 with a 515-output final linear
layer and standard ImageNet normalization at `224x224`.

The 515-vector is parsed, in standard module parameter order, as:

- `64x4` first-layer weights over `(R,G,B,t)`;
- 64 first-layer biases;
- `3x64` second-layer weights;
- three second-layer biases.

Velocity is `W2*tanh(W1*[RGB,t]+b1)+b2`. A fixed eight-step RK4 leg transports
content from `t=0` to `1`; an eight-step reverse leg uses the reference
parameters from `1` to `0`. No clamp is allowed.

This interpretation is frozen before checkpoint inspection. Failure cannot be
rescued by reading/copying official code or changing parameter order, input
normalization, solver or step count.

## Synthetic controls

Eight deterministic images contain gray/rgb-cube controls plus warm and cool
palettes with:

- two different geometries using the same pixel palette;
- an exact deterministic pixel permutation preserving the histogram.

The audit measures exact repeat, cosine similarity, palette-vs-geometry
distance ratio, same-embedding forward/inverse identity, RGB-cube range,
finite-difference Jacobian determinant/norm and transfer repeat.

## Frozen gates

- checkpoint size/hash, weights-only load and exact state-dict architecture;
- all embeddings finite and repeat-exact;
- same-palette geometry and permutation cosine similarity at least `.98`;
- different-palette distance at least `2x` same-palette geometry distance;
- same-embedding identity max error at most `.005`;
- raw transfer stays in `[0,1]` without clamp;
- sampled Jacobian determinant is positive and norm at most `8`;
- transfer and two formal reports are exact.

## Branches

- **Checkpoint/architecture fail:** close; do not use unlicensed source to force
  compatibility.
- **Palette controls fail:** close palette retrieval; do not fine-tune.
- **Range/Jacobian/norm fail:** close before images; do not clamp or lower
  strength.
- **Identity fail:** close this clean-room interpretation; do not change the
  solver after inspection.
- **All pass:** open only a separately frozen A0 non-stock reference safety
  pilot.

## DoD

- exact bounded download and manifest;
- isolated clean-room encoder/flow adapter;
- synthetic control generator and property tests;
- two formal reports;
- full CPU suite;
- decision propagation, scoped commits and push.

## Claim ceiling

Clean-room checkpoint compatibility and synthetic palette/shortcut/ODE
structure evidence for one external B0 ModFlows control. No unlicensed code
reuse, real-image safety, stock identity, film operator, calibration,
preference or production claim.
