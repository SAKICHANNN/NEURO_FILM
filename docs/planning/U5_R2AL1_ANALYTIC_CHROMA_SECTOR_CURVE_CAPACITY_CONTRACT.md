# U5.R2AL1 Analytic Chroma-Sector Curve Capacity Contract

Date frozen: 2026-07-28

## Question

Can one fixed, continuous analytic colour partition plus a compact bank of
endpoint-fixed monotone curves represent safe hue-selective colour reactions
materially better than global curves, while remaining competitive with the
existing similar-budget stationary cube flow?

This is a synthetic representation question. It is not a NamedCurves
reimplementation, an image-enhancement benchmark, a film-learning experiment
or a product candidate.

## Frozen inputs and access boundary

The only experiment input is
`configs/u5_r2al1_analytic_chroma_sector_curve_capacity_v1.json`.
Its frozen raw SHA-256 is
`53fe4134ba384d0523d5b2d87c6b0029e15c0e414dcdf5501deddcef71780559`.
Two independent processes must reconstruct all samples, targets, fits and
diagnostics from that config and produce byte-identical canonical reports.

Forbidden during execution:

- photographs or decoded image payloads;
- real-film pixels, metadata fitting or stock labels;
- NamedCurves/NamedCurves+ code, checkpoints, colour-naming assets or outputs;
- network, GPU or generative model access;
- production renderer/profile/schema changes.

The tracked worktree must be clean. A committed software identity and exact
raw config SHA-256 are mandatory run arguments.

## Fixed representation

Input is finite display-linear sRGB in `[0,1]^3`. The operator has one
achromatic branch and five chromatic branches.

### Continuous analytic partition

Direction comes from the project's D65 CIELAB `a,b` coordinates. To avoid an
undefined hue and to preserve the neutral axis exactly, chromatic activation
comes from the orthonormal linear-RGB opponent coordinates

\[
u=(2r-g-b)/\sqrt{6},\qquad v=(g-b)/\sqrt{2}.
\]

With \(C^2=u^2+v^2\) and frozen half-activation \(c_0=0.08\),

\[
q = \frac{C^2}{C^2+c_0^2}.
\]

Thus `q=0` exactly whenever `r=g=b`. Five fixed direction centres lie at
`0,72,144,216,288` degrees. Their scores use a softened normalized Lab dot
product with softening `4.0` and concentration `4.0`; a softmax produces
chromatic probabilities \(p_i\). Final weights are

\[
w_0=1-q,\qquad w_i=q\,p_i,
\]

so every weight is non-negative and the sum is one. There is no threshold,
semantic map, image statistic, learned naming model or spatial feature.

### Curve bank

Each scalar curve is a degree-six Bernstein polynomial with seven fixed input
control locations. Six positive output increments, each at least `0.04`, are
normalized to sum to one. Every curve therefore fixes `(0,0)` and `(1,1)` and
is monotone.

The achromatic branch uses one curve shared by R/G/B. Each chromatic branch
uses three curves. The bank has 16 curves, 96 raw logits and 80 effective
degrees of freedom after the per-curve softmax gauge.

For branch output \(B_i(x)\), the full-strength operator is

\[
F(x)=\sum_{i=0}^{5}w_i(x)B_i(x).
\]

User strength is only

\[
F_s(x)=(1-s)x+sF(x),\quad s\in[0,1].
\]

This makes the strength path exactly collinear and prevents `53/55/56`-like
strength variants from becoming separate modes. Monotone scalar curves and
convex output do not guarantee a positive 3D Jacobian; that remains a hard
measured gate.

## Frozen synthetic truths

The truth family is independent of the candidate curve parameterization. A
stage computes three smooth D65-Lab direction weights at config-fixed angles.
For channel `c`, the exact stage equation is:

\[
y_c=x_c+x_c(1-x_c)\,q
  \gamma\sum_j p_j\,g_{j,c}\left[1+s_j(2\ell-1)\right],
\]

Here \(\ell=0.2126r+0.7152g+0.0722b\). The config freezes all stage A/B
angles, gains and luma slopes, and freezes the global gain scale
\(\gamma=0.4\). Their convex direction weights and bounded
gains make each residual vanish on cube faces for its corresponding channel.
No clipping is permitted.

Both ordered compositions, `A then B` and `B then A`, are targets. Before
fitting, they must:

- differ by at least `0.002` RGB RMSE;
- stay in the cube to `1e-12`;
- have minimum finite-difference Jacobian determinant at least `0.2`;
- have maximum Jacobian spectral norm at most `3.0`.

Failure of a truth prerequisite invalidates execution; it does not authorize
retuning the truth.

The pre-freeze analytic feasibility check used only the equations above. On
the frozen 25-grid the two orders stay exactly in `[0,1]` and differ by
`0.00204497` RGB RMSE. A float64 `21^3` interior finite-difference check at
`h=1e-5` gives minimum determinants `0.51975/0.52039`, maximum spectral norms
`2.73176/2.87400` and zero non-positive samples. These are contract-design
checks, not formal AL1 results; the committed implementation must reconstruct
them independently.

## Development and confirmation

All candidates fit only the `11^3` development grid. Confirmation uses an
independent `13^3` cell-centred interior plus all faces, edges and vertices.
Normalization, initialization and selection use development samples only.

The deterministic float64 CPU fit uses one thread, seed `71281`, Adam,
learning rate `0.03`, four restarts and 2,500 steps. Lowest development
objective wins; restart identity breaks exact ties.

Controls receive the same restarts, steps and optimizer:

1. three global degree-six monotone Bernstein curves;
2. the existing stationary `K=3` cube flow with 81 raw parameters and eight
   integration steps.

## Automatic gates

Both target orders must independently pass.

### Structural

- identity maximum error `<=1e-12`;
- partition sum error `<=1e-12`, no negative weight;
- neutral-axis channel spread `<=1e-10`;
- cube range within `[-1e-12,1+1e-12]`;
- minimum Jacobian determinant `>=0.1`;
- zero negative-Jacobian samples;
- maximum Jacobian spectral norm `<=4.0`;
- inverse roundtrip maximum error `<=1e-5`;
- serialization and chunk/partition replay `<=1e-12`;
- output jump across every analytic sector bisector `<=1e-5`.

### Strength and ID11 regression precursors

- strengths `0,.25,.5,.75,1` remain collinear to `1e-12`;
- distance from identity never decreases beyond `1e-12`;
- a 4,097-sample red-dominant ramp has luma step `>=-1e-5`;
- its maximum luma second difference is `<=5e-4`.

These synthetic checks do not replace a later full-resolution ID11 visual
regression. They only prevent an obviously discontinuous or posterized
operator from advancing.

### Capacity

- candidate confirmation RGB RMSE `<=0.006`;
- at least 25% RMSE improvement over global curves;
- RMSE no more than `1.20x` stationary K3.

The first proves useful colour-selective capacity. The second prevents
retaining a novel but inefficient representation when an existing bounded
operator already does the job.

## Branch decision

If all automatic gates pass, retain this exact representation for a separately
frozen photographic challenger. A pass does not open current film pixels,
operator fitting, stock/mode claims, training or production integration.

Any truth, absolute, structural, global-control or K3-efficiency failure closes
the fixed representation. After seeing confirmation results it is forbidden
to add/move sectors, change bandwidths, add curves/control points, alter
truths/grids/optimizer/steps/gates, or rescue the branch with photographs or
new data.

## Claim ceiling

`repeat-exact clean-room synthetic representation-capacity and
structural-safety evidence for one fixed analytic colour-selective monotone
curve operator`.

There is no photograph, film-pixel, unpaired-identification, stock,
calibration, preference, severe-safety or production claim.
