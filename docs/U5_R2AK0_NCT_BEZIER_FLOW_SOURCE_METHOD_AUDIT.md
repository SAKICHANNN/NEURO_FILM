# U5.R2AK0 — NCT Bezier-Flow Source and Method Audit

## Decision

Retain the **time-dependent colour-flow trajectory** as a representation
prior. Close direct NCT execution, dataset acquisition and model training.

NCT is relevant because its neural encoder predicts parameters of an explicit
ODE colour flow rather than directly synthesizing the final RGB image. It does
not, however, solve the project's central unpaired operator-identification
problem. Its per-image canonical map is learned from each image's own colour
distribution, so content, lighting and capture nuisance remain inside the map
that is later composed as a “style” transform.

## Primary-source freeze

Audited on 2026-07-28:

- CVPR 2026 paper, 16,019,166 bytes, SHA-256
  `9d8935d4ee4b1aa4cfc426940747c6c816bf8befc2ae44ad90dfc439f460ed8a`;
- official supplemental PDF, 82,609,791 bytes, SHA-256
  `4a550ac440b59ae296e93762c3f4d9fa2a98232b25a26b666d1e74a369d12580`.

The [official CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Nonlinear_Color_Transfer_via_Learnable_Bezier_Flows_CVPR_2026_paper.html)
labels the work “Nonlinear Color Transfer via Learnable Bezier Flows” and
links a PDF plus a supplemental item. The paper says code is provided in the
supplement, but the official supplement link resolves only to a PDF. The
audited PDF contains additional analysis and images, not source code. No
official repository, checkpoint or code licence was found in the paper,
supplement, official page or bounded exact-title repository search.

This is a reproducibility and rights gap. It is not permission to reconstruct
or redistribute an undeclared implementation.

## What the method actually does

For each image \(I_i\) with empirical RGB distribution \(\mu_i\), NCT learns a
bijective map \(T_i\) to a shared uniform distribution \(U\). Colour transfer
is then

\[
T_{c \rightarrow s}(x) = T_s^{-1}(T_c(x)).
\]

Phase 1 independently fits each image's time-dependent velocity field. It
uses the quadratic Bezier path

\[
z_t=(1-t)^2z_0+2t(1-t)z_g+t^2z_1,
\]

where a small MLP predicts the control point \(z_g\). The paper reports 5,034
per-image flows, each trained for 10,000 iterations.

Phase 2 trains an EfficientNet-B6 encoder with four MoE experts for 750,000
iterations to predict a flow code for unseen images. Inference composes the
content and style flows using an eight-step ODE solver.

The useful architectural fact is narrow: a model can predict an explicit
colour-flow parameterization while the renderer remains pixelwise. The Bezier
curve bends the **training trajectory** between endpoint distributions; it
does not add evidence that those endpoints isolate a film operator.

## Why direct project execution closes

### Endpoint identifiability is unchanged

The map \(T_i\) is defined from one image's empirical colour distribution.
Scene palette, exposure, white balance, illumination, camera response and
content composition therefore alter the estimated canonical map. A nonlinear
path can reduce flow-matching or reconstruction error without proving that
the endpoint map represents a reusable look.

This is exactly the distinction already established internally:

- U5.R2W1: output-only reference recovery remains strongly content
  classifiable and fails unseen operator recovery;
- U5.R2S3: improved unpaired distribution loss does not recover the hidden
  operator;
- U5.R2S4: correct content-condition matching improves distribution alignment
  but still fails operator identification.

NCT does not publish a control that separates these endpoint-identification
failures from trajectory fit.

### Project safety gates are absent

The paper reports an average Lipschitz statistic and an image reconstruction
error. The audited sources do not establish the project's required:

- exact RGB-cube range;
- positive minimum Jacobian determinant;
- bounded negative-Jacobian fraction;
- inverse and partition replay;
- endpoint/new-boundary control;
- full-resolution severe-artifact veto;
- OOD fallback.

“Continuous” or “bijective” prose is not a substitute for those measured
gates. Eight numerical ODE steps also require an explicit reconstruction and
inverse error audit.

### Data and claims do not match film learning

The reported training corpus consists of general natural/art images and
per-image flow pseudo-labels. It contains no evidence-backed
`film_stock_id`, roll, process, scanner or paired digital/film operator truth.
Training that encoder would not open stock-first learning or latent modes.

## What survives

Only this candidate prior survives:

> A separately versioned clean-room explicit colour flow may use a quadratic
> temporal basis or another bounded time-dependent velocity field, if it asks
> a new synthetic representation/capacity question and preserves the existing
> cube, Jacobian, inverse and replay gates.

The project already owns the structurally stronger stationary
cube-diffeomorphic O0 representation. Any future time-dependent pilot must
show a material, preregistered advantage over O0 at a controlled parameter and
compute budget. It may not claim that curved trajectories solve unpaired
operator identification.

No NCT data download, full-model reimplementation, checkpoint execution,
photographic frontier, stock claim, training or integration opens.
