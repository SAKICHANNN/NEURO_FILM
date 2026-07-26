# U5.R2V0 cmKAN Source and Method Audit

Date: 2026-07-26

Decision: **exclude the published training and inference path from the
Style-safe mainline; retain only a bounded-operator representation prior**

## Why it was examined

[cmKAN](https://arxiv.org/abs/2503.11781) is a recent ICCV 2025 colour
matching method with supervised and unpaired training modes, released source,
checkpoints and a paired camera dataset. Its use of splines and a
hypernetwork initially appears compatible with the project's rule that ML may
predict explicit colour parameters.

The implementation boundary is materially different. This audit inspected
the official repository at commit
`91e7f9429f0a7dd7677f20d0fa63de8a566c96ab`, including the licence, model,
KAN layer, hypernetwork, unpaired pipeline and Volga2K configuration.

## What the implementation actually renders

For every input pixel, the content encoder and attention network predict a
different set of:

- cubic B-spline coefficients;
- spline weights;
- SiLU residual weights.

The KAN evaluates those spatially varying parameters against the input pixel
and returns the output image directly. This is not a single image-global LUT,
curve, matrix or bounded velocity field. The parameter generator has access
to local and global image structure, including convolutions, wavelet
downsampling and attention.

The published implementation does not constrain the resulting mapping to:

- remain inside the RGB cube;
- preserve cube endpoints;
- have a positive Jacobian;
- be invertible;
- remain spatially smooth or content independent;
- satisfy a fixed severe-artifact budget.

The fact that a spline appears inside the network therefore does not by
itself make the result an explicit Style-safe renderer.

## The “unsupervised” lane is adversarial

`UnsupervisedPipeline` instantiates two cmKAN generators and two patch
discriminators. Training uses least-squares adversarial losses plus cycle and
identity losses, following the CycleGAN pattern. It is non-diffusion, but it
is still a generative adversarial image-to-image system whose learned
spatially varying operator directly emits RGB.

This violates two active project boundaries:

1. ML may predict bounded explicit parameters but may not directly generate
   final Style-safe RGB.
2. Content and spatial features may not become shortcuts that masquerade as
   stock response.

Running the published unpaired checkpoint cannot establish a real
digital-to-film operator and would not answer the current S3/S4
identifiability question.

## Source, data and rights

The official repository is substantially more complete than the current
ColorFM release: code, experiment configurations, checkpoint links and
dataset links are present. The repository licence is CC BY-NC-SA 4.0 with an
additional explicit academic-research-only statement; product or commercial
reuse is not cleared by this audit.

Volga2K contains aligned captures from two camera sensor/processing paths.
Other listed tasks use Adobe FiveK, Samsung2iPhone and Zurich raw-to-sRGB.
These can test camera colour matching, but none is evidence for a named
photographic-film stock. Dataset availability is not a substitute for a
project-grade rights, lineage, duplicate and split-leakage audit.

No checkpoint or dataset download is justified for the current leaf.

## What remains useful

KAN splines remain a possible compact nonlinear colour representation. A
future project-compatible variant would need to predict one image-global,
bounded parameter set and render through a deterministic module with explicit
range, monotonicity/Jacobian, inverse, repeat and artifact gates. It would
still require eligible stock evidence and cannot rescue failed
identifiability by adding capacity.

The current ready work therefore remains the frozen bounded explicit-flow
S3/S4/U1 chain. cmKAN is retained as a well-documented external baseline and
architecture boundary, not an executable film-simulation candidate.
