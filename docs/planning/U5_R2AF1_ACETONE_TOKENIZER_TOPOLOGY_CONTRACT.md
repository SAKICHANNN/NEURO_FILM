# U5.R2AF1 — AceTone tokenizer topology contract

Date: 2026-07-28

## Parent and question

AF0 closes the published generative selector but allows one exact-checkpoint
stress of the separate LUT tokenizer. AF1 asks only:

> Does the frozen 64-token VQ-VAE preserve the topology of already safe
> analytic 3D LUTs?

It does not ask whether AceTone matches references, looks like film, learns a
stock, or improves the product.

## Definition of ready

- exact external repository revision, model-source hash, checkpoint hash and
  runtime are frozen;
- only the repository-included Apache-2.0 tokenizer is used;
- no Qwen, GRPO, benchmark, photograph, training or current project pixel is
  used;
- eight analytic `32^3` LUTs are fixed before checkpoint execution;
- every source LUT is range-safe and orientation-preserving under the same
  finite-cell Jacobian convention used by AE1.

## Fixed population

Each LUT is composed from channel curves
`f_c(x)=x+a_c*x*(1-x)` and a row-normalized, positive-determinant matrix.
The eight records include identity and seven deliberately different bounded
looks. The weakest source minimum determinant must remain at least `0.25`,
and source negative-Jacobian fraction must be exactly zero.

A duplicate `warm_soft` input verifies deterministic identity. A red/green
output-axis swap of identity is a metric-only negative control and must be
detected as at least `99.9%` negative Jacobian. It is not sent through the
tokenizer.

## Execution

Two independent CPU processes use CPython 3.12.10, PyTorch 2.11.0+cu128,
one thread, deterministic algorithms and seed 29041. The runner verifies
external revision, source and checkpoint hashes before importing the external
model. It uses `torch.load(..., weights_only=True)`, calls only
`encode_indices` and `decode_indices`, and writes arrays plus a strict
manifest under ignored outputs.

Tracked project evaluation consumes manifests and arrays without importing
AceTone.

## Frozen gates

All source controls, external lineage, exact two-run replay and duplicate
identity must pass. Every reconstructed LUT must be finite and inside
`[0,1]`, with:

- negative-Jacobian fraction at most `0.5%` at determinant threshold `-1e-5`;
- maximum local Jacobian spectral norm at most `20`;
- RGB RMSE at most `0.03`;
- median/p95 Delta E76 at most `2.0/5.0`;
- new interior hard clipping at most `0.5%`.

These gates are conjunctive. Average fidelity cannot compensate for topology
failure.

## Branches and definition of done

- Runtime or lineage mismatch: close without alternate checkpoint/environment.
- Source or metric control failure: invalidate and repair only project-owned
  harness defects before inspecting model metrics.
- Topology failure: close the learned tokenizer representation; no decoder,
  loss, checkpoint, projection, smoothing, clipping or threshold rescue.
- Fidelity failure: close without capacity/checkpoint search.
- Full pass: retain only compact-representation feasibility. Any
  non-generative parameter predictor would require a new frozen hypothesis.

No AF1 outcome opens film/stock/reference claims, current-pixel fitting,
FilmCase, LSM, product integration or visual review.
