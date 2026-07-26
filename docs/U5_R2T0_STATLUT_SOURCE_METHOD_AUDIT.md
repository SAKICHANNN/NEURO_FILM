# U5.R2T0 StatLUT Source and Method Audit

Date: 2026-07-26

Decision: **retain the image-driven architecture prior; do not execute,
download assets or reopen histogram-capacity rescue**

## Why this source was audited

StatLUT appeared on arXiv on 2026-07-09, after the project's earlier
reference-conditioned LUT reviews. It is unusually relevant because its
image-driven branch does not render RGB with a neural image generator:
spatially agnostic Lab statistics condition a network that predicts one
global `16x16x16` LUT, and the LUT performs the final rendering.

The audit used the complete 17-page v1 paper and visually checked its method,
training, Lab-extractor and patch-shuffle pages. No model, dataset, repository
or project image was downloaded.

Primary paper:
[Wang et al., StatLUT, arXiv:2607.08227v1](https://arxiv.org/abs/2607.08227).

## What the image-driven method actually learns

The reference and content images are reduced to:

- a 256-bin Lab lightness histogram;
- a `32x32` Lab chrominance histogram;
- a `32x32` chrominance-conditioned mean-lightness map.

A residual/base/global mapper and Transformer decoder predict a residual from
an identity LUT. The result is clamped to `[0,1]`; monotonicity and total
variation appear as training penalties.

This is not trained from unrelated unpaired domains. The paper constructs
exact synthetic supervision:

1. take a COCO content image;
2. apply a known random LUT to obtain the target image;
3. spatially augment that same target to make a reference;
4. directly supervise both the known LUT and its same-image rendered output.

That is useful synthetic operator-recovery supervision. It is not
identification of a digital-to-film operator from unpaired film photographs.

## Why it does not solve the project's main failure

The Lab extractor deliberately retains scene palette and illumination
statistics. Its `M_L|ab` term even restores the association between specific
colours and their lightness. Those are useful for generic reference colour
transfer, but they are also precisely where this project has measured strong
content shortcuts.

The paper's patch-shuffle experiment shows that predictions are mostly
insensitive to where reference pixels occur. It does not show that:

- two different scenes under one operator produce the same inferred operator;
- an identity-style pair with different content returns identity;
- source, scanner or uploader cues are absent;
- a hidden true operator is recovered rather than a plausible distribution
  match.

U5.R2S2 already supplies those more discriminating controls. Raw target
statistics achieved the lowest synthetic error there but strongly transformed
identity-style negatives; the bounded learned predictor remained
content-sensitive and structurally unsafe. Adding a Transformer to the same
unidentified observation does not answer that failure.

## Structural and lineage gaps

The paper reports 10,000 training LUTs: 4,000 from unspecified “professional
color grading resources” and 6,000 synthesized by statistical sampling. It
does not provide an asset-by-asset source or rights manifest. As of this audit,
the paper exposes no official code or checkpoint, and its arXiv licence grants
article distribution rather than a software/data licence.

The image branch reportedly trains for about 15 hours on eight A800 GPUs.
That is unnecessary and outside the current resource contract. More
importantly, clamp plus monotonicity/TV penalties is not a formal
positive-Jacobian, invertibility or no-fold guarantee. The project's validated
cube-preserving flow remains the safer explicit representation.

The text branch uses Qwen-generated captions, CLIP and a diffusion Transformer.
It is generative and is excluded completely; it is not needed to assess the
image-driven idea.

## Retained value

The useful prior is narrow:

```text
explicit Lab distribution descriptors
  -> bounded parameter predictor
  -> deterministic global colour operator
```

If a future evidence-eligible data leaf establishes independent-content
operator supervision, the Lab `L + ab + L|ab` descriptor can be a clean-room
challenger. It must face identity-style, shuffled, independent-content,
held-out-source and hidden-operator controls, and its output should use a
validated cube-preserving representation.

No such data gate is currently open. StatLUT therefore creates no new
training, download, real-image, stock, LSM or production permission.
