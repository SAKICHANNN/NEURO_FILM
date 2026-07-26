# U5.R2N0 ModFlows Source and Method Audit

**Date:** 2026-07-26

**Node:** `ULT > U5 > U5.R2 > U5.R2N0`

**Decision:** **feasible as an isolated clean-room external research control
only**. Open a checkpoint-integrity and synthetic-identifiability leaf for the
small B0 model. Do not copy the unlicensed official code, access stock-labelled
pixels or treat palette transfer as an identified film operator.

## Why this method is relevant

[Color Transfer with Modulated Flows](https://arxiv.org/abs/2503.19062)
learns one invertible RGB flow per image palette. At inference:

```text
content RGB
  -> content-conditioned forward flow
  -> common uniform colour cube
  -> inverse reference-conditioned flow
  -> transferred RGB
```

The encoder output is also evaluated by the paper as a palette embedding for
similar-style retrieval. This directly tests a stronger version of the
project's original idea: retrieve a specific style case and apply its explicit
transform instead of averaging many references into one bland LUT.

The B0 flow has 515 parameters. From the published equations and parameter
count, its clean-room architecture is a two-layer velocity MLP with four inputs
(`RGB+t`), 64 tanh hidden units and three outputs. The EfficientNet-B0 encoder
predicts those parameters; deterministic ODE integration applies them. There
is no image decoder or spatial resampling.

## Current source and licence facts

- paper: AAAI 2025; temporary PDF SHA-256
  `c9228e04...4a67449`;
- official repository commit:
  `e1884a0208160a00e0d48da7cc25e804b92772be`;
- the repository has no root licence, so its code cannot be copied or imported;
- the official Hugging Face model card labels weights MIT;
- pinned B0 weight: `18,970,914` bytes, LFS SHA-256
  `124f7b42...d3eacb`, repository revision
  `f7b70779...83635`;
- B6 weights are about 240MB each and are neither necessary nor authorized for
  this first audit;
- the paper reports B0 training on 4,767 rectified flows from
  `laion-art-en-colorcanny`; complete asset-level lineage is not supplied.

Accordingly, the checkpoint is internal research evidence only. It cannot
support commercial/production reuse, released weights or stock truth.

## Epistemic and safety risks

The method is unpaired palette transfer, not unpaired film-operator
identification. It can match a reference distribution while assigning the
wrong semantic colors. The paper explicitly reports unintended replacements
such as yellow becoming red, increasing artifacts at high strength or fewer
ODE steps, and an average map Lipschitz estimate around `37.26`.

The whole-image EfficientNet may also encode scene/content/source information,
despite the paper's palette-embedding interpretation. Current project evidence
already shows that scene colour and source geometry can dominate stock labels.
Therefore palette/content disentanglement must be tested rather than assumed.

## N1 gate

U5.R2N1 may:

1. download only the pinned 18.97MB B0 checkpoint;
2. verify its LFS hash and load it with weights-only deserialization;
3. implement the published flow equations independently;
4. audit checkpoint/architecture compatibility;
5. test determinism, identity, range, Jacobian/local gain, RGB-cube OOD,
   spatial permutation and palette-preserving/content-changing controls on
   synthetic images.

N1 may not copy official source, download B6, access current stock-labelled
pixels, render a stock claim, integrate production code or promote raw neural
ODE output. A passing synthetic audit would open only a separately frozen A0
non-stock reference safety pilot.

## Claim ceiling

Current-source feasibility for one MIT-labelled external B0 palette-flow
checkpoint and a clean-room synthetic audit. No code reuse, training-data
clearance, stock identity, film operator, safety, preference or production
claim.
