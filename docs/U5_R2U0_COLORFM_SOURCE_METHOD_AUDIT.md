# U5.R2U0 ColorFM Source and Method Audit

Date: 2026-07-26

Decision: **retain hierarchical colour coupling as a clean-room synthetic
control; do not treat it as identified film transport**

## Why it matters

[ColorFM](https://arxiv.org/abs/2607.07119) was accepted to ECCV 2026 and is
the closest current reference-transfer method to the project's explicit-flow
line. It produces colour through a global RGB velocity field rather than a
spatial image generator, and its paper explicitly targets the two failures
the user cares about: under-stylization and colour artifacts.

The audit read the complete 17-page v1 paper, visually checked the coupling,
parameter-prediction and data pages, and inspected the complete official
repository tree at commit
`153798ae878c291cdf6cbac71097714e1eb6a578`.

## ColorFM-O: useful transport prior, not observed truth

For one content/reference pair, ColorFM-O:

1. segments both images with a pretrained semantic model;
2. matches pixel distributions belonging to shared semantic labels;
3. recursively centres source and target RGB subsets, partitions them into
   corresponding octants, and randomly pairs pixels in terminal cells;
4. fits a two-layer velocity MLP to those constructed pairs;
5. renders by five-step midpoint ODE integration.

The hierarchical colour coupling (HCC) is a practical inductive bias. It
reduces the incoherent paths produced by random coupling and avoids the
desaturation the authors report for minibatch OT.

It is still a constructed coupling. No source pixel is the same physical
sample as its matched reference pixel. HCC selects one plausible transport
from many distribution-preserving maps; it cannot establish that this is the
hidden photographic operator.

## ColorFM-L: parameter prediction distilled from that prior

ColorFM-L uses 237,408 ColorFM-O-generated content/style/output triplets from
Unsplash and DIV2K. ViT semantic features and cross-attention predict the
weights of a shared pixel-wise velocity MLP. A forward and reverse one-step
Euler update produces the final colour.

This is not a generative image decoder, and predicting global operator
parameters is compatible with the project's broad ML boundary in principle.
The published operator is not Style-safe as-is:

- its weights and output range are not project-bounded;
- one-step Euler transport does not guarantee a cube map or invertibility;
- the reported Lipschitz diagnostic is not a positive-Jacobian guarantee;
- all supervision inherits ColorFM-O's semantic and coupling assumptions.

Any project adaptation must predict or fit the existing U5.R2O0
cube-preserving explicit flow instead.

## Source and data status

The official repository is a public template with 14 tree entries: a README
and static images/videos. It contains no source code, checkpoint or licence.
The paper itself uses the arXiv non-exclusive distribution licence.

The paper describes Unsplash images categorized into seven semantic classes,
DIV2K random pairing, and an independent 40-image Unsplash test. It does not
provide the 237,408-row asset/rights/provenance manifest required by this
project. Therefore neither published training nor the apparent online demos
are eligible evidence for stock learning or distributable weights.

## Distinct project experiment, if reached

HCC can still answer a useful synthetic question after S3/S4:

```text
independent source/target distributions
  -> correct generated condition correspondence
  -> HCC pseudo-pairs
  -> fit the validated bounded U5.R2O0 flow
  -> evaluate against hidden analytic operator
```

Required controls are random pairing, shuffled condition correspondence,
distribution-only fitting and the correct generated conditions. Evaluation
must keep distribution match separate from hidden-operator recovery,
A/B-replicate stability and structural safety.

A pass would show only that HCC is a useful prior under the generated
conditions. It would not identify a real film operator. Real use additionally
requires evidence-eligible stock pixels and auxiliary conditions independent
of source, scanner, scene colour and content. Those gates are currently
closed.
