# Reference Color Match StatLUT-Lite Evidence

## Scope and source

StatLUT (arXiv:2607.08227, submitted 2026-07-09) reports a spatially agnostic
Lab extractor, a 0.38M-parameter residual mapper and a 16-cube output. Its
published extractor contains:

- a soft-binned 1D lightness histogram;
- a square-root-normalized 2D stretched-chroma histogram;
- mean lightness conditioned on each chroma bin.

The paper reports 70% Top-1 in a 1,000-vote six-method study, but also reports
training the image mapper for 200 epochs on COCO and 10,000 LUTs using eight
A800 GPUs for about 15 hours. No official code or weights were located during
this audit. Neuro-film therefore implements only an independent feature
reproduction and a much smaller falsifiable linear baseline. It is not a
StatLUT reproduction or performance claim.

Primary source:
`https://arxiv.org/html/2607.08227`.

## Implemented extractor

`statlut-lab-features.v1` uses 32 L bins and 8x8 chroma grids for the bounded
diagnostic. It converts display-linear sRGB to D65 Lab, applies
`sign(c)*(abs(c)/128)^0.5`, performs linear/bilinear soft binning, square-root
normalizes chroma frequency, and records conditional mean L.

Tests prove:

- exact feature layout and normalization;
- finite neutral-axis behavior;
- spatial permutation invariance within `2e-15`;
- equal-image source-batch order invariance;
- fail-closed input, policy and feature validation.

## Linear residual-mapper experiment

The model receives concatenated source-batch statistics, styled-reference
statistics and their difference (480 dimensions). It predicts nine operator
parameters, which are projected into the existing positive-matrix and monotone
tone contract before any RGB render.

- Fit: all generated fit operators, two independent observations each.
- Alpha selection: validation only.
- Confirmation: previously unopened stress indices 16--31.
- Identity and same-look controls are mandatory.
- No neural model, real pixel, external weight or direct RGB prediction.

## Result

| Metric | Result | Gate |
|---|---:|---:|
| Grid RMSE median | 0.12670 | <= 0.07 |
| Grid RMSE p90 | 0.18144 | <= 0.10 |
| Median captured style | -1.04939 | >= 0.50 |
| Improved over identity | 16.67% | diagnostic |
| Improvement over global mean | -74.52% | >= 25% |
| Same-look replicate grid RMSE | 0.12222 | <= 0.04 |
| Identity-reference max grid RMSE | 0.16367 | <= 0.01 |

Every gate fails. Two reports are byte-identical:

- ID `f1e2bdb3d72b5f7dba084dd22ad562532bad9acfed1201f30a32bc814ae741eb`;
- SHA-256
  `0b4cf6c91cf632315bcd80931b024d4cc977637e0a54cf2b5f12ac69f9b68716`.

## Decision

`statlut-lite-linear-route-closed`.

The Lab extractor is retained as tested research infrastructure. The linear
mapper is rejected. Because identity and same-look controls both fail badly,
capacity alone is not an eligible rescue on the same generated contract.
A future full StatLUT-class challenger requires independently licensed code or
weights, its published training scale or an equivalently justified new
training program, and fresh photographic/identity/patch-shuffle evidence.
