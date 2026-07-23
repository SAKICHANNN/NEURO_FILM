# U5.R2D statistics-to-explicit-LUT shortcut audit contract

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2D`  
Status: **frozen before implementation or audit results**

## Research question

Can a StatLUT-style spatially agnostic colour-statistics representation support
strong, explicit, non-generative LUT prediction without merely learning scene
palette/content shortcuts or pretending that an unpaired reference uniquely
identifies a film operator?

This leaf tests representation assumptions. It does not train a film model.

## Primary-source boundary

[StatLUT v1](https://arxiv.org/abs/2607.08227) was submitted on 2026-07-09.
Its image branch:

- converts an image to CIE Lab;
- extracts a 256-bin lightness histogram `H_L`;
- nonlinearly stretches `a,b` with `gamma=0.5`, `S=128`;
- extracts a soft 32×32 chroma histogram `H_ab`;
- extracts a 32×32 chroma-conditioned mean-lightness map `M_L|ab`;
- maps the resulting 2,304 values plus content statistics to a residual 16³
  LUT using a six-layer Transformer;
- trains from COCO images transformed by 10,000 LUTs.

The paper reports that 4,000 LUTs come from professional resources and 6,000
are synthetic, but does not publish the asset inventory, per-asset rights or
synthetic generator. It reports roughly 15 hours on eight A800 GPUs. No
official runnable code repository is linked from the arXiv record as of this
freeze. Therefore:

- this project cannot claim an exact StatLUT reproduction;
- the 4,000 professional LUTs are not an eligible dataset;
- the 8×A800 training recipe is not an authorized local experiment;
- the H-Diffuser/text/diffusion branch is excluded as generative AI;
- “artifact-free” and “semantic decoupling” claims remain paper claims, not
  project facts.

[SA-LUT](https://github.com/Ry3nG/SA-LUT) is available under a non-commercial
S-Lab license and predicts a spatial 4D LUT/context map. It remains a
research-only comparison and is not copied into this global descriptor leaf.
Its local branch has a higher halo/block/flicker burden.

## Project-specific risk

The current project already observes that low-frequency 4×4 scene colour can
predict nominal stock labels better than global RGB, while source geometry can
reach 92.86% separability. A histogram can be invariant to pixel permutation
while still strongly encoding:

- sky/foliage/skin/object palette;
- day/night and indoor/outdoor mixture;
- exposure and white balance;
- source/scanner gamut;
- the fraction of colours actually present.

Thus “spatially agnostic” is not equivalent to “content agnostic,” “stock
identifiable” or “operator identifiable.”

## Frozen hypotheses

### H-R2D-1 — permutation invariance

A faithful Lab-statistics extractor should be numerically invariant to an
arbitrary permutation of the same pixels.

Pass: component maximum absolute error `<=1e-12` in float64.

Failure: implementation is not spatially agnostic and closes the leaf.

### H-R2D-2 — palette/content dependence

The same extractor should distinguish deliberately different scene palettes,
proving that permutation invariance does not remove content colour.

Development witness: equal-sized red-dominant and blue-dominant synthetic
images. Pass for the **risk witness** when descriptor L2 distance is `>0.1`.
A pass is negative shortcut evidence, not a representation success claim.

### H-R2D-3 — unobserved-colour operator non-identifiability

Two explicit global LUTs that are identical on every colour present in a
reference image but differ on an absent colour must yield identical reference
pixels and descriptors, while producing different outputs on the absent-colour
probe.

Pass for the **counterexample** when:

- reference pixel maximum error `<=1e-12`;
- reference descriptor maximum error `<=1e-12`;
- held-out probe RGB L2 difference `>=0.1`.

This proves that single-reference statistics cannot uniquely identify a global
operator outside observed support. Passing blocks operator-identification
language; it does not reject statistics as a retrieval/control feature.

### H-R2D-4 — component validity

For finite encoded-sRGB inputs:

- `H_L` shape is 256 and sums to one within `1e-12`;
- `H_ab` and `M_L|ab` shapes are 32×32;
- raw soft chroma weights sum to one within `1e-12`;
- every component is finite and deterministic;
- output vector length is exactly 2,304.

Invalid shape/range/nonfinite inputs fail closed.

## Implementation boundary

Add an isolated NumPy research module under `src/roll2film/`:

- encoded-sRGB input with an explicit CIE Lab conversion;
- paper-described chroma stretch;
- linear `H_L` soft binning and bilinear `H_ab`/`M_L|ab` soft binning;
- square-root `H_ab` output plus raw mass for audit;
- immutable structured descriptor and deterministic serialization.

The exact paper downsample/interpolation implementation is unspecified and no
official code is available. This leaf operates directly on supplied pixels to
isolate representation properties and labels itself
`paper_compatible_lab_statistics_v1`, not an exact reproduction.

No image download, pretrained weight, professional LUT pack, GPU, training,
fitting, renderer integration, production schema or stock pixel is allowed.

## Branch

- descriptor/property failure: repair only implementation errors; otherwise
  close statistics-to-LUT work;
- risk/counterexample passes: retain the descriptor only as a known
  content/palette-sensitive control and prohibit operator-identification
  claims;
- only after all properties pass may U5.R2D1 freeze a small CPU synthetic
  known-operator recovery benchmark with fully generated bounded LUTs;
- R2D1 must compare descriptor-only, content-only, descriptor-difference and
  explicit source+target controls;
- real-film fitting/training remains blocked by stock identifiability and
  rights regardless of synthetic results.

## DoD

- contract/config committed before code;
- independent reference checks for Lab conversion and soft-bin mass;
- two audit runs byte-identical;
- focused and full CPU tests pass;
- results and counterexample propagate to active authorities;
- scoped commits are pushed;
- Goal remains ACTIVE.
