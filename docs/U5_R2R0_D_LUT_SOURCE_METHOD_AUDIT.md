# U5.R2R0 D-LUT Source and Method Audit

Date: 2026-07-26

Node: `ULT > U5 > U5.R2 > U5.R2R0`

Decision: **open only a fixed published-asset structural audit**

## Relevance to the requested algorithm

[D-LUT](https://github.com/leemojiang/D-LUT) is a WACV 2025
reference-conditioned colour-transfer method. It trains a small score function
on the RGB pixels of one style photograph, moves an identity colour lattice
toward that palette with Langevin dynamics, averages 50 stochastic
trajectories, and exports the result as a reusable `15^3` LUT.

This directly tests the user's intuition that a related reference photograph
could determine how a particular input should be graded. It does not require
paired digital/film captures or a pretrained image generator.

## Generative-AI boundary

The word “diffusion” needs a precise classification here.

- D-LUT does not use a diffusion image model, VAE, denoising U-Net or
  text-to-image generator.
- Its 93K-parameter per-style MLP estimates only a three-dimensional RGB
  density score.
- Langevin dynamics generates LUT node values, not final image pixels.
- Final RGB is rendered by a deterministic LUT.

It is therefore eligible for an isolated explicit-operator audit under the
project's narrow “ML may predict LUT parameters” rule. That does not make the
published mapping safe or evidence-backed.

## Exact method and its identifiability limit

The score model estimates `grad(log p_style(rgb))` from a single image's
uncontrolled RGB pixel distribution. Each identity-LUT node follows 40
stochastic updates:

```text
x[t] = x[t-1] + epsilon/2 * score(x[t-1]) + sqrt(epsilon) * noise
```

Fifty independent LUT trajectories are averaged. This lowers random variance,
but the objective contains no content correspondence, exposure/WB control,
stock identity, scanner/process model, boundary constraint, orientation
constraint or semantic colour correspondence.

The paper itself says references with similar foreground/background colour
proportions work best and dissimilar references can give unnatural mappings.
That is direct evidence that D-LUT relies on scene-palette compatibility. It
cannot identify a stock operator from unrelated scans and would amplify the
scene-colour shortcut already measured in SF1.0B/SF1.3B.

## Source integrity

- Official paper: `7,265,681` bytes, SHA-256
  `ceaaebaa5638b80c6440982fe67fc63b47736c70ea998be19fe9b502096828d1`.
- Official Apache-2.0 repository revision:
  `01927996eb03ad21be7f9cdbeb7a4c6604e7431c`.
- Repository: 89 tracked files and `29,209,418` retained payload bytes outside
  `.git`.
- Published score checkpoint: `374,671` bytes, SHA-256
  `7686a2b0c42b89b59354a35233c67ef87e298d4cfc1b1073554d4bf03cb9d4f0`.
- Published trajectory: 41 contiguous `LUT_0` through `LUT_40` files,
  `4,156,160` bytes, canonical manifest SHA-256
  `a1263c4e547c8f2f66c44ac927a159c3e9599b73e737a35248601c59d1f2e1e3`.

The paper states `epsilon=0.0004`, while the released demo uses `0.0002`.
This contradiction is retained; the existing LUT trajectory is audited as an
artifact rather than claimed as an exact paper reproduction.

## Development inspection disclosure

Before freezing the exact R1 contract, the official LUTs were parsed
read-only to verify that the assets were meaningful. That inspection revealed
out-of-range nodes and negative local determinants. R1 is therefore explicitly
a post-exploratory deterministic characterization, not blind confirmatory
evidence. Its gates are inherited from existing bounded-operator contracts,
not tuned to the observed failures.

## Handoff

U5.R2R1 may:

- parse and hash all 41 official LUTs;
- audit identity, range, exact tetrahedral orientation and
  published-compatible trilinear Jacobians;
- run twice for byte-identical evidence;
- close before images if no nonidentity step survives.

It may not train, render project photographs, clip/project/smooth the LUT,
change the published schedule, or make any film-stock claim.

