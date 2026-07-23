# CanonCGT external ML source and feasibility audit

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2F0`  
Decision: **bounded external E2E control is feasible; full SSL reproduction is unavailable**

## Why this method matters

CanonCGT is unusually relevant to the project’s failed “similar photo should
suggest the right colour transform” hypothesis. It explicitly separates:

1. an input-conditioned canonicalizer that predicts a 17-cube LUT;
2. a reference-conditioned grader that predicts another 17-cube LUT;
3. deterministic LUT application to the full-resolution image.

The model therefore predicts explicit colour operators rather than a final
generative RGB image. Its canonical-pivot idea directly attacks the missing
neutralization problem measured in U5.R2D1/D2. The paper and official code are
available from the [CVPR 2026 paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.html)
and [official repository](https://github.com/Jinwon-Ko/CanonCGT).

## Immutable source inventory

| Item | Evidence |
|---|---|
| repository commit | `229a7d3ec24af19ea5d97bb136f0317762bb9261` |
| repository/license | Apache-2.0; local `LICENSE` SHA-256 `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| formal paper | 10-page CVPR 2026 PDF, 2,466,467 bytes, SHA-256 `5de5a83241f5ec4251248a7b5e565f616cf15d40de6fb6b1c842f15dddef9811` |
| public E2E checkpoint | 20,914,369 bytes, SHA-256 `916f7ad5028d3fef51cf915bb51febc9508e7a378c0babe9ff3d3992b8c594f7` |
| public SSL path | only a 2-byte placeholder, SHA-256 `7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6` |
| MobileNet-v2 initializer | 14,258,573 bytes, SHA-256 `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f` |
| local runtime | PyTorch `2.11.0+cu128`; RTX 5070 Ti Laptop 12,227 MiB |

The repository README says source code and pretrained weights are Apache-2.0.
The Lightroom presets and generated supervised dataset are explicitly not
redistributed and retain their original providers’ terms. This project does
not acquire, reconstruct or train on those preset assets.

## Paper evidence and claim boundary

The supervised phase uses MIT-Adobe FiveK Expert C as the canonical target and
56 Lightroom preset styles over 5,000 images, split 4,500/500. The full
self-supervised phase reports 88,516 training and 20,196 testing images from
six public datasets. Because no paired reference-grading dataset exists, the
paper uses a self-referential crop/perturb/reconstruction protocol rather than
real film/digital pairs.

The paper’s own limitation is important for Ultimate: strong global tints are
transferred selectively to preserve balance, and extreme references such as
black-and-white are not fully followed. This is a stability feature but may
produce exactly the subdued style salience the owner rejects.

The available `E2E.pth` is epoch 9 of the supervised end-to-end stage. It loads
all 5,056,383 forward parameters with zero missing keys; 57 extra stored style
centroids are unused by the inference class. The advertised SSL checkpoint is
not actually present. Therefore:

- an exact public E2E-control run is feasible;
- a complete published DP-CGT/SSL reproduction is not currently feasible;
- E2E results may not be presented as the paper’s complete SSL result;
- FiveK/preset training makes this a generic reference-grading control, not a
  real-film or named-stock model.

## Executable smoke evidence

The official input/reference pair `00` runs successfully on the local GPU and
is repeat-exact:

- input SHA-256:
  `370d671678bd758ba2464ff9e4784d3e5fefd6792e5c573358f260c0baf6c062`;
- reference SHA-256:
  `90e6dce18809bb94dece1b6b010f40f3cc63c968857322e5c11f1a704e79c9c1`;
- output SHA-256:
  `fbcc6e97a90214942a3a5b78fb0ef1b11de076b69a03a89a75b381ac3fbfcb1c`;
- output style Delta E76 `14.0259`;
- matched-basic residual Delta E76 `3.5688`;
- new hard clipping `0.1498%`.

The output is visibly strong and spatially intact, but this one example is
mostly explainable by basic adjustment under the project’s frozen diagnostic.
It is mechanics evidence, not a FilmStyleSafe pass.

## Explicit-operator risk

The method uses deterministic 3D LUT application, but its LUTs are not bounded
or monotone by construction. On the official pair:

- canonical LUT range is `[-0.2733, 1.3915]`, with `12.68%` outside `[0,1]`;
- canonical image range is `[-0.2292, 1.0173]`, with `0.773%` outside;
- grading LUT range is `[-0.0528, 1.0264]`, with `1.968%` outside;
- final tensor range is `[-0.02382, 0.99341]`, with `0.132%` outside;
- the official visualizer clamps before sRGB8 save.

Consequently CanonCGT satisfies “ML predicts explicit LUT parameters,” but it
does not satisfy Ultimate’s bounded-operator safety contract without an
independent fail-closed wrapper and clipping/artifact audit.

## Allowed next experiment

U5.R2F1 may run the immutable public E2E checkpoint as an isolated external
ML challenger under these constraints:

- weights-only checkpoint loading and exact source/checkpoint hashes;
- no training, fine-tuning or preset acquisition;
- reference images are conditions, never teacher truth or paired targets;
- any real-film reference remains A0 internal aesthetic evidence and cannot
  establish stock response;
- record both predicted LUTs, tensor ranges, clamp fractions and repeat
  identity;
- frozen gold-first automatic and full-resolution severe-artifact veto;
- compare style/non-basic value against safe-rich and anchor56;
- low confidence, out-of-range or severe output rejects the candidate;
- no production import, schema/default change or public-weight claim.

The first F1 feasibility pass should be small and fixed before inference:
official pairs for mechanics plus a bounded internal reference bank whose
rights, hashes and claim ceiling are explicit. It must test reference
sensitivity and same-reference consistency across multiple content images.

## Forbidden interpretation

Do not claim that CanonCGT:

- learned a film stock;
- recovered a digital-to-film operator;
- solved neutral canonicalization for this project;
- reproduced the paper’s SSL stage;
- is bounded merely because it outputs LUTs;
- validates raw image similarity or latent stock modes;
- can be promoted from one attractive example.

Real-film fitting, stock learning, LSM and production integration remain
closed. The Ultimate Goal remains ACTIVE.
