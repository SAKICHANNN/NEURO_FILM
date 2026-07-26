# U5.R2Q0 SA-LUT Source and Training-Boundary Audit

Date: 2026-07-26  
Node: `ULT > U5 > U5.R2 > U5.R2Q0`  
Decision: **do not run the official checkpoint; retain only the explicit
architecture as a clean-room prior**

## Why this method matters

[SA-LUT](https://github.com/Ry3nG/SA-LUT) is unusually close to the project's
allowed ML boundary. At inference, a neural network does not directly decode
the final RGB image. It predicts:

- a softmax vector over 64 learned LUT bases;
- one bounded spatial context map;
- an identity-residual `2 x 17 x 17 x 17` 4D LUT.

The final image is produced by explicit quadrilinear interpolation over the
context and RGB coordinates. This is materially more interpretable than a
direct neural RGB generator and is a useful future architecture prior.

## Exact inference boundary

The official ICCV 2025 paper and revision
`3e62f9c9b59ea7b30d220ff6ba61047dab898c9a` agree on the main structure:

1. VGG features and AdaIN combine style and content information.
2. A classifier predicts 64 nonnegative weights summing to one.
3. Those weights combine learned 4D LUT residual bases with an identity LUT.
4. A content-style cross-attention network predicts a one-channel map in
   `[0,1]`.
5. The context map and RGB image are transformed by quadrilinear
   interpolation.

The fused LUT nodes are clamped to `[0,1]`. This bounds node values, but it
does not prove semantic safety, full-resolution clipping safety, smooth
spatial behaviour, seam freedom, OOD behaviour or absence of posterization.
The context map is a neural spatial parameter, so it would need separate
finite-support, resolution, content-shortcut and severe-artifact gates.

## Why the published training route is ineligible

The official checkpoint does **not** have a clean “network predicts only
explicit parameters” training lineage.

The paper describes two streams:

- synthetic supervision from professional 3D LUTs applied to Log images;
- a real-style stream in which a learned `Style2Log` model generates a
  synthetic Log content image and adversarial loss supplies the target signal.

The released trainer makes this concrete. It loads a frozen
`Style2VLogImage2ImageNet`, directly generates `fake_vlog` RGB tensors, and
uses a learned style discriminator with manual GAN optimization. The direct
image generator is upstream of the LUT predictor even though it is absent at
inference. That violates the current no-generative/pseudo-teacher training
boundary for project evidence.

The committed training config also leaves the image and LUT roots blank.
There is no manifest for the training photographs, professional LUTs,
Style2Log training assets or final checkpoint lineage. Consequently the
published checkpoint cannot be used as reproducible project evidence.

## Source and asset integrity

- Official ICCV PDF: `14,587,761` bytes, SHA-256
  `a8849149136e7ff1234a720fa759d661c2b0a323d46b8ac740cc0281ca619949`.
- Official source: 62 tracked files at revision
  `3e62f9c9b59ea7b30d220ff6ba61047dab898c9a`.
- Source licence: S-Lab License 1.0, non-commercial use only.
- Included VGG weight: `80,102,481` bytes, LFS SHA-256
  `804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a`.
- Included Style2Log checkpoint: `286,604,792` bytes, LFS SHA-256
  `c7256590a50fb52176c5fccc83c04f4e362f9ac03e0ccf2ce8569d3ed4c4d790`.
- Published inference state: `218,400,931` bytes, LFS SHA-256
  `5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b`;
  no model card; not downloaded.

The source clone and its two automatically fetched LFS assets remain ignored
and protected. No project source imports or copies this implementation.

## PST50 is not the missing film dataset

The authors' [PST50 benchmark](https://huggingface.co/datasets/zrgong/PST50)
is labeled CC-BY-4.0 and reports 250 files/rows and 35.5 GB. It contains 50
Rec.709 contents, 50 Log contents, 50 paired references, 50 professionally
graded ground truths and 50/51 unpaired style references, plus video.

PST50 evaluates generic photorealistic style transfer. Its paired ground truth
was produced with professional LUTs and DaVinci Resolve refinement, not by
capturing a named film stock. It supplies no `film_stock_id`, roll, process or
scanner groups. Downloading 35.5 GB would not repair current stock
identifiability, so no dataset payload was requested.

## Frozen decision

Do not download or run the official SA-LUT inference checkpoint and do not
train the released pipeline. This is a provenance and architecture-boundary
closure, not a claim that 4D LUTs are useless.

Retain the following clean-room idea for a future evidence-eligible leaf:

```text
stock/mode already selected
  -> bounded global LUT-basis weights
  -> separately bounded low-resolution context grid
  -> explicit 4D interpolation
  -> deterministic full-resolution renderer
  -> OOD/global-champion fallback
```

Any future implementation must remove the direct Style2Log generator and GAN
teacher, use manifest-complete rights-cleared supervision, keep global and
spatial contributions separately ablatable, and pass severe-artifact,
content-shortcut, range, monotonicity, resolution and tile-parity gates.

This audit opens no checkpoint evaluation, PST50 download, film fitting,
stock/mode claim or production integration.
