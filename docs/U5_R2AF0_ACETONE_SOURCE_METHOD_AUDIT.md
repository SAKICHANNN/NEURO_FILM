# U5.R2AF0 — AceTone source and method audit

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AF0`  
Decision: `published_selector_excluded_tokenizer_stress_only`

## Question

Does AceTone provide a legally and epistemically usable, non-generative,
orientation-preserving reference-to-LUT algorithm that is distinct from the
closed `U5.R2AE1` spectral LUT bank?

## Exact sources

- Paper: `AceTone: Bridging Words and Colors for Conditional Image Grading`,
  arXiv `2604.00530v1`, 2026-04-01.
- Official repository: `martian422/AceTone`, commit
  `916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5`, 2026-04-15.
- Repository licence: Apache-2.0; exact `LICENSE` SHA-256
  `6d1d968fb225eca367cb7f0b8831ab012a35d92b547e945e17ef8e7b05c3e5cc`.
- Included LUT-tokenizer checkpoint:
  `model/acetone-vqvae-d64.pt`, 21,383,912 bytes, SHA-256
  `115e4e8c147655fe0ae0c7fd494ec6cd3a8ff5a441ce536a886edb1ea556ef92`.
- Public preview model: Hugging Face revision
  `9416b259dec1bf7bb01ba2dab38cf6470cfa98f0`, card licence Apache-2.0.
- Public transfer benchmark: Hugging Face revision
  `b504dfc28882281b1e77511227153294a6782360`; 1,024 rows each of raw,
  reference, ground-truth and LUT files. It has no dataset card or declared
  licence in the live API response.

All external repositories remain under ignored `tmp/`; no external source,
weight, image, LUT or dataset payload is copied into tracked project files.

## Method findings

### The complete selector is forbidden

AceTone explicitly formulates grading as a generative colour transformation.
It extends Qwen2.5-VL with 256 LUT tokens and autoregressively generates 64
tokens from text or reference images. SFT is followed by GRPO using colour and
aesthetic rewards. This is generative AI under the project contract even
though the decoded final RGB is rendered by an explicit 3D LUT. The complete
AceTone selector, preview model, VLM training and RL path are therefore
excluded.

### The released supervision is not film evidence

The paper reports approximately 10,000 licensed filter LUTs, about 34,000
PPR-10K expert Lightroom-derived LUTs, and an 8,192-LUT fused library. Images
come from COCO, MIT-Adobe FiveK and PPR-10K. Random image/LUT pairing creates
the reference-transfer supervision. Qwen2.5-VL-32B generates instruction
annotations. None of this identifies a photographic-film stock or an
unpaired digital-to-film operator.

The repository does not release the 10,000-LUT per-asset rights manifest, the
full AceTone-800K corpus or the fused training library. The public benchmark
has no current dataset-level licence declaration. It is not eligible for
training, product integration or stock claims.

### The tokenizer is bounded but not topology constrained

The separate 3D VQ-VAE maps a `32^3 x 3` LUT to 64 tokens and decodes through
a sigmoid. Its frozen training objective is voxel MSE plus VQ commitment loss.
There is no monotonicity, positive-Jacobian, invertibility, Lipschitz or
orientation-preservation term.

Training augmentation adds smooth Gaussian noise at sigma `0.05`, an
unconstrained random `3x3` colour warp, brightness/contrast and gamma, then
clips to `[0,1]`. These operations are not certified to preserve LUT topology.
Consequently, `[0,1]` output and average reconstruction Delta E do not imply
artifact-safe colour geometry.

## Non-duplication and retained value

This does not reopen AE1 and does not repair its folded film LUTs. AceTone's
tokenizer is a distinct representation question:

> Can a compact learned explicit-operator representation reconstruct already
> safe analytic LUTs without introducing orientation reversal?

One frozen, synthetic-only `U5.R2AF1` checkpoint audit may answer that
question. It must use the exact included checkpoint, analytic
orientation-preserving LUTs, the existing AE1 Jacobian convention and
pre-registered gates. It may not run the Qwen selector, use photographs,
download the benchmark, train/fine-tune, alter the decoder, project or smooth
outputs, or claim film/reference/product validity.

## Branch decision

- Full AceTone/VLM/GRPO path: closed by the non-generative-AI contract.
- AceTone-800K/filter/fuse libraries: unavailable or lineage-incomplete; closed.
- Public benchmark pixels: unnecessary and rights-unidentified; do not download.
- Included VQ-VAE tokenizer: allowed only for the bounded AF1 synthetic
  topology stress test.
- A failed AF1 closes this representation without decoder/loss/threshold
  rescue. A pass would establish representation feasibility only and would
  not open current-pixel learning, FilmCase, LSM, stock claims or production.

