# IP2P Grid Search Results

> Last updated: 2026-05-26 on Windows RTX machine.

## Model Under Test

- Model: `outputs/ip2p_finetune/all_1000`
- Training data: `data/ip2p_train/all/train`
- Training size: 1440 pseudo-pairs, 8 film styles x 180 train pairs
- Training run: 1000 steps, 256px, batch 1, gradient accumulation 1, fp16
- Reload check: `StableDiffusionInstructPix2PixPipeline.from_pretrained(..., local_files_only=True)` succeeded with 8-channel UNet

## Probe Grid

- Input image: `data/ip2p_train/all/train/portra_400/input/fl_020a5cc43131bde2_99026de7b4.jpg`
- Output directory: `outputs/ip2p_grid/all_1000_portra400_probe`
- Manifest CSV: `outputs/ip2p_grid/all_1000_portra400_probe/fl_020a5cc43131bde2_99026de7b4_grid_manifest.csv`
- Manifest JSON: `outputs/ip2p_grid/all_1000_portra400_probe/fl_020a5cc43131bde2_99026de7b4_grid_manifest.json`
- Styles: `ektar_100`, `hp5`, `portra_400`, `portra_800`, `tri_x_400`, `velvia_50`, `vision3_250d`, `vision3_500t`
- Grid: `image_guidance_scale = 1.0, 1.5, 2.0`; `guidance_scale = 5.0, 7.5, 10.0`
- Steps: 20
- Max side: 512
- Seed: 42
- Output count: 72 images

## Contact Sheets

Per-style 3x3 contact sheets were generated for Mac-side review:

- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/ektar_100_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/hp5_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/portra_400_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/portra_800_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/tri_x_400_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/velvia_50_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/vision3_250d_contact_sheet.jpg`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets/vision3_500t_contact_sheet.jpg`

## Current Decision

The grid was visually reviewed and should be treated as a failed experiment for production use.

Observed failure:

- Many outputs have severe painterly smearing.
- High `guidance_scale` values often destroy scene structure.
- Even the most conservative candidates are not reliable enough to use as a default film translation path.

The earlier provisional picks remain recorded only as a trace of the review, not as recommended project defaults:

```text
docs/IP2P_BEST_PARAMS_PROVISIONAL.json
```

Conclusion: abandon this SD1.5-IP2P fine-tune as a deliverable. Keep it as an experimental artifact only. The next usable path should prioritize SDXL LoRA img2img/SDEdit at low strength, with optional IP-Adapter/ControlNet only after baseline content preservation is visually acceptable.

Final Mac review can still inspect the sheets, but no per-style best parameter should be promoted from this run without a new, non-smeared validation set.

## SDXL-IP2P Feasibility Check

The official `diffusers v0.38.0` SDXL InstructPix2Pix script was smoke-tested through `scripts/train_ip2p_sdxl_official.py`.

Results:

- 256px, batch 1, gradient checkpointing, fp16: OOM at optimizer state initialization.
- 256px, plus 8-bit Adam and TF32: OOM at optimizer state initialization.
- 128px, plus 8-bit Adam and TF32: OOM at optimizer state initialization.

Conclusion: full-UNet SDXL-IP2P training is not practical on this 12GB Windows GPU. Prefer the completed SD1.5 IP2P fallback, SDXL LoRA img2img, cloud training, or a future parameter-efficient SDXL-IP2P LoRA route.
