# Windows Artifact Handoff

> Last updated: 2026-05-26. Windows workspace: `C:\Users\hhvrf\Documents\neuro_film`.

## Pull From Mac

Run on Mac after SSH access to Windows is available:

```bash
cd ~/neuro_film

scp hhvrf@192.168.1.103:C:/Users/hhvrf/Documents/neuro_film/docs/WINDOWS_ARTIFACT_MANIFEST.json docs/
scp -r hhvrf@192.168.1.103:C:/Users/hhvrf/Documents/neuro_film/loras .
scp -r hhvrf@192.168.1.103:C:/Users/hhvrf/Documents/neuro_film/outputs/ip2p_finetune/all_1000 outputs/ip2p_finetune/
scp -r hhvrf@192.168.1.103:C:/Users/hhvrf/Documents/neuro_film/outputs/ip2p_grid/all_1000_portra400_probe outputs/ip2p_grid/
scp -r hhvrf@192.168.1.103:C:/Users/hhvrf/Documents/neuro_film/outputs/color_baseline outputs/
scp -r hhvrf@192.168.1.103:C:/Users/hhvrf/Documents/neuro_film/outputs/sdxl_lora_grid outputs/

python scripts/verify_windows_artifacts.py
```

`rsync` is not installed on this Windows machine, so `scp` is the portable default.

## High-Value Artifacts

SDXL LoRA weights:

- `loras/portra_400.safetensors`
- `loras/portra_800.safetensors`
- `loras/vision3_500t.safetensors`
- `loras/vision3_250d.safetensors`
- `loras/ektar_100.safetensors`
- `loras/tri_x_400.safetensors`
- `loras/velvia_50.safetensors`
- `loras/hp5.safetensors`

IP2P fine-tune:

- `outputs/ip2p_finetune/all_1000`

IP2P grid review:

- `outputs/ip2p_grid/all_1000_portra400_probe`
- `outputs/ip2p_grid/all_1000_portra400_probe/contact_sheets`
- `docs/IP2P_GRID_SEARCH_RESULTS.md`
- `docs/WINDOWS_ARTIFACT_MANIFEST.json`
- `scripts/verify_windows_artifacts.py`

Deterministic color baseline:

- `configs/film_color_stats.json`
- `scripts/build_film_color_stats.py`
- `scripts/pipeline_color_baseline.py`
- `docs/COLOR_BASELINE_RESULTS.md`
- `outputs/color_baseline`

Failed SDXL LoRA validation:

- `scripts/grid_search_sdxl_lora.py`
- `docs/SDXL_LORA_VALIDATION_RESULTS.md`
- `outputs/sdxl_lora_grid`

## GitHub Note

The generated model weights are large enough to require Git LFS or direct machine-to-machine transfer. Direct `scp` over SSH is the safer default for the current Mac/Windows loop; commit only docs and scripts through normal Git unless a separate Git LFS quota decision is made.
