# Windows Task Tracker

> **Historical machine-task snapshot.** Active cross-platform work is tracked in `docs/ULTIMATE_EXECUTION_TRACKER.md` as of 2026-07-10.

> Purpose: track all tasks assigned to the Windows RTX 5070 Ti machine by the latest 10 commits.
> Last updated: 2026-05-26

## Machine Role

Windows is the remote CUDA training and inference machine. Mac is the development and verification machine.

Daily loop:

```text
Mac develops code -> GitHub -> Windows pulls -> Windows trains/runs CUDA jobs -> Windows pushes weights/results -> Mac verifies
```

Known Windows workspace:

```text
C:\Users\hhvrf\Documents\neuro_film
```

Known LAN SSH target:

```text
hhvrf@192.168.1.103
```

## Status Legend

| Status | Meaning |
|--------|---------|
| done | Verified on this Windows machine |
| partial | Some pieces exist, but follow-up is required |
| pending | Not started or not verified |
| blocked | Cannot proceed until a dependency is resolved |
| manual | Requires an action from Mac or the user account owner |

## Dependency Order

This order is the execution contract for the Windows machine. Work from top to bottom; skip only blocked Mac-side confirmations and continue with the next unblocked item.

Official references checked before implementation:

- OpenSSH key auth on Windows: <https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement>
- Diffusers SDXL LoRA script reference: <https://github.com/huggingface/diffusers/blob/v0.38.0/examples/text_to_image/train_text_to_image_lora_sdxl.py>
- Diffusers InstructPix2Pix training guide: <https://huggingface.co/docs/diffusers/v0.38.0/en/training/instructpix2pix>

Implementation note: the current `main` branch SDXL LoRA example requires `diffusers 0.39.0.dev0`; this repo is locked to `diffusers 0.38.0`, so Windows uses the `v0.38.0` example script.

| Order | Task | Depends On | Status | Completion Test |
|:---:|------|------|:---:|------|
| 0 | Keep repo synced with GitHub | Git remote access | done | `HEAD == origin/codex/experiment-log` |
| 1 | Windows SSH server reachable from Mac | SSH server, firewall, Windows account | done | `ssh hhvrf@192.168.1.103` reaches Windows |
| 2 | Mac key auth confirmed | Order 1, Mac private key, Windows admin authorized keys | manual | Windows has the Mac public key and correct permissions; Mac must run `ssh hhvrf@192.168.1.103` without password |
| 3 | CUDA runtime smoke check | Order 1, `.venv`, PyTorch CUDA wheel | done | `torch.cuda.is_available()` is true; RTX 5070 Ti detected |
| 4 | Core model cache verification | Order 3, network/HF access | done | SDXL and IP2P load from local `from_pretrained` cache |
| 5 | Dataset exact duplicate audit | Film data restored | done | SHA256 duplicate groups are zero or documented |
| 6 | Dataset perceptual duplicate audit | Film data restored, Pillow/Numpy | done | dHash report generated at `data/processed/perceptual_duplicate_report.json` |
| 7 | Training manifest rebuild | Orders 5-6, duplicate policy | done | `data/processed/manifest.jsonl` reflects current data |
| 8 | SDXL LoRA training script | Orders 3-4, manifest | done | `scripts/train_sdxl_lora.py --help` works |
| 9 | SDXL LoRA smoke training | Order 8 | done | 1-step Portra 400 CUDA run wrote `outputs/smoke/portra_400_reboot_smoke/pytorch_lora_weights.safetensors` |
| 10 | Full SDXL film LoRA training | Order 9 | done | `portra_800`, `tri_x_400`, `velvia_50`, and `hp5` trained and copied to `loras/` |
| 11 | IP2P dataset builder | Order 7 | done | `data/ip2p_train/*` has train/val pseudo-pairs and imagefolder-compatible metadata |
| 12 | IP2P / SDXL-IP2P fine-tuning | done | SD1.5 IP2P fine-tune completed but grid review shows severe smearing, so it is archived as a failed experiment; SDXL full-UNet IP2P OOMs on 12GB even at 128/256 smoke with 8-bit Adam |
| 13 | Grid search script | Orders 4 and 7 | done | `scripts/grid_search_ip2p.py --help` works |
| 14 | Hyperparameter search | done | 72-image grid and contact sheets generated; review found severe smearing, so no IP2P settings are promoted |
| 15 | SDXL LoRA img2img validation | Orders 10 and 13 | done | Low strength is nearly unchanged; medium strength changes content, so do not promote as default |
| 16 | Deterministic color baseline | Order 7 | done | `scripts/pipeline_color_baseline.py` preserves geometry and produces visible color/B&W changes; see `docs/COLOR_BASELINE_RESULTS.md` |
| 17 | Mac validation loop | Any trained weights/results | manual | Mac pulls results, runs `scripts/verify_windows_artifacts.py`, and records A/B verdict |

Parallel-safe tasks:

- Orders 3-4 can be checked from Windows locally before Mac-side key auth is confirmed.
- Orders 5-7 can proceed once transfer is complete, even before training code exists.
- Orders 8 and 11 can be implemented in parallel after the manifest schema is stable.

## 1. Remote Access

| Task | Status | Evidence / Next Action |
|------|:---:|------|
| Install SSH server on Windows | done | `sshd` is running via Microsoft OpenSSH Preview, not Windows Optional Capability |
| Enable SSH server autostart | done | `sshd` StartType is `Automatic` |
| Open firewall for TCP/22 | done | `Test-NetConnection localhost -Port 22` succeeds |
| Mac -> Windows password SSH | done | Windows accepts SSH on `192.168.1.103:22` |
| Mac -> Windows key auth | manual | Mac public key is installed in `C:\Users\hhvrf\.ssh\authorized_keys`; final no-password login must be tested from Mac |
| Remote/VPN access outside LAN | manual | Tailscale is recommended without Exit Node; Windows CLI/MSI install attempts could not complete without interactive elevation/login; see `docs/REMOTE_ACCESS_MANUAL_STEPS.md` |

Notes:

- Windows built-in `OpenSSH.Server~~~~0.0.1.0` failed due to DISM/CBS component corruption (`0x80073712`, later repair `0x800f0915`).
- Current working server was installed through Microsoft OpenSSH Preview MSI.

## 2. Windows Environment

| Task | Status | Evidence / Next Action |
|------|:---:|------|
| Python 3.12 virtualenv | done | `.venv` exists and was previously verified as Python 3.12.10 |
| PyTorch CUDA environment | done | Previously verified: `torch 2.11.0+cu128`, CUDA available, RTX 5070 Ti visible |
| Core diffusion dependencies | done | Previously verified: `diffusers 0.38.0`, `accelerate 1.13.0`, `peft 0.19.1`, `safetensors 0.8.0-rc.0` |
| CUDA smoke check after latest sync | done | `torch 2.11.0+cu128`; CUDA OK on NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| xformers/Blackwell fallback check | done | Model loading and CUDA inference work without xformers/Triton; avoid relying on Triton-only optimizations |

## 3. Data And Local Assets

| Task | Status | Evidence / Next Action |
|------|:---:|------|
| `.env` with Flickr/Civitai credentials | done | Local `.env` exists; do not commit it |
| Flickr film-domain data restored | done | All 8 film directories meet or exceed documented targets |
| Check for exact duplicate files | done | SHA256 duplicate audit found no exact duplicate groups during transfer |
| Perceptual duplicate audit | done | dHash threshold 4 report generated; near-duplicate pairs are recorded but not deleted |
| Rebuild training manifest | done | `manifest.jsonl` has 4210 accepted records; 2 oversized images rejected from training |
| Physics PDFs | done | `data/physics/` PDFs were verified as valid `%PDF-` files |
| CIE calibration data | done | `data/calibration/cie/` files exist |

Current `data/film_domain/` counts:

| Film | Current | Target | Delta |
|------|:---:|:---:|:---:|
| `ektar_100` | 534 | 499 | +35 |
| `hp5` | 501 raw / 500 accepted | 500 | 0 accepted |
| `portra_400` | 689 | 500 | +189 |
| `portra_800` | 500 | 500 | 0 |
| `tri_x_400` | 501 raw / 500 accepted | 500 | 0 accepted |
| `velvia_50` | 384 | 384 | 0 |
| `vision3_250d` | 539 | 403 | +136 |
| `vision3_500t` | 564 | 500 | +64 |

Duplicate policy:

- Exact SHA256 duplicates: none found.
- Perceptual dHash near-duplicates: record only; do not delete automatically.
- Oversized Pillow `DecompressionBombError` images: reject from manifest, keep source files on disk for manual review.

## 4. LoRA And Model Weights

| Task | Status | Evidence / Next Action |
|------|:---:|------|
| SD1.5 self-trained film LoRA set | done | 8 `*_sd15.safetensors` files exist in `loras/` |
| SDXL community LoRA: Portra 400 | done | `loras/portra_400.safetensors` exists |
| SDXL community LoRA: Vision3 500T | done | `loras/vision3_500t.safetensors` exists |
| SDXL community LoRA: Vision3 250D | done | `loras/vision3_250d.safetensors` exists |
| SDXL community LoRA: Ektar 100 | done | `loras/ektar_100.safetensors` exists |
| SDXL self-trained LoRA: Portra 800 | done | `loras/portra_800.safetensors`; 3000 steps, rank 64, 512px |
| SDXL self-trained LoRA: Tri-X 400 | done | `loras/tri_x_400.safetensors`; 3000 steps, rank 64, 512px |
| SDXL self-trained LoRA: Velvia 50 | done | `loras/velvia_50.safetensors`; 3000 steps, rank 64, 512px |
| SDXL self-trained LoRA: HP5 | done | `loras/hp5.safetensors`; 3000 steps, rank 64, 512px |
| HF film photography style LoRA | done | Downloaded from `cinquecentoiso/film-photography-style` to `loras/film_photography_style.safetensors`; license metadata not provided upstream |
| HF film grain LoRA | done | Downloaded SDXL FilmGrainRedmond from `artificialguybr/filmgrain-redmond-filmgrain-lora-for-sdxl` to `loras/film_grain.safetensors`; bespoke LoRA license, review before publication |
| SDXL base model cache | done | `StableDiffusionXLPipeline.from_pretrained(..., local_files_only=True)` succeeds |
| InstructPix2Pix model cache | done | `StableDiffusionInstructPix2PixPipeline.from_pretrained(..., local_files_only=True)` succeeds |

## 5. First Verification Checklist

Run after SSH login from Mac:

```powershell
cd C:\Users\hhvrf\Documents\neuro_film
.venv\Scripts\activate

python -c "import torch; assert torch.cuda.is_available(); props=torch.cuda.get_device_properties(0); print(f'CUDA OK: {torch.cuda.get_device_name(0)} ({props.total_memory/1e9:.1f}GB)')"
```

Then confirm model loading:

```powershell
python -c "
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionInstructPix2PixPipeline
print('Downloading/loading SDXL...')
sdxl = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, use_safetensors=True)
print('Downloading/loading IP2P...')
ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None)
print('All models available')
"
```

| Task | Status | Evidence / Next Action |
|------|:---:|------|
| CUDA check through SSH session | done | Verified locally on Windows; Mac SSH invocation still can repeat same command |
| SDXL model load/download | done | Local cache load succeeds |
| IP2P model load/download | done | Local cache load succeeds |
| Quick CUDA inference test | done | `scripts/pipeline.py` loaded `loras/portra_800.safetensors` on CUDA and wrote `outputs/smoke/sdxl_lora_portra800_smoke.jpg` |
| SDXL LoRA quality validation | done | Validation found the strength tradeoff unacceptable: low strength is nearly unchanged, medium strength rewrites content; see `docs/SDXL_LORA_VALIDATION_RESULTS.md` |
| Deterministic color baseline validation | done | Lab-stat color baseline produces visible changes without geometry edits; see `docs/COLOR_BASELINE_RESULTS.md` |

## 6. Training Tasks

These are the main tasks assigned to Windows by the latest commit.

| Priority | Task | Status | Notes |
|:---:|------|:---:|------|
| P0 | Implement or add `scripts/train_sdxl_lora.py` | done | Wrapper prepares `metadata.jsonl` and calls the pinned official diffusers v0.38.0 SDXL LoRA script |
| P0 | Train SDXL full UNet LoRA for each film | done | Missing SDXL styles trained: `portra_800`, `tri_x_400`, `velvia_50`, `hp5`; final weights copied into `loras/` |
| P1 | Build IP2P instruction fine-tuning dataset | done | `scripts/build_ip2p_dataset.py --all --max-per-style 200` generated 8 styles x 200 pseudo-pairs |
| P1 | Fine-tune SD1.5 InstructPix2Pix | done | `outputs/ip2p_finetune/all_1000` completed and reloads locally, but visual grid review shows severe smearing; archive as failed experiment |
| P1 | Train SDXL InstructPix2Pix | blocked | Official SDXL script loads SDXL Base and data, but full-UNet training OOMs on RTX 5070 Ti 12GB at optimizer init even with `--use_8bit_adam`, `--gradient_checkpointing`, and 128px smoke |
| P1 | Implement or add `scripts/grid_search_ip2p.py` | done | Dry-run verified on a Portra 400 image |
| P1 | Run IP2P hyperparameter grid search | done | 72 outputs and 8 contact sheets generated; no best picks promoted because outputs are visibly smeared/unusable |
| P1 | Validate SDXL LoRA SDEdit quality | done | Low-strength grid was too weak; medium-strength grid rewrote identity/details; archive as experimental |
| P1 | Implement deterministic color baseline | done | `scripts/build_film_color_stats.py`, `scripts/pipeline_color_baseline.py`, `configs/film_color_stats.json` |
| P1 | Validate trained weights on Mac | manual | Mac pulls weights/results, verifies `docs/WINDOWS_ARTIFACT_MANIFEST.json`, and runs subjective/quantitative checks |

Suggested SDXL LoRA target from `docs/WIN_REMOTE_SETUP.md`:

```text
rank 64
3000 steps
512x512
CUDA FP16
one .safetensors LoRA per film
```

## 7. Near-Term Execution Order

1. Confirm Mac can SSH into Windows without password.
2. Run CUDA and model-loading verification from the Mac SSH session.
3. Decide duplicate policy for over-target film directories.
4. Rebuild `data/processed/manifest.jsonl`. done
5. Add/implement `scripts/train_sdxl_lora.py`. done
6. Run one short SDXL LoRA smoke training on one film, preferably `portra_400`. done
7. If smoke succeeds, train priority films: `portra_800`, `tri_x_400`, `velvia_50`, `hp5`. done
8. Implement IP2P dataset preparation and grid search scripts. done
9. Run IP2P fine-tuning experiments. SD1.5 archived as failed due smearing; SDXL full-UNet blocked by 12GB VRAM.
10. Validate SDXL LoRA img2img/SDEdit. done; not promoted due unchanged/rewritten-content tradeoff.
11. Establish deterministic content-safe color baseline. done.
12. Transfer large weights/results back to Mac with the commands in `docs/WINDOWS_ARTIFACT_HANDOFF.md`.

## 8. Files Mentioned By The Latest 10 Commits

Primary Windows handoff files:

- `docs/WIN_REMOTE_SETUP.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/MAC_ARM_MIGRATION.md`
- `docs/DATA_REPRODUCTION_MANIFEST.json`
- `TASK_BOARD.md`
- `IMPL_PLAN.md`
- `AGENTS.md`
- `scripts/setup_win.ps1`
- `scripts/download_data.py`
- `scripts/download_loras.py`
- `scripts/pipeline.py`
- `scripts/scrape_films.py`
- `scripts/train_ip2p.py`
- `scripts/train_lora_sd.py`
- `scripts/translate.py`

Locally added helper files for this Windows setup:

- `scripts/setup_win_ssh.ps1`
- `scripts/run_setup_win_ssh_as_admin.bat`
- `scripts/install_openssh_preview_server.ps1`
- `scripts/run_install_openssh_preview_server_as_admin.bat`
- `scripts/build_data_manifest.py`
- `scripts/train_sdxl_lora.py`
- `scripts/build_ip2p_dataset.py`
- `scripts/combine_ip2p_dataset.py`
- `scripts/train_ip2p_official.py`
- `scripts/train_ip2p_sdxl_official.py`
- `scripts/grid_search_ip2p.py`
- `scripts/grid_search_sdxl_lora.py`
- `scripts/build_film_color_stats.py`
- `scripts/pipeline_color_baseline.py`
- `configs/film_color_stats.json`
- `docs/IP2P_GRID_SEARCH_RESULTS.md`
- `docs/IP2P_BEST_PARAMS_PROVISIONAL.json`
- `docs/SDXL_LORA_VALIDATION_RESULTS.md`
- `docs/COLOR_BASELINE_RESULTS.md`
- `docs/WINDOWS_ARTIFACT_HANDOFF.md`
- `docs/WINDOWS_ARTIFACT_MANIFEST.json`
- `docs/REMOTE_ACCESS_MANUAL_STEPS.md`
- `scripts/verify_windows_artifacts.py`
- `.env.example`
