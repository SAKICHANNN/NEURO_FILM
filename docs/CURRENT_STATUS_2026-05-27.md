# Current Status - 2026-05-27

This snapshot records the Windows-side work completed after the last pushed commit
`f8b93b4` on `origin/codex/experiment-log`.

## Branch

- Working branch: `codex/windows-baseline-handoff-20260527`
- Base branch/commit: `origin/codex/experiment-log` at `f8b93b4`

## Windows Tasks Completed

- Windows SSH server was installed through Microsoft OpenSSH Preview after the Windows Optional Capability path failed.
- Mac public key was installed into `C:\Users\hhvrf\.ssh\authorized_keys`; final no-password login must still be tested from Mac.
- Data audits were completed:
  - exact SHA256 duplicate audit
  - perceptual duplicate audit
  - manifest rebuild
- SDXL LoRA training wrapper was added and used.
- Missing SDXL film LoRAs were trained on Windows CUDA:
  - `portra_800`
  - `tri_x_400`
  - `velvia_50`
  - `hp5`
- IP2P pseudo-pair datasets were built and combined.
- SD1.5 IP2P fine-tuning completed, but validation showed severe painterly smearing.
- SDXL full-UNet IP2P was smoke-tested and found infeasible on the 12GB GPU due CUDA OOM, even at 128/256px with gradient checkpointing and 8-bit Adam.
- SDXL LoRA + SDEdit was validated:
  - low strength preserves content but is nearly unchanged
  - medium strength produces visible changes but rewrites identity/details
  - this route is not promoted as the default
- A deterministic content-safe color baseline was implemented and validated:
  - `scripts/build_film_color_stats.py`
  - `scripts/pipeline_color_baseline.py`
  - `configs/film_color_stats.json`

## Current Technical Direction

The project should not currently rely on diffusion as the default film translation
path. Both IP2P and SDXL SDEdit showed unacceptable failure modes for content
preservation.

The current best baseline is deterministic Lab color-stat transfer:

- it preserves geometry and identity;
- it produces visible style changes;
- it supports B&W stocks without prompt obedience;
- it is a foundation for later H&D curves, better grain, halation, and optional
  diffusion embellishment.

## Important Artifacts

Large generated data and weights remain ignored by Git and should be transferred
directly from Windows when needed.

Tracked records and helpers:

- `docs/WINDOWS_TASK_TRACKER.md`
- `docs/COLOR_BASELINE_RESULTS.md`
- `docs/SDXL_LORA_VALIDATION_RESULTS.md`
- `docs/IP2P_GRID_SEARCH_RESULTS.md`
- `docs/WINDOWS_ARTIFACT_HANDOFF.md`
- `docs/WINDOWS_ARTIFACT_MANIFEST.json`
- `scripts/verify_windows_artifacts.py`

Ignored local artifacts:

- `loras/*.safetensors`
- `outputs/ip2p_finetune/all_1000`
- `outputs/ip2p_grid/all_1000_portra400_probe`
- `outputs/sdxl_lora_grid`
- `outputs/color_baseline`
- `data/film_domain`
- `data/ip2p_train`

## Remaining Manual Actions

Only these require the user/Mac:

1. Test Mac to Windows no-password SSH:
   `ssh hhvrf@192.168.1.103`
2. Install/sign in to Tailscale on Windows and Mac if remote access outside LAN is needed.
3. Pull large ignored artifacts from Windows using `docs/WINDOWS_ARTIFACT_HANDOFF.md`.
4. Run `python scripts/verify_windows_artifacts.py` on Mac after transfer.
5. Visually validate the deterministic color baseline on real user photos.
