# Scripts Index

This directory intentionally stays flat so command examples in docs and handoff
notes keep working. Use this index to find the right entrypoint without moving
files.

## Primary Rendering

| Script | Purpose |
|--------|---------|
| `render_film.py` | Integrated content-preserving renderer entrypoint. |
| `pipeline_color_baseline.py` | Deterministic safe Lab color baseline and safe-rich preset. |
| `pipeline_filmfx_layers.py` | Deterministic film-effects layer pipeline. |
| `pipeline.py` | Legacy/general pipeline entrypoint. |
| `translate.py` | Older translation entrypoint. |

## Color Engine Research

| Script | Purpose |
|--------|---------|
| `evaluate_color_pipeline.py` | Evaluate the safe-rich color baseline on the seed set. |
| `compare_color_engines.py` | Compare challenger summaries against the safe-rich baseline. |
| `evaluate_local_color_maps.py` | Evaluate Local/Semantic bounded color maps. |
| `evaluate_local_color_map_strength_sweep.py` | Render multiple local-map strengths in one pass. |
| `evaluate_film_response_volume.py` | Evaluate stock-specific film response volume experiments. |
| `train_neural_lut.py` | Train Neural LUT imitation experiments. |
| `evaluate_neural_lut.py` | Evaluate trained Neural LUT checkpoints. |
| `build_neural_film_targets.py` | Build Neural Film LUT V2 pseudo-target images. |
| `fit_distilled_film_lut.py` | Fit the distilled Scheme A SepLUT candidate. |
| `evaluate_distilled_film_lut.py` | Evaluate the distilled Scheme A SepLUT candidate. |
| `fit_distilled_neural_variants.py` | Fit distilled Scheme B NILUT and Scheme C context-4D candidates. |
| `evaluate_distilled_neural_variants.py` | Evaluate distilled Scheme B/C candidates. |
| `summarize_neural_lut_results.py` | Summarize Neural Film LUT V2 candidate metrics. |
| `train_neural_film_seplut.py` | Torch scaffold for Scheme A Style-Separated SepLUT. |
| `evaluate_neural_film_seplut.py` | Evaluate torch Scheme A SepLUT checkpoints. |
| `train_lut.py` | Older LUT training script. |

## Safety, Audit, And Contact Sheets

| Script | Purpose |
|--------|---------|
| `evaluate_render_safety.py` | Shared clipping, structure, HF, and color safety metrics. |
| `audit_color_baseline.py` | Audit deterministic color baseline outputs. |
| `list_eval_sources.py` | List/evaluate source images for seed evals. |
| `make_eval_contact_sheet.py` | Build contact sheets from evaluation outputs. |
| `make_rawpixls_velvia_preview.py` | Rawpixls/Velvia preview helper. |
| `verify_windows_artifacts.py` | Verify local Windows artifacts. |

## Film Effects

| Script | Purpose |
|--------|---------|
| `smoke_filmfx_layers.py` | Smoke test film-effects layer rendering. |
| `smoke_filmfx_effects.py` | Smoke test lower-level film-effects functions. |
| `evaluate_physical_halation_v2.py` | Evaluate physical-prior halation V2/V2.3 expert, locked-control, or rule-family sweeps and contact sheets. |
| `evaluate_halation_physics_suite.py` | Run synthetic halation radius, hue, background, and blue-leakage checks. |
| `export_halation_gui_schema.py` | Export machine-readable locked halation GUI schema, presets, valid combinations, and slider contract. |
| `build_halation_real_photo_manifest.py` | Build a license-aware Commons/citation manifest and optional ignored local analysis cache for real-photo halation validation. |
| `mine_halation_patches.py` | Mine candidate real-photo halation patches, metrics, and review contact sheets from the manifest. |
| `evaluate_halation_real_photo_alignment.py` | Compare mined display-level patch statistics with current physical halation presets and write alignment contact sheets/reports. |
| `smoke_chroma_residual.py` | Smoke test bounded chroma residual module. |

## Data And Stats

| Script | Purpose |
|--------|---------|
| `download_data.py` | Download configured datasets. |
| `build_fivek_auto_optimize_cache.py` | Build compact paired FiveK Expert C cache for future auto-optimization layer experiments. |
| `build_fivek_raw_cache.py` | Build RAW/default render to Expert C FiveK cache smoke assets from the local FiveK tar. |
| `evaluate_fivek_response_baseline.py` | Render/evaluate deterministic FiveK response baselines, including tone-locked and legacy RGB-delta modes. |
| `extract_fivek_response_stats.py` | Extract compact tone/chroma response statistics from a FiveK RAW-derived cache. |
| `inspect_image_input.py` | Inspect raster/RAW metadata for the shared preprocessing pipeline. |
| `scrape_films.py` | Film reference scraping helper. |
| `build_data_manifest.py` | Build local dataset manifest. |
| `build_film_color_stats.py` | Build per-stock Lab color statistics. |
| `build_ip2p_dataset.py` | Build InstructPix2Pix dataset artifacts. |
| `combine_ip2p_dataset.py` | Combine IP2P dataset shards. |
| `download_loras.py` | Download LoRA files. |

## Diffusion, LoRA, And Model Training

| Script | Purpose |
|--------|---------|
| `infer_lora.py` | LoRA inference helper. |
| `grid_search_ip2p.py` | IP2P grid search. |
| `grid_search_sdxl_lora.py` | SDXL LoRA grid search. |
| `train_ip2p.py` | IP2P training script. |
| `train_ip2p_official.py` | Official IP2P training variant. |
| `train_ip2p_sdxl_official.py` | Official SDXL IP2P training variant. |
| `train_lora.py` | LoRA training script. |
| `train_lora_sd.py` | Stable Diffusion LoRA training variant. |
| `train_sdxl_lora.py` | SDXL LoRA training script. |
| `train_unet.py` | UNet training script. |
| `smoke_neural_lut.py` | Smoke test Neural LUT operators. |

## Windows Setup And Remote Access

| Script | Purpose |
|--------|---------|
| `setup_win.ps1` | Windows setup helper. |
| `setup_win_ssh.ps1` | Windows SSH setup helper. |
| `install_openssh_preview_server.ps1` | Install OpenSSH preview server. |
| `run_setup_win_ssh_as_admin.bat` | Admin launcher for SSH setup. |
| `run_install_openssh_preview_server_as_admin.bat` | Admin launcher for OpenSSH install. |

## Notes

- Generated outputs belong under `outputs/` and should remain ignored.
- Local datasets, weights, caches, and credentials must not be committed.
- Before moving any script, update all references in `docs/`, `AGENTS.md`,
  `agent_reminder/`, and command examples.
