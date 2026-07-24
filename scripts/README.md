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
| `run_u5_r2a_constrained_operator_audit.py` | Audit the reusable constrained affine/spline/tetrahedral-LUT representation. |
| `run_u2_3a_smooth_residual_composition.py` | Audit exact identity and one frozen bounded nontrivial tetrahedral residual after the U2.2B base. |
| `run_u5_r2i0_sensitometry_print_frontier.py` | Render/evaluate the fixed U2.2B operator along one bounded strength path on the frozen gold/stress set. |
| `run_u5_r2b_global_operator_frontier.py` | Audit the frozen fixed global-policy bank and build survivor-only blind sheets. |
| `audit_real_film_prov_register_recon.py` | Run the bounded metadata-only PROV negative-register accessibility audit. |
| `run_roll2film_e0.py` | Run the data-independent known-operator/group-size/shuffled-control Roll2Film identifiability gate. |
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
| `evaluate_fivek_response_baseline.py` | Render/evaluate deterministic FiveK response baselines, including tone-locked, WB-anchored, chroma-guarded, and legacy RGB-delta modes. |
| `extract_fivek_response_stats.py` | Extract compact tone/chroma response statistics from a FiveK RAW-derived cache. |
| `extract_fivek_freeze_response_stats.py` | Extract response statistics from high-precision FiveK freeze-pack TIFFs. |
| `build_fivek_filtered_targets.py` | Build conservative FiveK pseudo-targets that keep Expert C tone while guarding WB and chroma. |
| `build_fivek_freeze_pack.py` | Build bounded high-precision FiveK freeze assets before deleting the large raw dataset. |
| `download_real_film_yfcc_stock_metadata.py` | Freeze and resume the bounded YFCC15M metadata-only stock-discovery subset. |
| `audit_real_film_yfcc_stock_metadata.py` | Scan frozen YFCC15M metadata for exact stock phrases under licence and UID gates. |
| `download_real_film_yfcc_velvia50.py` | Reverify live CC-BY pages and download the bounded Velvia50 pixel pilot. |
| `audit_real_film_yfcc_velvia50.py` | Run pixel integrity/source gates and render Velvia50 contact sheets. |
| `audit_real_film_yfcc_matched_stock_metadata.py` | Test prospective YFCC same-source support for Ektar100 and UltraMax400. |
| `run_real_film_connected_stock_identifiability.py` | Run the four-cell author-held-out stock/source/content/colour diagnostic. |
| `download_real_film_yfcc_full_index.py` | Resume and hash the frozen 65.64GB full YFCC100M metadata SQLite. |
| `audit_real_film_yfcc_full_index.py` | Filter full YFCC metadata for exact stocks and shared-author support. |
| `decide_real_film_yfcc_full_index.py` | Require two identical full-index audits and apply the SF1.1 pass/close branch. |
| `repair_yfcc_full_index_range.py` | Repair a rejected full-index byte interval through validated fixed Range chunks. |
| `audit_real_film_yfcc_shared_author_rights.py` | Run the SF1.2 bounded Flickr-page-only live-rights feasibility gate. |
| `audit_real_film_apollo7_metadata.py` | Run the SF2.0A bounded NASA/JSC Apollo 7 stock-magazine metadata feasibility gate; never requests image payloads. |
| `audit_real_film_nasa_sts098_snapshot.py` | Run the SF2.0B0 keyless NASA/JSC STS098 exact-stock result-table snapshot; never requests photo pages or images. |
| `download_real_film_yfcc_shared_author_pixels.py` | Acquire only the frozen SF1.3A shared-author Ektar/Velvia derivatives. |
| `audit_real_film_yfcc_shared_author_pixels.py` | Verify SF1.3A hashes, decodes, duplicates and shared-author support, then render contact sheets. |
| `inspect_image_input.py` | Inspect raster/RAW metadata for the shared preprocessing pipeline. |
| `scrape_films.py` | Film reference scraping helper. |
| `build_data_manifest.py` | Build local dataset manifest. |
| `audit_filmcase_manifest.py` | Read-only legacy-manifest audit that writes quarantined FilmCase manifest-v2 and duplicate evidence under ignored `outputs/filmcase/`; never modifies source data. |
| `verify_reproducibility_baseline.py` | Verify tracked checksums/tests/benchmark registries and optionally write an ignored local environment report; never renders or reads image assets. |
| `freeze_filmcase_evaluation_set.py` | Freeze hashes and split membership from the local union source index into an ignored provisional FilmCase artifact-evaluation set; never copies or renders images. |
| `build_filmcase_blind_audit.py` | Create a three-round anonymous FilmCase review sheet, anonymous copied assets and a private ignored label map; it records no score and never alters source images. Quote comma-separated PowerShell arguments. |
| `aggregate_filmcase_blind_audit.py` | Aggregate existing raw blind-review JSONL against the private label mapping into an ignored fail-closed report; never generates review scores. |
| `make_filmcase_blind_contact_sheet.py` | Make a review-only contact sheet from public anonymous assets for one blind round; never reads the private label map or alters inputs. |
| `diagnose_filmcase_chroma_speckle.py` | Write diagnostic-only high-frequency chroma-island metrics for a before/after pair to guide full-resolution artifact review; never makes a pass/fail decision. |
| `render_filmcase_anchor_set.py` | Re-render the five owner anchors plus a bland control as color-only, same-input gold replays under ignored `outputs/filmcase/`; defaults to gold only and records input/output hashes. |
| `build_film_color_stats.py` | Build per-stock Lab color statistics. |
| `run_roll2film_e0.py` | Run the preserved variable-sample affine E0 foundation experiment. |
| `run_roll2film_e0_fixed_budget.py` | Run fixed-total-pixel affine controls for partition, nuisance boundaries, independent support, mixed operators, prior sensitivity, and scanner confounding. |
| `run_roll2film_e0_l2_fixed_budget.py` | Run fixed-total-pixel affine-plus-monotone-spline controls with exposure/WB/scene nuisance, prior, scanner, Jacobian, and LUT-bake diagnostics. |
| `audit_roll2film_filmset.py` | Hash the local FilmSet tree with a resumable cache and write isolated pair-blind source/target, internal-dev, and sealed final-628 manifests under ignored outputs. |
| `build_roll2film_ct5_cache.py` | Verify frozen CT5 manifest/payload hashes and build deterministic equal-image linear-sRGB training caches; internal-dev decoding requires an explicit evaluator flag. |
| `run_roll2film_ct5_pilot.py` | Fit the frozen CPU baseline ladder from unpaired CT5 training caches and evaluate only the internal pilot fold; confirmatory and final-628 data are not loaded. |
| `run_roll2film_ct5_confirmatory.py` | Replay frozen pilot operator bundles on the untouched internal confirmatory fold and run cluster-bootstrap best-basic comparisons without loading final-628 data. |
| `run_roll2film_ct5_fullres.py` | Render frozen CT5 finalists on all confirmatory images, compute full-resolution diagnostics, and save deterministic worst-case review sheets; automatic metrics never clear the severe gate. |
| `run_u5_r2a_constrained_operator_audit.py` | Audit the versioned affine/spline/tetrahedral-LUT numerical contract and its required semantic non-safety counterexample without fitting film pixels. |
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
