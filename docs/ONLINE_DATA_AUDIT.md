# Online Data Audit — 2026-05-25

> Scope: reconciled current docs with git history and live source checks. This file is the short source-of-truth for data that was previously marked as possible, missing, or placeholder.

## Git History Reconciliation

| Commit | Fact to carry forward |
|------|------|
| `80d058c` | Added Flickr scraper, SD 1.5 LoRA training/inference scripts, and recorded 3,896 Flickr film-domain images across 8 stocks. The trained LoRA files and Flickr images are ignored and are not present in this checkout. |
| `339a011` | Added per-stock Flickr download commands and counts to `docs/EXPERIMENT_LOG.md`. |
| `458b99e` | Added Windows data directory checklist; this describes a populated Windows/Mac workspace, not necessarily the current git checkout. |

Current checkout status on this machine: `data/film_domain/` and `loras/` are missing; `data/physics/` and `data/calibration/` contain restored local files but are ignored by git.

## Civitai LoRA Verification

Checked with `https://civitai.com/api/v1/models/{id}` and query searches on 2026-05-25.

| Stock | SDXL status | Civitai model/version | Notes |
|------|------|------|------|
| Kodak Portra 400 | Verified | model `723250`, version `808680`, file `Porta400_xl.safetensors` | API reports `allowCommercialUse={Sell}`, derivatives allowed. |
| Kodak Vision3 500T | Verified | model `725625`, SDXL version `820808` | Creator license is restrictive: API reports `allowCommercialUse={RentCivit}`, derivatives not allowed. |
| Kodak Vision3 250D | Verified | model `725620`, SDXL version `820761` | Corrects the earlier accidental reuse of the 500T model id. Same creator/license pattern as 500T. |
| Kodak Ektar 100 | Verified | model `779013`, SDXL version `1167852` | API reports `allowCommercialUse={RentCivit}`, derivatives allowed. |
| Kodak Portra 800 | Not verified for exact SDXL | Flux-only hit found: model `669785`; no exact SDXL stock model confirmed | Treat as self-trained for SDXL until a precise SDXL model is verified. |
| Kodak Tri-X 400 | Not verified for exact SDXL | Previous id `521049` is unrelated | Treat as self-trained for SDXL. |
| Fujifilm Velvia 50 | Not verified for exact SDXL | Search found Fuji generic/C100-C200 LoRAs, not Velvia 50 | Treat as self-trained. |
| Ilford HP5 Plus | No exact SDXL | model `1941507` is generic Ilford analog/film on SD 1.5 | Treat as self-trained for SDXL. |

## Flickr Film-Domain Data

Recorded by `docs/EXPERIMENT_LOG.md` from the `80d058c` workflow:

| Stock | Target directory | Count |
|------|------|:---:|
| Kodak Portra 400 | `data/film_domain/portra_400/` | 500 |
| Kodak Portra 800 | `data/film_domain/portra_800/` | 500 |
| Kodak Vision3 500T | `data/film_domain/vision3_500t/` | 500 |
| Kodak Vision3 250D | `data/film_domain/vision3_250d/` | 403 |
| Kodak Ektar 100 | `data/film_domain/ektar_100/` | 499 |
| Kodak Tri-X 400 | `data/film_domain/tri_x_400/` | 500 |
| Fujifilm Velvia 50 | `data/film_domain/velvia_50/` | 384 |
| Ilford HP5 Plus | `data/film_domain/hp5/` | 500 |

Restore with `scripts/scrape_films.py` and a local `.env` containing Flickr credentials. Do not commit downloaded images.

For Civitai private/subscriber downloads, `scripts/download_loras.py` reads `CIVITAI_API_TOKEN` from `.env` or the process environment and passes it without printing the token value.

## Physics PDF Gaps

The Fujifilm US Velvia 50 support page lists the Professional Film Data Guide, Velvia 50 Product Information Bulletin, and Fujichrome Velvia RVP PDF. The Velvia 100 brochure is available from Fujifilm assets. These sources close the manifest gap for `fujifilm_velvia_rvp/technical_data.pdf` and `fujifilm_velvia_100/brochure.pdf`.

## Dependency Version Check

PyPI JSON checked on 2026-05-25:

| Package | Project lock | PyPI latest observed |
|------|------|------|
| torch | `2.11.0` / `2.11.0+cu128` | `2.12.0` |
| torchvision | `0.26.0` / `0.26.0+cu128` | `0.27.0` |
| diffusers | `0.38.0` | `0.38.0` |
| accelerate | `1.13.0` | `1.13.0` |
| transformers | `5.9.0` | `5.9.0` |
| peft | `0.19.1` | `0.19.1` |
| safetensors | `0.8.0rc0` | `0.7.0` stable / `0.8.0rc0` required by diffusers 0.38.0 |
| kornia | `0.8.2` | `0.8.3` |
| filmgrainer | git install only | not published on PyPI |

Recommendation: keep the existing lock files for reproducibility and update only after a smoke test on CUDA and MPS.

## Sources

- Civitai API: `https://civitai.com/api/v1/models/723250`, `725620`, `725625`, `779013`, `1941507`.
- Civitai search API: `https://civitai.com/api/v1/models?query=...&types=LORA`.
- Fujifilm Velvia 50 support: `https://www.fujifilm.com/us/en/business/professional-photography/film/velvia-50/support`.
- Fujifilm Velvia 100 brochure: `https://asset.fujifilm.com/www/us/files/2020-03/226db49cde8b549d243c6498a80479cc/Velvia100_Brochure_Final.pdf`.
- Hugging Face diffusers docs: img2img strength and IP-Adapter guides.
- PyPI JSON endpoints: `https://pypi.org/pypi/<package>/json`.
- filmgrainer GitHub: `https://github.com/larspontoppidan/filmgrainer`.
