# FiveK Auto-Optimize Umbrella Tracker

> Created: 2026-06-13
>
> Split: 2026-06-15
>
> Branch: `research/fivek-auto-optimize-cache`
>
> Purpose: keep the business boundary, cross-tracker dependencies, and FiveK
> deletion gate in one place. Detailed implementation plans now live in the
> focused trackers linked below.

## 1. Why This Was Split

The original tracker became too broad. It mixed four related but separate
concerns:

```text
shared input preprocessing
FiveK cache compression
FiveK-trained auto-base modeling
FiveK storage/deletion policy
```

That coupling was risky because it could make the project treat "FiveK work" as
the same thing as "input preprocessing". They are not the same:

- input preprocessing is a product-wide foundation used by every renderer;
- FiveK is only training/compression material for an optional pre-film
  auto-base layer.

No content was intentionally dropped. The detailed preprocessing plan was moved
to `docs/PREPROCESSING_INPUT_PIPELINE_TRACKER.md`; the detailed FiveK cache and
auto-base plan was moved to `docs/FIVEK_AUTO_BASE_MODEL_TRACKER.md`.

## 2. Focused Trackers

| File | Role |
|------|------|
| `PREPROCESSING_INPUT_PIPELINE_TRACKER.md` | Product-grade input decoding, color management, HDR/gain-map handling, RAW development, and `WorkingImage` contracts. |
| `FIVEK_AUTO_BASE_MODEL_TRACKER.md` | FiveK cache assets, RAW-derived cache V2, compact response extraction, and auto-base model candidates. |
| `FIVEK_AUTO_OPTIMIZE_TRACKER.md` | This umbrella: business boundary, dependency summary, and deletion gate. |

## 3. Business Goal

FiveK should support a **pre-film automatic base optimization layer**, not the
film stock style layer.

The product pipeline should remain:

```text
compressed SDR/HDR image or RAW
  -> shared input preprocessing
  -> optional FiveK-trained auto-base layer
  -> film stock color layer
  -> grain / halation / defects
  -> output encoding
```

The auto-base layer may produce a clean, neutral, film-ready photographic base:

```text
allowed:
  exposure correction
  white-balance stabilization
  contrast normalization
  highlight shoulder
  mild shadow toe
  reduced digital harshness

not allowed:
  Velvia / Portra / Vision3 stock identity
  hidden saturation-heavy LUT style
  grain, halation, bloom, or defects
  content-changing image generation
```

The wording should be **film-ready photographic base**, not "film style".

## 4. Cross-Tracker Dependency Summary

| Order | Area | Status | Completion Test |
|------:|------|:---:|-----------------|
| 1 | FiveK source inspection | done | local `data/raw/fivek` structure and sizes recorded |
| 2 | Cache V1 scaffold | done | Expert C proxy cache, manifest, summary, and contact sheet generated |
| 3 | Shared preprocessing API | done | `WorkingImage` schema and decoder contracts implemented under `src/preprocess/` |
| 4 | Input inspector | done | `scripts/inspect_image_input.py` reports raster and RAW metadata |
| 5 | SDR raster decode | partial | JPEG/PNG/TIFF decode to float32 `linear_srgb`; HEIF support pending |
| 6 | HDR/gain-map detection | pending | HDR/gain-map images are detected and preserved/flagged instead of silently flattened |
| 7 | RAW decode V1 | partial | rawpy/LibRaw inspection and generic decode path exist; broader camera validation pending |
| 8 | FiveK RAW-derived Cache V2 | partial | 8-image smoke and 64-image mini cache pair RAW/default render with Expert C target; manual visual validation pending |
| 9 | Compact FiveK response assets | partial | mini64 tone/chroma/luma response assets exist; broader representative cache pending |
| 10 | Auto-base candidate | partial | mini64 baseline split into tone-locked default and optional color/WB residual; user visual validation pending |
| 11 | Storage decision | pending | explicit keep/cold-store/delete recommendation for FiveK raw assets |

## 5. Current FiveK Source State

Current local FiveK assets:

```text
data/raw/fivek/fivek_dataset.tar
data/raw/fivek/fivek_dataset.tar.sha1
data/raw/fivek/fivek_index.html
data/raw/fivek/fivek_expert_a_paths.txt
data/raw/fivek/fivek_expert_c_paths.txt
data/raw/fivek/expert_tiff/c/
```

Important size facts measured before this split:

```text
data/raw/fivek/                      about 290 GB
data/raw/fivek/fivek_dataset.tar     about 50.8 GB
data/raw/fivek/expert_tiff/c/        about 260 GB
```

The source archive and TIFFs are ignored local data and must not be committed.

## 6. Cache V1 Result

Cache V1 is already generated and remains useful as an engineering scaffold.

Generated command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_auto_optimize_cache.py `
  --count 256 `
  --size 512 `
  --output-dir outputs\fivek_auto_optimize\cache_v1
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/cache_v1/inputs_512/
outputs/fivek_auto_optimize/cache_v1/targets_512/
outputs/fivek_auto_optimize/cache_v1/pairs_512/
outputs/fivek_auto_optimize/cache_v1/manifest.csv
outputs/fivek_auto_optimize/cache_v1/summary.json
outputs/fivek_auto_optimize/cache_v1/contact_sheet.png
```

Measured cache size:

```text
inputs_512:   256 files, about 12.17 MB
targets_512:  256 files, about 71.66 MB
pairs_512:    256 files, about 26.77 MB
manifest.csv: 256 rows
```

Cache V1 must be treated as:

```text
allowed:
  manifest discipline
  pair layout validation
  training/evaluation interface smoke tests
  contact sheet policy
  small-model plumbing

not allowed:
  final quality judgment
  FiveK deletion justification
  claims of RAW-to-expert learning
  stock-specific film style training
```

## 7. Deletion Gate

FiveK raw assets are not deletable merely because Cache V1 exists.

Minimum gate before deleting or cold-storing `data/raw/fivek/expert_tiff/c/`:

```text
1. RAW-derived Cache V2 exists.
2. Cache V2 has manifest, summary, contact sheets, and decode parameter logs.
3. A compact response/statistics artifact exists.
4. A representative validation subset exists and is visually inspected.
5. At least one auto-base candidate is trained or fitted from the compressed
   assets and compared against Expert C targets.
6. The result does not create stock-specific color style or over-stylized
   "fake film" before the film layer.
7. Rebuild instructions for the compressed assets are committed.
```

Minimum gate before deleting or cold-storing `data/raw/fivek/fivek_dataset.tar`:

```text
1. RAW decode V1 is implemented and tested on multiple cameras from the tar.
2. The RAW-derived cache contains all source images needed for near-term
   training and regression.
3. A small RAW regression subset is preserved outside the deleted archive or
   documented for external cold storage.
4. The user explicitly approves deletion/cold storage after reviewing the cache
   and storage report.
```

Current recommendation:

```text
Keep data/raw/fivek/ for now.
Do not use Cache V1 as deletion justification.
Next useful work is shared preprocessing design + RAW-derived Cache V2.
```

## 8. Non-Goals

- Do not train or promote a film stock style model from FiveK alone.
- Do not delete FiveK raw assets in this branch.
- Do not modify the film color default.
- Do not replace `data/film_domain` with FiveK.
- Do not promise exact proprietary camera-maker RAW rendering.
- Do not silently flatten HDR/gain-map images into SDR without a warning.
- Do not make the FiveK auto-base layer responsible for stock-specific film
  identity.
