# FiveK Auto-Optimize And Input Preprocessing Tracker

> Created: 2026-06-13
>
> Branch: `research/fivek-auto-optimize-cache`
>
> Goal: use FiveK only where it belongs: as training/compression material for a
> future automatic base-photo optimization layer, while separately planning the
> product-grade input preprocessing layer that every renderer path will share.

## 0. Reflection On The Expanded Plan

The expanded preprocessing plan is directionally right, but several parts need
to be tightened before implementation.

What was good:

- It correctly separates input normalization from film-stock rendering.
- It treats SDR/HDR/RAW/ICC handling as a product-grade foundation rather than a
  `PIL.convert("RGB")` convenience wrapper.
- It keeps FiveK as a pre-film auto-base dataset, not as a film identity source.
- It recognizes that RAW-derived FiveK pairs are required before FiveK can be
  treated as a serious deletion candidate.
- It points toward mature high-resolution enhancement families such as
  bilateral-grid affine transforms, image-adaptive LUTs, and local parametric
  filters instead of a large full-resolution U-Net.

What was weak or risky:

- It was too broad to live entirely inside a FiveK tracker. Format decoding,
  color management, HDR metadata, and RAW handling should become a shared
  `src/preprocess/` module used by all renderers. FiveK should only contribute
  training/cache assets for the optional auto-base layer.
- The earlier `cache_v1` input proxy is useful as scaffolding, but it can teach
  the wrong problem because the "input" is artificially degraded from the
  expert target. It must not be used to judge whether the original FiveK TIFF or
  RAW assets are deletable.
- "Film-ready" in the FiveK layer can easily become hidden film stylization.
  The auto-base layer may add a gentle photographic shoulder/toe and neutral
  polish, but it must not learn Velvia/Portra/Vision3 color identity.
- HEIF/HDR/gain-map support is a trap if treated as ordinary 8-bit image IO.
  The first implementation may preserve SDR base plus metadata, but it must
  explicitly flag incomplete HDR reconstruction rather than silently flatten HDR.
- RAW support "without camera-specific profiles" is feasible only in the sense
  of using LibRaw/rawpy's camera metadata and generic processing. It should not
  promise exact Adobe/Apple/Camera-vendor rendering.
- Deleting FiveK should be blocked until compressed assets include true
  RAW-derived inputs, representative validation subsets, and enough response
  statistics to reproduce the desired auto-base behavior.

## 1. Product Boundary

FiveK is not a film-stock dataset. It should not teach Velvia, Portra, Vision3,
grain, halation, or stock identity.

FiveK may teach a separate pre-film auto-base layer:

```text
input image or RAW-derived RGB
  -> auto optimize layer
       exposure / white balance / contrast / highlight-shadow balance
       clean neutral base image
  -> film stock color layer
  -> grain / halation / defects
```

The output of the auto optimize layer must remain a neutral photographic base.
It should improve the input before film rendering, not replace the film stock
transform.

The full product pipeline should be:

```text
compressed SDR/HDR image or RAW
  -> shared input preprocessing
       decode
       orientation
       ICC/CICP/nclx/HDR metadata
       RAW development
       high-precision working image
  -> optional FiveK-trained auto-base layer
       neutral exposure / WB / tone / contrast polish
       optional gentle film-ready shoulder and toe
       no stock-specific style
  -> film stock color layer
  -> grain / halation / defects
  -> output encoding
```

Therefore, implementation should create a shared preprocessing module and not
hide input decoding inside the FiveK cache code.

Recommended code layout:

```text
src/preprocess/
  __init__.py
  types.py
  pipeline.py
  raster_decode.py
  raw_decode.py
  heif_decode.py
  color_management.py
  hdr.py
  export.py

src/models/auto_base/
  __init__.py
  fivek_response.py
  bilateral_grid.py
  adaptive_lut.py

scripts/
  inspect_image_input.py
  build_fivek_auto_optimize_cache.py
  build_fivek_raw_cache.py
  evaluate_preprocess_pipeline.py
```

`scripts/` should stay flat per `docs/PROJECT_STRUCTURE.md`; reusable logic
belongs under `src/`.

## 2. Local Source State

Current local FiveK assets:

```text
data/raw/fivek/fivek_dataset.tar
data/raw/fivek/fivek_dataset.tar.sha1
data/raw/fivek/fivek_index.html
data/raw/fivek/fivek_expert_a_paths.txt
data/raw/fivek/fivek_expert_c_paths.txt
data/raw/fivek/expert_tiff/c/
```

Important size facts measured before this tracker:

```text
data/raw/fivek/                      about 290 GB
data/raw/fivek/fivek_dataset.tar     about 50.8 GB
data/raw/fivek/expert_tiff/c/        about 260 GB
```

The source archive and TIFFs are ignored local data and must not be committed.

## 3. Compression Target

Create ignored output assets under:

```text
outputs/fivek_auto_optimize/cache_v1/
```

The first cache version should contain:

```text
inputs_512/
targets_512/
pairs_512/
manifest.csv
summary.json
contact_sheet.png
```

Each row should represent one paired supervised example:

```text
id
source_name
target_expert
target_tiff
input_proxy
target_512
pair_512
width
height
target_min
target_max
target_luma_mean
target_luma_std
target_chroma_mean
```

The first implementation uses available Expert C TIFFs as targets. Because RAW
decoding from the 50GB tar is a separate risk and dependency, the first cache
uses a conservative input proxy produced from the same expert TIFF:

```text
expert TIFF
  -> input proxy: compressed, lower-contrast, mildly desaturated display image
  -> target: Expert C downsampled display image
```

This does not claim to be true RAW-to-expert supervision. It is a compact first
asset for testing the auto-optimize model surface, manifest discipline, metrics,
and storage policy. True RAW-derived inputs are a later cache version.

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

## 4. Dependency Order

| Order | Task | Status | Completion Test |
|------:|------|:---:|-----------------|
| 1 | Create branch and inspect FiveK layout | done | local `data/raw/fivek` structure measured |
| 2 | Write tracker and boundaries | done | this file exists |
| 3 | Implement cache builder | done | `scripts/build_fivek_auto_optimize_cache.py` builds Expert C paired cache |
| 4 | Generate initial cache | done | `manifest.csv`, `summary.json`, and `contact_sheet.png` exist |
| 5 | Verify generated cache | done | manifest row count equals generated image pairs; contact sheet nonblank |
| 6 | Document storage decision | partial | keep raw FiveK for now; cache v1 is not enough to delete raw TIFF/RAW |
| 7 | Design shared preprocessing API | pending | `WorkingImage` schema and decoder contracts documented |
| 8 | Implement input inspector | pending | reports format, bit depth, ICC/CICP/nclx, HDR/gain-map, EXIF orientation, RAW metadata |
| 9 | Implement SDR raster decode path | pending | JPEG/PNG/TIFF/SDR HEIF decode to high-precision working RGB with profile handling |
| 10 | Implement HDR/gain-map detection path | pending | HDR and gain-map images are detected and preserved/flagged instead of silently flattened |
| 11 | Implement RAW decode V1 | pending | common RAW formats decode through LibRaw/rawpy into a high-bit-depth working image |
| 12 | Build FiveK RAW-derived cache V2 | pending | neutral RAW/default render paired with Expert C target and validation contact sheets |
| 13 | Extract compact FiveK response statistics | pending | tone/chroma/histogram/local-bucket assets exist without full-size TIFF dependency |
| 14 | Train or fit first auto-base candidate | pending | candidate outputs neutral photographic base without stock-style drift |
| 15 | Storage decision after V2 | pending | explicit keep/cold-store/delete recommendation for FiveK raw assets |

## 5. Initial Cache Rules

- Default expert: `c`.
- Default image size: `512`.
- Default count: `256` examples for the first real cache.
- Deterministic sampling with a fixed seed.
- Avoid committing generated cache images.
- Use PNG for targets and JPEG for input proxies unless a later test shows
  compression artifacts harm training.
- Keep all generated assets under ignored `outputs/`.

## 6. Future Cache Versions

### V2: RAW-Derived Inputs

Use `fivek_dataset.tar` to locate the matching original RAW/DNG files, decode
them with LibRaw/rawpy or a documented CLI, and pair:

```text
RAW neutral/default render -> Expert C TIFF render
```

This is the correct supervision for RAW input auto optimization.

V2 requirements:

- preserve source RAW path and Expert C TIFF path in the manifest;
- record LibRaw/rawpy parameters, white-balance mode, demosaic mode, output bit
  depth, and whether auto-brightness was disabled;
- generate at least 1024px training pairs and a smaller 512px smoke subset;
- produce contact sheets with `RAW/default render | Expert C target | delta`;
- include representative validation subsets for dark, high-key, foliage, skin,
  sky, indoor tungsten, mixed light, and high dynamic range scenes.

### V3: Multi-Expert Targets

Add Expert A/B/D/E where available. This can teach user preference variation or
allow target-style choice:

```text
neutral
bright
contrast
warm
natural
```

### V4: Compact Statistical Knowledge

Extract small global/local response assets:

```text
tone curve pairs
Lab histogram deltas
highlight/shadow correction distributions
white-balance shift distributions
scene brightness buckets
```

These assets can support a deterministic auto optimizer without a large neural
network.

### V5: Product-Grade Input Preprocessing

This is not a FiveK cache, but it is the gate before FiveK deletion because it
defines what "input" means for future training and rendering.

Required support:

```text
SDR raster:
  JPEG
  PNG
  TIFF
  ordinary HEIF/HEIC

HDR / wide gamut:
  HEIF/HEIC with 10/12-bit payloads where backend support allows
  JPEG/HEIF gain-map detection and sidecar preservation
  P3, sRGB, Adobe RGB, Rec.2020-style metadata where available

RAW:
  DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2, and other LibRaw-supported formats
  no user-supplied camera profile required for V1
```

Internal representation:

```text
WorkingImage
  pixels: float32 HWC
  working_space: linear Rec.2020 or ACEScg
  transfer_state: scene_linear / display_linear / display_referred
  source_profile: ICC / CICP / nclx / RAW metadata / assumed_sRGB
  hdr_metadata: PQ / HLG / gain map / headroom / none / unknown
  orientation_applied: bool
  alpha_policy: preserved / composited / absent
  bit_depth_in
  warnings
```

This representation should feed both:

```text
auto-base layer
film stock color layer
```

without forcing premature 8-bit sRGB conversion.

## 7. Non-Goals

- Do not train or promote a film stock style model from FiveK alone.
- Do not delete FiveK raw assets in this branch.
- Do not modify the film color default.
- Do not replace `data/film_domain` with FiveK.
- Do not promise exact proprietary camera-maker RAW rendering.
- Do not silently flatten HDR/gain-map images into SDR without a warning.
- Do not make the FiveK auto-base layer responsible for stock-specific film
  identity.

## 8. Manual Decisions Left

- Whether the cache quality is good enough to cold-store or remove
  `data/raw/fivek/expert_tiff/c`.
- Whether RAW-derived V2 is worth implementing before training the auto layer.
- Whether FiveK-trained auto optimization should be enabled by default, or only
  exposed as a pre-film optional layer.
- Whether deletion means permanent deletion, external cold storage, or keeping
  only the 50GB RAW tar plus compressed caches.

## 9. Cache V1 Result

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

Summary means from `summary.json`:

```text
target_luma_mean:  0.2447
target_luma_std:   0.2494
target_chroma_mean: 0.0502
input_luma_mean:   0.2219
input_luma_std:    0.1872
input_chroma_mean: 0.0276
```

Verification:

- `py_compile` passed for the cache builder.
- Smoke cache generation with 4 examples passed.
- Full cache generation produced 256 manifest rows.
- Each of `inputs_512`, `targets_512`, and `pairs_512` contains 256 files.
- `contact_sheet.png` is visually nonblank and shows the expected
  `input_proxy | Expert C target` layout.

Current storage decision:

- Do not delete `data/raw/fivek/` from this branch.
- Cache V1 is useful as a compact supervised scaffold, but it is not a
  replacement for RAW-derived FiveK supervision.
- The large `expert_tiff/c` tree can only be considered for cold storage after
  either V2 RAW-derived inputs are built or the auto-optimization model proves
  the proxy cache is sufficient for the desired product behavior.

## 10. Deletion Gate

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
