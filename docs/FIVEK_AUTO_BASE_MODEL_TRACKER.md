# FiveK Auto-Base Model Tracker

> Created: 2026-06-15
>
> Split from: `docs/FIVEK_AUTO_OPTIMIZE_TRACKER.md`
>
> Goal: compress FiveK into small supervised assets and use those assets to
> build an optional pre-film auto-base layer.

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

The output of the auto-base layer must remain a neutral photographic base. It
should improve the input before film rendering, not replace the film stock
transform.

## 2. Film-Ready But Not Film-Styled

The auto-base layer may add:

```text
natural exposure correction
clean white balance
reasonable contrast
gentle highlight shoulder
mild shadow toe
subtle reduction of harsh digital rendering
```

The auto-base layer must not add:

```text
Velvia saturation or hue identity
Portra skin-color identity
Vision3 shadow color identity
grain
halation
bloom
obvious LUT style
content-changing generation
```

The product wording should be **film-ready photographic base**, not "film
style".

## 3. Local Source State

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

## 4. Cache V1 Scaffold

Create ignored output assets under:

```text
outputs/fivek_auto_optimize/cache_v1/
```

The first cache version contains:

```text
inputs_512/
targets_512/
pairs_512/
manifest.csv
summary.json
contact_sheet.png
```

Each row represents one paired supervised example:

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
decoding from the 50GB tar is a separate risk and dependency, Cache V1 uses a
conservative input proxy produced from the same expert TIFF:

```text
expert TIFF
  -> input proxy: compressed, lower-contrast, mildly desaturated display image
  -> target: Expert C downsampled display image
```

This does not claim to be true RAW-to-expert supervision. It is a compact first
asset for testing the auto-optimize model surface, manifest discipline, metrics,
and storage policy.

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

## 5. Cache V1 Result

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

## 7. Model Candidate Order

Prefer mature, high-resolution-friendly enhancement families:

```text
1. Bilateral-grid affine / HDRNet-style model
2. Image-adaptive 3D LUT
3. Local parametric filters
```

Avoid starting with:

```text
large full-resolution U-Net
diffusion enhancement
content-changing generative model
```

The candidate must be optional, controllable, and easy to disable.

## 8. Dependency Order

| Order | Task | Status | Completion Test |
|------:|------|:---:|-----------------|
| 1 | Inspect FiveK layout | done | local source paths and sizes recorded |
| 2 | Implement Cache V1 builder | done | `scripts/build_fivek_auto_optimize_cache.py` builds Expert C proxy cache |
| 3 | Generate Cache V1 | done | manifest, summary, and contact sheet exist |
| 4 | Build RAW-derived Cache V2 | partial | 8-image smoke cache pairs RAW/default render with Expert C target; user visual validation pending |
| 5 | Extract compact response assets | pending | tone/chroma/histogram/local-bucket assets exist without full-size TIFF dependency |
| 6 | Fit deterministic response baseline | pending | response assets produce neutral film-ready base on validation subset |
| 7 | Train first lightweight candidate | pending | candidate improves base tone without stock-style drift |
| 8 | Compare to Expert C | pending | objective metrics and contact sheets generated |
| 9 | Storage recommendation | pending | explicit keep/cold-store/delete recommendation for FiveK assets |

## 9. Non-Goals

- Do not train or promote a film stock style model from FiveK alone.
- Do not delete FiveK raw assets from this tracker.
- Do not modify the film color default.
- Do not replace `data/film_domain` with FiveK.
- Do not make the FiveK auto-base layer responsible for stock-specific film
  identity.

## 10. RAW-Derived Cache V2 Smoke Result

Generated command:

```powershell
.\.venv\Scripts\python.exe scripts\build_fivek_raw_cache.py `
  --count 8 `
  --size 512 `
  --output-dir outputs\fivek_auto_optimize\raw_cache_v2_smoke
```

Generated ignored outputs:

```text
outputs/fivek_auto_optimize/raw_cache_v2_smoke/raw_renders_512/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/targets_512/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/pairs_512/
outputs/fivek_auto_optimize/raw_cache_v2_smoke/manifest.csv
outputs/fivek_auto_optimize/raw_cache_v2_smoke/summary.json
outputs/fivek_auto_optimize/raw_cache_v2_smoke/contact_sheet.png
```

Measured smoke size:

```text
raw_renders_512: 8 files, about 0.37 MB
targets_512:     8 files, about 2.09 MB
pairs_512:       8 files, about 0.82 MB
manifest.csv:    8 rows
missing_count:   0
```

What this proves:

- FiveK Expert C TIFF names can be matched to RAW/DNG tar members by stem.
- The local `fivek_dataset.tar` can be read without full extraction.
- RAW members can be temporarily extracted, decoded through the shared
  `src.preprocess` path, and paired with Expert C targets.
- The smoke contact sheet is nonblank and correctly laid out as
  `RAW/default render | Expert C`.

What this does not prove yet:

- The generic LibRaw/rawpy RAW render is the best input representation for the
  final auto-base model.
- The Expert C target is the preferred product aesthetic.
- The 8-image smoke subset is representative enough to justify deleting or
  cold-storing FiveK.

Manual visual validation needed:

```text
outputs/fivek_auto_optimize/raw_cache_v2_smoke/contact_sheet.png
```
