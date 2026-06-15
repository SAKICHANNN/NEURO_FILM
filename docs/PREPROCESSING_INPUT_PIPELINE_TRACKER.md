# Preprocessing Input Pipeline Tracker

> Created: 2026-06-15
>
> Split from: `docs/FIVEK_AUTO_OPTIMIZE_TRACKER.md`
>
> Goal: design the product-grade input normalization layer shared by every
> renderer path. This is independent from FiveK except that FiveK RAW-derived
> cache generation will use it.

## 1. Reflection

The original expanded plan was directionally right, but it was too broad to live
inside a FiveK-specific tracker.

What is good and should be preserved:

- Input normalization must be separate from film-stock rendering.
- SDR/HDR/RAW/ICC handling is a product-grade foundation, not a convenience
  wrapper around `PIL.convert("RGB")`.
- HEIF/HDR/gain-map inputs must be detected and preserved or warned about; they
  must not be silently flattened into SDR.
- RAW support without user-supplied camera profiles is feasible only through a
  generic LibRaw/rawpy path using camera metadata; it should not promise exact
  Adobe, Apple, or camera-vendor rendering.

What this tracker intentionally avoids:

- It does not train FiveK models.
- It does not define film stock identity.
- It does not decide whether FiveK can be deleted.

## 2. Business Role

The preprocessing layer creates a high-precision `WorkingImage` from any
supported input:

```text
compressed SDR/HDR image or RAW
  -> decode
  -> orientation
  -> color metadata
  -> RAW development if needed
  -> high-precision working image
```

It feeds both:

```text
optional FiveK-trained auto-base layer
film stock color layer
```

without forcing premature 8-bit sRGB conversion.

## 3. Required Format Scope

### SDR Raster

```text
JPEG
PNG
TIFF
ordinary HEIF/HEIC
```

### HDR / Wide Gamut

```text
HEIF/HEIC with 10/12-bit payloads where backend support allows
JPEG/HEIF gain-map detection and sidecar preservation
P3, sRGB, Adobe RGB, Rec.2020-style metadata where available
PQ / HLG / CICP / nclx metadata where available
```

### RAW

```text
DNG
CR2 / CR3
NEF
ARW
RAF
ORF
RW2
other LibRaw-supported formats
```

V1 does not require user-supplied camera profiles.

## 4. Recommended Code Layout

Reusable code belongs under `src/`; command entrypoints stay flat under
`scripts/` per `docs/PROJECT_STRUCTURE.md`.

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

scripts/
  inspect_image_input.py
  build_fivek_raw_cache.py
  evaluate_preprocess_pipeline.py
```

## 5. WorkingImage Contract

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

The implementation should preserve enough metadata to explain how an image was
decoded. Silent assumptions should become explicit warnings.

## 6. Dependency Order

| Order | Task | Status | Completion Test |
|------:|------|:---:|-----------------|
| 1 | Design `WorkingImage` schema | pending | typed contract documents pixels, color state, HDR state, source metadata, warnings |
| 2 | Implement input inspector | pending | reports format, bit depth, ICC/CICP/nclx, HDR/gain-map, EXIF orientation, RAW metadata |
| 3 | Implement SDR raster decode path | pending | JPEG/PNG/TIFF/SDR HEIF decode to high-precision working RGB with profile handling |
| 4 | Implement color-management utilities | pending | ICC-tagged sRGB/P3/AdobeRGB images map consistently to working RGB |
| 5 | Implement HDR/gain-map detection | pending | HDR and gain-map images are detected and preserved/flagged instead of silently flattened |
| 6 | Implement RAW decode V1 | pending | common RAW formats decode through LibRaw/rawpy into a high-bit-depth working image |
| 7 | Add evaluation fixtures | pending | small local fixtures cover SDR, wide-gamut, TIFF, HEIF, HDR/gain-map, and RAW cases |
| 8 | Connect FiveK RAW cache builder | pending | FiveK RAW-derived cache generation uses shared preprocessing APIs |

## 7. Accuracy Rules

- Prefer float32 processing for internal pixels.
- Avoid clipping unless explicitly mapping to an output-referred format.
- Preserve or record input ICC/CICP/nclx metadata.
- Apply EXIF orientation exactly once.
- For unknown raster profiles, assume sRGB only with a warning.
- For HDR/gain-map images, preserve the SDR base and metadata even if full HDR
  reconstruction is not implemented yet.
- For RAW, record LibRaw/rawpy settings including white balance, demosaic,
  output bit depth, and auto-bright behavior.

## 8. Non-Goals

- Do not train the FiveK auto-base model here.
- Do not implement film stock style here.
- Do not use this layer to add grain, halation, bloom, or defects.
- Do not promise exact proprietary camera-maker RAW rendering.
- Do not silently flatten HDR/gain-map images into ordinary SDR.

## 9. Storage Relationship To FiveK

This tracker is a gate before deleting FiveK because FiveK RAW-derived cache
generation depends on knowing what "input" means.

FiveK deletion remains blocked until:

```text
1. RAW decode V1 exists.
2. FiveK RAW-derived Cache V2 is generated through this preprocessing layer.
3. Decode parameters and representative regression fixtures are preserved.
```
