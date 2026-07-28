# U5.R2AJ0B HaldCLUT acquisition v1 failure

## Decision

`U5.R2AJ0B` v1 is closed. The exact external archive was acquired at
421,602,289 bytes and its published MD5 passed, but the first independent
complete audit stopped at:

```text
HaldCLUT/Black-and-White/Agfa/Agfa APX 25.png
observed Pillow mode: L
frozen allowed modes: RGB, RGBA
```

No single-run manifest, repeat decision, CLUT application, photograph render
or structural-ready state was produced. The v1 config remains unchanged at
SHA-256 `fe38f696...d3350ef4`.

## Exact retained source

| Field | Value |
|---|---|
| bytes | `421602289` |
| MD5 | `4742e362a70c1a1c0fb9042a17d285e1` |
| SHA-256 | `0ffca81f30c72d7bbb85adab0ed98bbdeb0e1034cd6f5986f9f993bb66999dc2` |
| implementation commit | `2d154f7130000843f1f440dc96cae9e9328f7908` |
| retained path | ignored `data/external/rawtherapee_haldclut/20150920/HaldCLUT.zip` |

The archive is retained because it is the exact bounded, licensed source
approved by AJ0A. It is not copied into Git or a release package.

## Diagnostic after the frozen failure

After v1 had closed, a read-only in-memory copy of its config added mode `L`
solely to enumerate decode metadata. That diagnostic is not a formal pass.
It decoded all 295 images and found:

| Path family | Count | Format / mode / bits | Hald level |
|---|---:|---|---:|
| Color except exact Ektar level-16 path | 226 | PNG / RGB / 8-bit x3 | 12 |
| `Color/Kodak/Kodak Ektar 100.png` | 1 | PNG / RGB / 8-bit x3 | 16 |
| Black-and-White | 66 | PNG / L / 8-bit | 12 |
| `Negative.png` | 1 | PNG / RGB / 16-bit x3 | 12 |
| `Hald_CLUT_Identity_12.tif` | 1 | TIFF / RGB / 16-bit x3 | 12 |

All 295 report no embedded ICC. The exact diagnostic image-record and primary
path hashes are frozen in
`configs/u5_r2aj0b_haldclut_acquisition_decision_v1.json`.

The failure is therefore a false all-RGB inventory assumption, not evidence
that a Color primary CLUT is grayscale. It also does not establish that any
CLUT is safe, useful or authentic.

## Branch

The original v1 gate is not changed after seeing its result. A separately
versioned `U5.R2AJ0B2` may confirm the exact observed path-family profiles in
two new processes. AJ0C remains closed until B2 passes.

The 16-bit root controls are integrity records only in B2. Pillow can expose
16-bit RGB metadata while loading RGB samples at 8-bit precision; future
operator work must generate the identity lattice analytically or use a
verified bit-preserving decoder. It must not use Pillow samples from either
16-bit root control.
