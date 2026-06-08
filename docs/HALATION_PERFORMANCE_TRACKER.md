# Halation Performance Tracker

> Created: 2026-06-08
>
> Branch: `research/halation-real-photo-validation-v1`
>
> Status: in progress
>
> Goal: make halation rendering robust for large diffusion presets, GUI
> previews, and eventual still-photo inputs up to 100 megapixels.

## Problem Statement

The V2.3 halation renderer originally called `scipy.ndimage.gaussian_filter`
directly for every blur scale. This is fragile for two reasons:

1. Gaussian filter cost grows with kernel radius. SciPy's default radius is
   proportional to `sigma`, so wide halation tails can become expensive.
2. The product target includes very large still images, potentially up to
   100 megapixels, where multiple full-resolution large-sigma blurs are not
   acceptable for interactive preview or batch processing.

The preset-wide real-photo follow-up increased diffusion values enough to
expose this problem in local validation. This is a performance bug in our
renderer contract: user-facing presets can request a blur field that the
backend is not required to execute safely.

## Research Notes

Sources checked:

- SciPy `gaussian_filter` documentation: the Gaussian kernel radius defaults to
  `round(truncate * sigma)`, and multidimensional filtering is implemented as a
  sequence of one-dimensional convolution passes.
- SciPy signal documentation: FFT and overlap-add convolution are available for
  large convolutions, but boundary handling and memory use need care.
- OpenCV GaussianBlur documentation: production image libraries expose Gaussian
  blur as a primitive but still require kernel-size control.
- Dask/image-style map-overlap designs are appropriate for future tiled
  processing, but they add dependency and scheduling complexity.

## Design Direction

Use a tiered backend:

1. **Direct separable blur for small/medium sigma.**
   - Avoid importing `scipy.ndimage` in `src.filmfx` hot paths.
   - Use a compact separable NumPy implementation for small kernels.
   - Keep behavior deterministic and dependency-light.

2. **Downsampled large-sigma blur for wide halation tails.**
   - If sigma is large relative to current working resolution, downsample,
     blur at lower resolution, then upsample.
   - This is appropriate for halation tails because they are low-frequency
     energy fields, not detail-preserving transforms.

3. **Future tiled/streaming path for 100MP.**
   - Full 100MP arrays need tile + overlap or pyramid/cache execution.
   - This tracker does not require that full production system in the first
     patch, but all new helpers must be compatible with it.

## Dependency Order

```text
Order 1: Remove scipy.ndimage import from filmfx effects hot path
Order 2: Add safe Gaussian helper with direct/downsample paths
Order 3: Preserve filmfx layer API behavior
Order 4: Add regression tests for large-sigma small-image safety
Order 5: Re-run halation tests
Order 6: Regenerate preset-wide contact sheets/metrics
Order 7: Document remaining 100MP work
```

## Task Board

| Order | Task | Status | Completion Test |
|:---:|------|:---:|-----------------|
| 1 | Remove scipy.ndimage import from filmfx effects | done | `import src.filmfx` does not import `scipy.ndimage` |
| 2 | Add safe Gaussian helper | done | large sigma on 48x64 returns quickly |
| 3 | Wire helper into halation/grain | done | filmfx layer tests build all presets |
| 4 | Add regression tests | done | test covers sigma larger than image |
| 5 | Run tests | done | halation/color safety tests pass |
| 6 | Regenerate outputs | done | updated contact sheets written under `outputs/` |
| 7 | Future 100MP notes | done | this tracker records remaining production work |

## V1 Implementation Result

Date: 2026-06-08

Implemented:

- `src/filmfx/fast_blur.py`
- `tests/test_fast_blur.py`
- `src/filmfx/effects.py` now uses `gaussian_filter_safe` for film-effect
  blurs instead of importing `scipy.ndimage.gaussian_filter`.

Validation:

```text
python -m pytest tests/test_fast_blur.py -q
2 passed

python -m pytest tests/test_halation_controls.py tests/test_color_baseline_safety.py -q
12 passed
```

Regenerated outputs:

```text
outputs/eval/halation_real_photo_v1/contact_sheets/alignment_contact_sheet.png
outputs/eval/halation_real_photo_v1/metrics/alignment_summary.json
outputs/eval/halation_real_photo_v1/metrics/simulated_patch_metrics.json
outputs/eval/halation_real_photo_v1/metrics/parameter_suggestions.json
outputs/eval/halation_real_photo_v1/report.md
```

Measured median visible radius after preset-wide enhancement:

| Item | Median radius |
|------|--------------:|
| real candidate patches | 99.48 |
| `vision3_clean` | 5.00 |
| `vision3_push` | 5.00 |
| `cinestill_balanced` | 29.43 |
| `cinestill_strong` | 68.83 |
| `cinestill_amber` | 68.72 |
| `classic_soft` | 6.08 |
| `bw_neutral` | 5.12 |
| `bw_warm` | 5.14 |

Interpretation:

- The immediate hang was caused by the renderer allowing very wide Gaussian
  blur requests to hit an unsafe direct backend.
- The fix is a bounded-cost approximation suitable for halation energy fields.
- It is not yet the final 100MP production renderer.
- CineStill presets now move materially toward the real-photo candidate radius,
  while Vision3/B&W remain intentionally restrained by their stock-family role.

## Acceptance Criteria

1. `src.filmfx` imports without requiring `scipy.ndimage`.
2. Large-sigma blur on small images cannot hang tests.
3. Existing locked-control invariants remain true.
4. Existing preset names remain unchanged.
5. Generated outputs remain under ignored `outputs/`.
6. No claim is made that the downsampled blur is exact physical calibration.

## Future 100MP Work

The first patch is a safety/performance floor, not the whole 100MP production
renderer. Remaining work:

- tile + overlap execution for memory-bounded full-resolution processing;
- cache shared source maps and multi-scale blurs across presets;
- preview/full-resolution split with explicit quality modes;
- optional FFT/overlap-add backend for very large exact-ish convolution;
- benchmark matrix across 12MP, 24MP, 45MP, and 100MP inputs;
- GPU/OpenCL/CUDA/OpenCV backend evaluation if product requirements demand it.
