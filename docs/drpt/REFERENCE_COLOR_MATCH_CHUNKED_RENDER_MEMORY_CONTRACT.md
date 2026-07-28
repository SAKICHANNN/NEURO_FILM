# P156 Chunked Reference-Render Memory Contract

## Trigger

P154 proved exact outer-object lifetime improvements but failed its whole-path
memory gates. The follow-up phase trace measured the candidate worker peak
inside `render-gamut-safe-lab` and the following full-frame gamut verification,
while `render-style-lab` was materially lower.

The hypothesis is narrow: gamut compression, Lab-to-linear-RGB conversion and
final gamut verification are row-independent for a fixed already-computed
source/styled Lab pair. Applying the existing authoritative functions to fixed
row blocks and writing into one preallocated result should preserve every
float32 output bit while bounding those stages' temporary arrays.

This does not tile or change safe-Lab style math. It does not change the fitted
recipe, policy, guard, producer boundary or main renderer.

## Frozen implementation

- Functional parent: `de57918e23568ad6321262903addfac7f2349d3c`.
- Allowed production file: `src/color_match/render.py`.
- Fixed internal row chunk: 128.
- The implementation may only row-chunk `_gamut_safe_lab`, Lab-to-linear-RGB
  conversion and the final output-Lab gamut check.
- Both `source` and `chroma` gamut policies must match the previous full-frame
  implementation exactly on adversarial geometry, non-divisible final chunks,
  in-gamut and compressed cases.
- No schema, algorithm ID, recipe identity, policy default, diagnostic field,
  public export or output encoding may change.

The implementation commit must be created before the measured comparison. The
tracked execution config will then bind that exact commit without changing
these gates.

## Frozen verification and gates

Unit/integration evidence must prove:

1. full-frame versus row-chunked gamut Lab is exact float32 for both gamut
   modes;
2. full-frame versus row-chunked Lab-to-RGB is exact float32;
3. prior recipe, render, guard, file and replay tests pass;
4. P154's 6 MP deterministic output PNG, recipe and normalized report remain
   exact under default identity fallback.

The measured order remains baseline, candidate, candidate, baseline with fresh
processes and 20 ms complete process-tree sampling. Relative to the P153
baseline, candidate median peak RSS must:

- fall by at least 134,217,728 bytes (128 MiB); and
- be no more than 0.88 of baseline median.

Candidate median worker wall time must be no more than 1.10 of baseline. All
workers must leave zero observed process and zero transaction temporary. A
failed gate remains a negative result; no threshold may be adjusted after
measurement.

## Claim boundary

A pass supports only an exact row-chunked relative-SDR reference-render memory
path on this local Windows/Python environment. It is not 100 MP evidence,
allocator-independent proof, mobile/Apple runtime, RAW/HDR/video support,
algorithm-quality promotion, or a change to A1/A4/A5.
