# Physical Film Image Formation Programme (U6.P0-P9)

Status: active, data-gated research programme. This is a cross-tree child of
U2 (exposure, density and interpretation), U5 (stock/mode/case colour
operators), U6 (image formation) and U7 (runtime). Existing U6.1-U6.6 history
and frozen experiments are unchanged.

## Objective and claim boundary

Build a deterministic, explicit film-image-formation path that can be visibly
stylized without severe artifacts, while keeping every physical claim tied to
controlled evidence. The programme permits two truthful outcomes:

- `generic-physical-inspired` when the mechanism and units are explicit but
  stock/process/scanner calibration data are absent;
- `calibrated-reference` only after rights-cleared controlled measurements and
  independent roll/process/scanner holdouts pass.

AO9 is a clean-room capacity control for the published two-matrix,
three-sigmoid colour equation. The local 71 author-rendered display-proxy
pairs are not digital/film capture pairs, so AO9 cannot establish emulsion,
process or stock response. AO6 remains a downstream display/look residual. It
must later survive a combined-path ablation and must not double-count exposure,
development, scanner or display effects.

For normally processed colour film, the developed silver image is removed by
bleach and fix; colour image structure is therefore represented as dye
cloud/dye-density structure. Metallic-silver structure is a distinct B&W (or
explicit silver-retention) branch. Missing process or scanner metadata remains
`unknown` or `hypothesis_only`.

## Canonical domain order

```text
WorkingImage scene-linear + color-state
  -> optional neutral base
  -> user-selected stock / global-or-hard-mode / sparse-case parameters
  -> sensor-to-film layer exposure
  -> exposure-domain forward scatter + backing reflection/halation
  -> per-layer characteristic/development response
  -> developed image structure
       B&W: metallic silver
       colour: dye clouds / dye density
  -> density/process-dependent MTF, adjacency/acutance and dye diffusion
  -> negative / slide / B&W print-or-scan interpretation
  -> scanner spectral response, flare, MTF and noise
  -> optional AO6-like downstream display/look residual
  -> separate creative bloom, dust, scratch and light leak
  -> versioned OCIO/output transform
  -> one final quantization
```

No stage may silently relabel display RGB as exposure, density,
transmittance or scan-linear data. Scanner is an independent output profile and
nuisance variable, never a stock or latent mode. ML may fit or predict bounded
explicit parameters, one hard mode or sparse case weights; it may not generate
the final RGB image.

## Execution nodes

| Node | Executable result | Stop/branch rule |
|---|---|---|
| U6.P0 | Typed domains, units, profile identity, deterministic seed/tile contract and Preview/Standard/Reference measurement semantics; fail-closed checks and density/transmittance roundtrip tests. | A domain mismatch or unstable identity blocks downstream work. |
| U6.P1 | Offline float64 high-fidelity reference simulator and synthetic chart suite; full-resolution or marked-Poisson/Monte-Carlo computation is allowed. | Reference/compiler evidence only, never stock calibration or a product dependency. |
| U6.P2 | Reuse U2.2 sensitometry for exposure/development; add separate negative, slide and B&W interpretations with explicit normal/push/pull unknown branches. | No duplicated tone curve; missing measurements cap the claim at generic physical-inspired. |
| U6.P3 | Layer/spectrum-aware forward scatter and base return before development using positive, energy-bounded PSF mixtures; compile to separable kernels/pyramids. | Screen blend, scanner glare and creative bloom are invalid substitutes. Validate radial colour/exposure scale and energy on point/edge charts. |
| U6.P4 | Deterministic physical-coordinate marked-Poisson/Boolean silver-grain or dye-cloud fields with density-dependent probability, geometry and optical density; compile dense subpixel regions to an NPS-preserving approximation. | U6.2B salt-like bright speckles are a frozen hard negative. Tile, resolution and frame coordinates preserve the counter-based field. |
| U6.P5 | Keep forward scatter, adjacency/acutance, dye diffusion, film MTF and scanner MTF independently parameterized in physical units. | A single sharpening slider cannot satisfy this node. Measure slanted edge, line pairs, MTF50/10, overshoot, NPS and ACF. |
| U6.P6 | Explicit direct-slide, neutral-scan and print-chain scanner models: illuminant/spectral response, flare, Dmax, noise, MTF and optional sharpen/denoise. | Repeat scans and multiple scanners identify nuisance/output profiles only; rights and lineage precede fitting. |
| U6.P7 | Freeze colour-only, physics-only, combined, cheap approximation and shuffled/wrong-profile controls. | Severe artifacts veto first. If physics or routing has no held-out gain, retain the simpler global colour champion. |
| U6.P8 | Compile a hashed compact `FilmProfileBundle` for canonical CPU, C++/SIMD, Apple Metal/MPS and Android Vulkan. Optional tiny routers may use ONNX/LiteRT/CoreML. | Runtime has no Python, GCP or large-model dependency. GPU failure gets the same-quality CPU path. |
| U6.P9 | Validate still export first, then independently switchable temporal grain, registration/gate weave and flicker. | Video failure does not block still Ultimate. |

P0 and P1 are immediately executable. P2 reuses existing U2.2 code. P3-P5
open only after P0/P1 provide typed reference inputs and measurable synthetic
witnesses; they do not wait for every stock/mode/retrieval branch to finish.

Current P7 boundary: a neutral-gauged 4000-dpi generic challenger is
repeat-exact, severe-clean on the frozen full-resolution set and narrowly wins
two of three autonomous blind rounds. Row partitions and execution order are
exact at that reference identity. Independently recomputing a lower-resolution
render is not equivalent to rendering at 4000 dpi and then resampling; AO6
source-context/resampling order is the dominant real-image contributor. P8 may
therefore compile only a fixed-reference bundle. Preview remains
reference-derived or a separately validated approximation.

Current P8 boundary: P8A-P8W compile and validate a hash-bound, artifact-only
Python canonical profile consumer. The fixed P7 identity remains float-exact
through forward/reverse partitions and the frozen 32-row oracle. Successive
attributed buffer changes reduce the 12 MP local process-tree peak from
1.79-1.88 GiB at P8H to 717.14-717.15 MB at P8W; the desktop Python
reference memory target passes, while native/mobile latency and memory remain
unproved. Python buffer micro-optimization is closed. P8X next freezes a
portable native CPU ABI and small-vector conformance oracle; this does not
change the production default or any calibration claim.

## Quality tiers and measurement contract

- **Preview:** deterministic 1-2 MP interaction path; approximations are
  versioned and compared against Standard.
- **Standard:** production still path for 12 MP mobile and 24 MP desktop.
- **Reference:** float64/offline reference and 100 MP tiled validation; never a
  product runtime dependency.

Measure warm and cold p50/p95 time, process-tree peak RSS/VRAM, output
repeatability, tile seams, CPU-reference error and thermal/energy evidence where
the platform exposes it. Initial engineering targets are provisional until
U4.5/P0 benchmarks freeze them: mobile extra memory 256-384 MB, desktop CPU
below 1 GB (100 MP may be separately below 4 GB), 12 MP mobile final 3-8 s,
24 MP desktop GPU below 3 s, M5 below 5 s and ordinary CPU below 15 s.
Failure reduces scales/LOD, not the physical domain order.

The canonical CPU path is deterministic. A GPU backend may use a frozen
numerical tolerance rather than cross-device bit identity, but backend-local
repeatability, recipe/profile identity and artifact gates are mandatory. All
full-resolution operators are tile-streamed, avoid redundant intermediates and
quantize only once.

## Evidence and data gate

Manufacturer sheets are parameter priors, not fitted truth. Community scans,
the 71 display-proxy pairs and AO6/AO9 cannot fit real halation, grain, MTF or
scanner response. Calibrated work requires rights-cleared:

- step wedges and uniform density fields;
- point/highlight, edge and line-pair charts over multiple exposures;
- multiple independent rolls and process sessions;
- repeat scans and multiple scanner conditions;
- scanner dark/flat fields and MTF measurements;
- group-held-out manifests with exact lineage and leakage checks.

Until those exist, the valid result is a generic mechanism, Look Approximation
or a documented data gap. GCP may accelerate reference simulation, multi-seed
fitting, ablation and profile compilation, but cannot repair identifiability.

## Minimum source basis

- [KODAK VISION3 500T technical information](https://www.kodak.com/content/pdfs/KODAK-VISION3-5219-7219-technical-information.pdf):
  sensitometry, density-dependent diffuse RMS granularity and MTF in cycles/mm.
- [KODAK motion-picture processing](https://www.kodak.com/content/products-brochures/Film/Processing-KODAK-Motion-Picture-Films-Module-5.pdf):
  bleach converts metallic silver so fixing can remove it.
- [SMPTE ACES standards](https://www.smpte.org/standards/aces-standards):
  APD/ADX distinguish film-density exchange from scene-linear/display RGB.
- [FIAF Digital Statement II](https://www.fiafnet.org/images/tinyUpload/2022/05/Digital_Statement_Part_II_2021_V13-validated.pdf):
  scanner optics, illumination, sensor, noise and aliasing are separate from
  photochemical grain.
- [Newson, Delon and Galerne 2017](https://doi.org/10.1111/cgf.13159):
  resolution-independent stochastic grain as a research baseline.
- [Jarvis 1995](https://doi.org/10.1080/00223638.1995.11738635):
  dye-cloud diffusion jointly affects MTF, NPS and granularity.
- [Emulating Emulsion 2025](https://musicofmusix.github.io/siggraphposters25):
  compact paired colour-equation family; AO9 retains it only as a capacity
  baseline under the local proxy evidence ceiling.
