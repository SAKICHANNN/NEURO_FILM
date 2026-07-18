# U1.6G4E Staged Density-Halation Executor Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4E`

**Status:** frozen / implementation ready

**Config SHA-256:** `90785681500d9d2c217a755202591c37edd72ef90de834867e631f1b5f3ae69b`

## Purpose

G3's density graph is the smaller current family: six blurs, one percentile and
three global-grid fields. G4D proves its prerequisite providers exist, but no
graph executes them. G4E freezes the first executor for the **current default
`density_halation_layer` controls only**.

The executor is research-only and disconnected from renderer/profile/CLI paths.

## Operator identity and scope

Version: `staged-density-halation-v1-defaults`

Allowed input:

- finite float32 `HxWx3` display-sRGB in `[0,1]`;
- `H,W >= 2`;
- no separate `source_linear_rgb` in this first leaf.

Controls are fixed to the current default density effect: percentile 99.7,
limiter 2.2 stops, softness 0.46, gamma 1.35, local diffusion 1.0, global
diffusion 0.20, background gain/target 1.25/0.20, tint `(1,.96,.86)`, skin
protect .35, impact .85 and alpha cap .26. Parameter expansion is a later leaf.

## Frozen execution schedule

1. validate input, tile size and source/coarse-row chunks before allocation;
2. compute source-raw through repeatable bounded row factories and resolve 99.7
   with G2 exact two-pass float32 percentile;
3. build `local_mean` from background-probe row windows through G4C;
4. build `tail` and `glare` from normalized/weighted-source row windows through
   G4C; each source window obtains edge gradients through G4A;
5. process output tiles in deterministic row-major order;
6. reconstruct global fields only for required original-coordinate windows;
7. evaluate `local_abs`, `near` and `mid` with current direct float32 Gaussian
   on radius-expanded windows and crop the core;
8. evaluate visibility/density/alpha formulas in the unchanged order;
9. write fixed tint RGB plus alpha into the required full `FilmLayer` outputs;
10. return immutable execution metadata, with no retained derived full-frame
    scalar field.

The input and required RGB/alpha output are explicitly full-frame. Recomputed
row/tile sources, three coarse contexts and percentile histograms are allowed;
hidden full `y`, `source`, `edge`, visibility or blur arrays are forbidden.

## Reference hierarchy

1. **Materialized v2 reference:** identical formulas and G2/G4C/direct kernels
   on full derived arrays. This defines the executor's numerical target, not a
   production path.
2. **Current legacy density effect:** compatibility control using
   `gaussian_filter_safe` and current full-array code. Because v2 intentionally
   changes the global-grid operator, exact legacy bytes are not required.
3. **Identity/no-effect:** visual context only, not an effect-parity target.

## Development evidence and frozen thresholds

Development-only cases `u41-01` and `u41-11` compare materialized v2 to legacy.
Their maximum composite differences are `1.96e-5` and `1.06e-4`; mean
differences are `1.34e-6` and `1.28e-6`; changed uint8 fractions are
`2.92e-4`/`3.16e-4`, maximum one code. Alpha maximum is at most `1.12e-4`.
Visual comparison is indistinguishable and 50x differences are near-black.

Development metrics SHA-256: `c3f311336d3fa93d99657cfbf01e32fb117e0c54e053ecd9d1289912e914fc37`.
Development contact SHA-256: `16a63e0c29e72e756f678395538b73e0bd0ca8f730ac518aa44a32d947658697`.

These two images are excluded from confirmatory promotion decisions.

## Confirmatory data

Use the three separately hashed fixed raw-neutral inputs in
`configs/u1_6g4e_staged_density_executor_v1.json` (`02`, `09`, `14`) plus two
new-seed synthetic highlight/gradient cases. No threshold changes after viewing
their results.

## DoR

- G2, G3, G4A, G4C and G4D pass;
- static density plan has no missing capabilities;
- development/confirmatory cases are separated;
- clean pre-contract HEAD is `ac665f4`;
- no renderer, training, download or GPU process is active.

### Pre-implementation geometry correction

The initial config used coarse-row chunk 17. Before executor code or any
confirmatory run, DoR arithmetic showed that the `193x277` synthetic glare
plan has coarse height 17 and therefore requires a smaller strict chunk. The
uniform chunk is corrected to 7 and the config hash is refrozen below. No case,
algorithm, threshold or gate changed.

## DoD and frozen gates

### A. Staged versus materialized v2

- alpha/composite maximum absolute error `<= 2e-6`;
- seam maximum error `<= 2e-6` at every frozen tile boundary;
- rounded sRGB8 composite bytes identical;
- repeated same-policy float outputs and metadata byte-identical;
- different frozen tile sizes satisfy the same gates independently.

### B. Legacy compatibility

- alpha max/mean drift `<= 5e-4` / `2e-5`;
- composite max/mean drift `<= 5e-4` / `2e-5`;
- changed rounded uint8 channel fraction `<= 0.003`;
- maximum uint8 code delta `<= 1`.

### C. Resource/failure

- every G2/G4A/G4C source request is ordered, bounded and below full height;
- metadata reports input/output, three contexts, histograms, maximum source
  window, maximum tile workspace, actual reread bytes and scratch bytes;
- no persistent derived full-frame scalar allocation is part of the executor;
- 24MP arithmetic reports categories separately; no 100MP claim;
- invalid dtype/range/shape/tile/chunk, nonfinite provider data and injected
  provider failure return no partial public layer.

### D. Visual/severe

- zero confirmed full-resolution seam, band, block, clipping expansion or other
  severe artifact on all three fixed real confirmatory cases;
- compare input, legacy, materialized-v2, staged and amplified difference;
- intended neutral/density glow is not an artifact by itself;
- visual pass is autonomous field/effect evidence, not population preference or
  physical calibration.

### E. Verification

- current legacy density tests remain unchanged;
- focused executor/prerequisite tests pass;
- complete CPU suite passes.

## Branches

- **Pass:** retain experimental density executor and open separate renderer/
  orchestration and 24MP memory/runtime leaves; physical-colour expansion still
  needs its own executor/visual contract.
- **Staged-v2 numerical failure:** repair window/order logic; do not loosen gates.
- **Legacy drift failure:** close or version the behavior explicitly; do not
  silently replace current effect.
- **Severe artifact:** reject regardless of metric proximity.
- **Resource failure:** retain numerical evidence only and keep integration off.

## Claim ceiling

A pass proves the default density-family research executor on frozen cases. It
does not prove physical calibration, colour-family execution, renderer/CLI
integration, streaming decode, total 100MP memory, stock response or default
product promotion.
