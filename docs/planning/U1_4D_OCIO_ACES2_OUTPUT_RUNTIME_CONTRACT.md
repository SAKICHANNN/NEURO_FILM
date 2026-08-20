# U1.4D official OCIO ACES 2 output-runtime contract

Date frozen: 2026-08-21  
Node: `ULT > U1 > U1.4D`  
Status: frozen after package/API discovery and before fixture scoring

## Why this leaf

The project has explicit linear-sRGB/Rec.2020 primitives but no ACES output
transform. This leaf does not invent another tone curve. It tests the current
official implementation shipped in OpenColorIO 2.5.2.

Primary official sources:

- ACES 2 output-transform structure:
  <https://docs.acescentral.com/system-components/output-transforms/>
- OCIO 2.5 built-in ACES 2 configs:
  <https://opencolorio.readthedocs.io/en/latest/releases/ocio_2_5.html>
- ACES CG config usage:
  <https://opencolorio.readthedocs.io/en/v2.5.2/configurations/aces_cg.html>

## Frozen runtime

- Python package: `opencolorio==2.5.2`
- Python module: `PyOpenColorIO`
- built-in config URI: `ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5`
- discovered config cache ID:
  `351c1452fc2bde6177841947c5fce086:6001c324468d497f99aa06d3014798d8`
- source colour space: `ACEScg`
- transform A: display `sRGB - Display`, view
  `ACES 2.0 - SDR 100 nits (Rec.709)`
- transform B: display `Rec.2100-PQ - Display`, view
  `ACES 2.0 - HDR 1000 nits (Rec.2020)`

The local package install and API-name enumeration are mechanics discovery,
not a scientific result. No image or fixture output was scored before this
contract.

## Frozen fixture

The fixture is source-free numerical input in ACEScg:

1. the Cartesian RGB cube over
   `[-0.125, 0, 0.001, 0.01, 0.18, 1, 4, 16, 64]` (`729` rows); and
2. a 257-row neutral ramp uniformly spanning `[-0.125, 64]`.

Rows are float32. Canonical and exact reversed enumeration are both run. Each
transform is evaluated through two independent official CPU APIs:

- scalar `CPUProcessor.applyRGB`; and
- packed float32 `PackedImageDesc` + `CPUProcessor.apply`.

## Gates

All must pass:

1. exact package version, config URI/cache ID, source space, display and view
   names exist and the config validates;
2. all outputs are finite;
3. packed versus scalar maximum absolute error is at most `2e-6` for each
   transform;
4. canonical versus reversed row results are exact after restoring row order;
5. two fresh-process normalized reports are byte exact;
6. neutral-ramp maximum RGB channel spread is at most `1e-4` and mean neutral
   output is non-decreasing within `1e-6` for both transforms;
7. the two official output transforms are materially distinct, with maximum
   absolute difference at least `.01`; and
8. input fixture bytes remain unchanged.

## Stop rules and claim ceiling

Any package/config drift, missing transform, non-finite result, API mismatch,
neutral failure, replay mismatch or input mutation closes the exact runtime.
Do not change fixture range, views, tolerances, processor optimization, bit
depth or config version after scoring.

A pass establishes only isolated Windows CPU feasibility and deterministic
agreement of two official OCIO APIs for these exact ACES 2 output transforms.
It may add the exact dependency pin and retain a private callable in a separate
leaf. It does not establish ACES conformance certification, camera IDT, image
quality, HDR file encoding/metadata, display calibration, GPU/macOS parity,
renderer integration, public schema/capability, film/stock or product support.
