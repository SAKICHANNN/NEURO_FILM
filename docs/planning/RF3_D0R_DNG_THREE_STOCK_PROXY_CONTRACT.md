# RF3.D0R — DNG three-stock proxy execution contract

## Question

Can the exact five-camera P98 ForwardMatrix raster feed the pinned official
ACES 2 SDR output transform and then execute the existing fixed Velvia 50,
Portra 400 and Ektar 100 K=1 proxy operators, with AO6 retained only as the
Velvia display-proxy comparator, without numerical, boundary or replay
failure?

## Frozen execution

- Inputs are exactly the five rows in `configs/p98_dng_forward_raster_v1.json`.
- Decode is the unchanged opt-in P98 camera-linear DNG path.
- Display interpretation is the unchanged OCIO 2.5.2 / ACES 2.0 SDR
  100-nit Rec.709 view.
- The three legacy K=1 arms use the unchanged safe-rich profile assets.
- AO6 uses the unchanged `ao6-source-context-display-look` component and may
  be labelled only as a Velvia 50 display-proxy Look Approximation.
- Every arm is written as exact sRGB RGB16 PNG and read back sample-exactly.
- Two fresh processes run canonical and reverse source order. No fitting,
  strength search, adaptive operator, retrieval or target-film read is allowed.

## Gates and stop rule

All inputs, parent evidence, transforms and profile assets must rehash exactly.
All arrays must remain finite and inside the display interval before encoding.
The three legacy arms must remain mechanically separated by at least one
population-median Delta E76, AO6 must remain separated from legacy Velvia by
at least one, output/new-boundary fractions must stay below the frozen config,
and both reports plus ordered RGB16 inventories must replay exactly.

Failure closes this exact RAW-to-proxy composition without retuning. Passing
retains only deterministic input-path mechanics. It cannot establish target
closeness, stock distinguishability, calibrated stock response, photographic
quality, product promotion or a completed multi-stock system.
