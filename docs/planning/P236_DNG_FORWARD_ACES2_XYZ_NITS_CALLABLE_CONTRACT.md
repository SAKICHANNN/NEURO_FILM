# P236 — Exact-Five DNG ForwardMatrix to ACES 2 XYZ-D65 Nits Callable

## Position

P236 is a private engineering child of retained P98 ForwardMatrix raster mechanics
and retained P224 ACES 2 XYZ-D65-nits runtime compatibility. It does not reopen the
closed P99 exposure, P100 normalized output, P229 normalized-PQ callable, or R1CS
analytic-PQ routes.

## Frozen question

Can the exact five P98 scene-linear Rec.2020 `WorkingImage` values be passed,
without exposure, clipping, normalization, or tone rescue, through the official
ACES 2 P3-D65 1000-nit output transform to owned float32 absolute XYZ-D65 values
in cd/m2, while matching an independently assembled official OCIO processor?

## Information flow

- Inputs are the exact five already-consumed P98 DNG identities.
- The callable may read one source DNG and the pinned OCIO runtime/config only.
- P99 exposure metadata is not applied.
- No reference, target, preferred render, or output image is read or written.
- The output is an owned finite float32 `H x W x 3` XYZ-D65-nits array.

## Fixed transform

1. Unchanged P98 DNG camera-linear raster to scene-linear Rec.2020.
2. Official `Linear Rec.2020` to `ACEScg` processor.
3. Official builtins `ACEScg_to_ACES2065-1` and
   `ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - HDR-1000nit-P3-D65_2.0`.
4. Multiply the official relative XYZ result by exactly `100.0` nits.

## Gates

- all five frozen source identities and P98 input hashes remain exact;
- output is owned, C-contiguous, float32, shape-preserving, finite, and has no
  negative component below `-1e-5` nits;
- callable output is byte-exact to an independently assembled official chain;
- exact black maps within `2e-4` nits of zero and neutral probes remain neutral
  within `2e-4` nits;
- source bytes and P98 pixels remain unchanged;
- forward/reverse fresh-process reports are byte-identical.

## Stop rules

Any failure closes this exact five-DNG composition. Do not change exposure,
black floor, output scale, builtins, rows, tolerances, OCIO version, gamut,
clipping, tone mapping, or target on this leaf.

## Claim ceiling

At most: private exact-five-DNG Windows CPU scene-linear Rec.2020 to official
ACES 2 P3-D65 1000-nit XYZ-D65 absolute-light callable mechanics. No calibrated
IDT, arbitrary DNG, photographic/display quality, encoded media, default loader,
public package/schema/capability, product, film-stock, or preference claim.
