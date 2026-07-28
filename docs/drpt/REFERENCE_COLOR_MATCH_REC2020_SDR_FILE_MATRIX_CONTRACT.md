# P165 — BT.2020 SDR PNG/CICP file matrix contract

## Question

Does the current file transaction preserve its already advertised
display-linear BT.2020 SDR 16-bit PNG/CICP boundary deterministically, both
alone and beside an sRGB source in one ordered batch?

## Frozen matrix

At 2048-by-1536:

1. BT.2020 SDR PNG16/CICP reference and one BT.2020 source;
2. sRGB PNG16/ICC reference with ordered sRGB and BT.2020 sources.

Every case runs twice against the exact product source at `f2ea6f7`.
Outputs are 16-bit PNG. Source/output rail order, embedded profile kind,
output/recipe/normalized-report identity, identity fallback, resource limits
and cleanup are mandatory.

## Frozen resources

- peak process-tree RSS <= 1.5 GiB per worker;
- repeat peak ratio <= 1.15 per case;
- worker wall time <= 60 seconds;
- no orphan worker or staging temporary.

## Boundary

This is a relative display-linear SDR boundary. It is not PQ/HLG, absolute
HDR, scene-linear RAW, arbitrary ICC/CICP conversion, OCIO/ACES, target-device
runtime, non-identity quality or product-readiness evidence.
