# U1.2B Raster Alpha Fail-Closed Results

**Date:** 2026-07-18

**Decision:** pass; transparent raster ingress now fails closed

The previous loader discarded RGBA/LA alpha while claiming it was preserved,
and silently discarded palette transparency. Commit `56a1e91` corrects this at
the raster ingress boundary:

- RGBA and LA with any alpha below 255 fail before RGB working pixels exist;
- palette/`tRNS` transparency follows the same rejection;
- fully opaque alpha is stripped, returns `alpha_policy="absent"`, and records
  `opaque_alpha_discarded`;
- the stripped opaque-alpha pixels are byte-identical to an explicit RGB PNG;
- the integrated renderer rejects transparent input and creates no output.

Verification: 40 focused ingress/renderer/claim tests pass, followed by 563
complete CPU tests in 33.26 seconds. No matte, alpha output, HDR, wide-gamut,
recipe/schema or calibrated-Reference capability was added.

Claim ceiling: honest current SDR RGB ingress only. Alpha preservation and
compositing remain unimplemented and require a separate explicit product
contract.
