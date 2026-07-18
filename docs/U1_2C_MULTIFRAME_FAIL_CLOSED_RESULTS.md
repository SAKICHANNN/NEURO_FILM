# U1.2C Multi-Frame Raster Fail-Closed Results

**Date:** 2026-07-18

**Decision:** pass; silent frame-zero fallback is closed

Commit `e03bf77` makes `load_raster_working_image` require exactly one inspected
frame before pixel decode. A two-frame GIF and two-page TIFF both reject with
the explicit unsupported multi-frame error. The integrated renderer creates no
output for the animated GIF, while ordinary single-frame raster regressions
remain unchanged.

Verification: 43 focused ingress/renderer/claim tests and 566 complete CPU
tests pass. No animation, frame selection, video, burst, temporal processing,
stack or multi-page export support was added.

Claim ceiling: honest single-frame raster ingress only.
