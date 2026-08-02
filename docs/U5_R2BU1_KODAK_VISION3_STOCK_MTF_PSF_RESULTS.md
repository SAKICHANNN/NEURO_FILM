# U5.R2BU1 VISION3 stock MTF Gaussian PSF

## Decision

Close the exact one-Gaussian-per-layer compiler without rescue. The fitted
stock-specific kernels retain a strong source-domain stock signal, but their
confirmation median RMSE is `.05307`, above the frozen `.05` gate.

## Evidence

- two byte-identical reports: `c06d79c5...f64f8`;
- stable evidence ID: `a4f58ddf...3174d`;
- candidate median/worst confirmation RMSE: `.05307/.07254`;
- shared and cyclic wrong-stock median RMSE: `.10098/.17400`;
- aggregate improvement over shared/wrong: `47.45%/69.50%`;
- all nine stock-layer rows beat identity, and every stock beats both controls;
- only the pre-registered median-error gate fails.

## Boundary

The result supports a manufacturer source-domain stock-specific spatial
signature and rejects a single Gaussian as a sufficient compiler. It does not
identify placement, real-roll MTF, scanner response, a photographic look or a
product profile. No Gaussian-mixture or threshold rescue opens on these traces.
