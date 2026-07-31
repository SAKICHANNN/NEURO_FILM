# U6.P6L — Same-plate repeat-scan nuisance contract

## Decision question

Can three rights-cleared scans of the same physical photographic plate isolate
a repeat-acquisition residual that materially constrains the generic P6A
scanner-noise model?

The source is the University of Chicago dataset *Precise Photometric
Measurements from a 1903 Photographic Plate Using a Commercial Scanner*
([record](https://knowledge.uchicago.edu/records/qw505-gq084),
[DOI](https://doi.org/10.6082/uchicago.3901)). The repository marks the
dataset CC BY 4.0 and describes `f0007.tif`, `f0008.tif`, and `f0009.tif` as
repeated scans of the same central plate area at the same focus.

## Frozen execution

1. Acquire only the three exact TIFFs plus `README.txt` (88,475,062 bytes).
2. Verify repository MD5, local SHA-256, single-frame decode, equal
   shape/dtype and two independent audit reports.
3. Estimate bounded translation registration and require every registered
   pair to correlate at least `.995`.
4. Only after the source gate passes, fit per-scan affine intensity nuisance
   on development regions and measure the residual variance-versus-signal,
   two-dimensional NPS, ACF, and row/column structure.
5. Compare those measurements with the existing P6A independent
   shot-plus-read noise family. A mismatch may open a separately versioned
   scanner-noise challenger; it does not authorize parameter tuning in P6A.

## Epistemic boundary

The 1903 plate stock, exposure, development and full scanner state are
unknown. Fixed plate texture is common signal, not scanner noise. Difference
residual is scanner/workflow nuisance plus registration error, not emulsion
grain. This leaf cannot fit colour, identify a stock, calibrate a scanner, or
enter production.
