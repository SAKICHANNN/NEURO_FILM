# U5.R2H0A Velvia 50 datasheet spectral witness contract

Date: 2026-07-24
Status: **frozen before curve data, implementation or numerical results**
Parent: `U5.R2H0`

## Hypothesis

A minimal direct-positive operator assembled only from the official Velvia 50
bulletin and official CIE tables may retain a film-specific nonlinear colour
direction that is stable to the missing scene spectrum. The competing null is
that RGB metamer choice or graph ambiguity dominates the apparent effect.

This is a falsifiable `film-inspired/datasheet-prior` hypothesis. It is not an
identified digital-to-film transform.

## Fixed approximation

The bulletin does not publish analytical dye amounts or a scanner/projector
profile. H0A therefore freezes the smallest disclosed approximation:

- input is bounded linear display-sRGB interpreted as D65 tristimulus colour;
- a smooth bounded reflectance is reconstructed on 380--720 nm at 5 nm;
- two bounded null-space alternatives are constructed to match the same D65
  XYZ while stressing film-layer response;
- layer exposure is the integral of D65, reflectance and the digitized
  blue/green/red sensitivities;
- channel gains make 18% neutral reach `log H=-0.65` in all three layers;
- digitized Status-A density minus each curve's D-min is associated
  `B -> yellow`, `G -> magenta`, `R -> cyan`;
- each density scales the corresponding separated dye curve, which the sheet
  defines at density 1.0 above D-min;
- transmittance is integrated under both D65 and D50. D50 XYZ is Bradford
  adapted to D65 before linear-sRGB conversion;
- no scan curve, creative strength, gamut mapper, local effect or fitted
  correction is permitted in H0A.

Status-A density is not analytical dye amount. The association above is an
explicit witness approximation and a claim limitation, not hidden physics.

## Graph digitization contract

The three embedded one-bit graph images on PDF page 8 are exact source
artifacts. Axis calibration is fixed from printed grid intersections in the
config. Curves are stored as annotated pixel coordinates, never as values
chosen directly in physical units. The implementation converts pixels through
the fixed affine axes and uses shape-preserving interpolation.

Every trace must meet its minimum annotated-point count. Each annotation must
be within two pixels of black source ink after excluding the known grid-line
coordinates. A generated overlay must show every annotation and reconstructed
trace on the exact source image. This distance check proves transcription to
the page, not that an ambiguous crossing was semantically labelled correctly;
crossings and line-style ambiguity remain documented uncertainty.

No point may be moved after the first numerical report. A semantic trace error
closes this version and requires a new version with an explicit erratum.

## Synthetic population

- encoded-sRGB cube: nine equally spaced values per channel, 729 colours;
- neutral ramp: 33 equally spaced encoded values;
- spectral grid: 380--720 nm inclusive at 5 nm;
- no project photograph, real-film image, style anchor or prior output is read;
- seed `20260724` is used only for deterministic ordering and report identity.

## Frozen gates

All gates are conjunctions.

### Source and trace integrity

- exact PDF, embedded-image and CIE hashes;
- affine axis calibration maximum residual `<=2.0 px`;
- annotated ink distance maximum `<=2.0 px`;
- minimum annotated points: characteristic 9/curve, sensitivity 8/curve,
  dye density 12/curve;
- exact repeat report and array hashes.

### RGB spectral reconstruction

- every spectrum finite and inside `[0,1]` within `1e-9`;
- base smooth-spectrum D65 reconstruction: median Delta E76 `<=0.25`,
  p95 `<=0.75`, maximum `<=2.0`;
- both metamer alternatives versus the same input under D65: p95 Delta E76
  `<=0.75`, maximum `<=2.0`;
- alternatives must be distinct: median spectral RMS from base `>=0.01` for
  non-neutral colours with encoded maximum at least 0.25.

### Witness validity

- all layer exposures, densities, spectra and output XYZ are finite;
- all generated transmittances stay in `[0,1]` within `1e-9`;
- D65 neutral-ramp Y is nondecreasing with tolerance `1e-6`;
- neutral-ramp output chroma `sqrt(a^2+b^2)` maximum `<=4.0` after white
  normalization;
- output RGB gamut excursions and sampled finite differences are reported but
  are not silently clipped into a pass.

### Missing-spectrum identifiability

For every non-neutral synthetic colour, compute the maximum pairwise output
Delta E76 among base/positive/negative metamers and the base witness effect
Delta E76 from input. Define `ratio = metamer_spread / max(effect, 1.0)`.

- median metamer spread `<=2.0`;
- p95 metamer spread `<=5.0`;
- median ratio `<=0.25`;
- p95 ratio `<=0.50`.

These thresholds are engineering preregistration values, not published
psychophysical limits. They require the missing-spectrum uncertainty to be
materially smaller than the transform it purports to support.

## Branch decision

- all integrity, reconstruction, validity and identifiability gates pass:
  `stable_enough_for_fixed_visual_pilot`; H0B may freeze one exact LUT/output
  policy before touching A0 images;
- graph/reconstruction/validity fails: `numerically_invalid`; close H0A v1;
- those pass but any missing-spectrum gate fails:
  `canonicalizer_sensitive_unidentified`; close the exact spectral witness;
- raw gamut excursion alone does not open visual work. A passing witness may
  first require a separately frozen, data-independent output mapping leaf.

No result opens real-film fitting, training, stock calibration, authenticity,
LSM, production integration or a router. No failed gate may be rescued by
changing graph points, exposure, white balance, viewing light, metamer scale,
strength or thresholds in this version.

## Evidence bundle and DoD

- `configs/u5_r2h0a_velvia_datasheet_witness_v1.json` and its hash;
- versioned pixel-coordinate curve data and exact source hashes;
- extraction/digitization overlay, deterministic implementation and tests;
- two byte-identical numerical runs, manifest, report and array hashes;
- result document, decision JSON, tracker propagation and scoped commits.

Claim ceiling: deterministic datasheet-constrained spectral Look Approximation
feasibility and metamer sensitivity on a synthetic sRGB grid only.
