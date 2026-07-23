# U5.R2E0 density-domain explicit operator contract

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2E0`  
Status: **frozen before implementation or audit results**

## Question

Can a clean-room, continuous negative-to-print density chain provide a
replayable explicit parameter space with materially non-basic colour
directions, while remaining bounded and free of spatial/image-generation
artifacts?

This leaf builds and audits a primitive. It does not fit a stock or select a
product look.

## Primary-source synthesis

- Geigel and Musgrave's photographic-development simulation separates
  exposure, characteristic density response, spectral sensitivity, resolution
  and granularity. This leaf uses only the global colour/density portion.
- The 2025 *Emulating Emulsion* poster demonstrates a compact
  matrix -> per-layer sigmoid -> matrix formulation, but its Velvia result is
  fitted to controlled paired RAW/chart/scan measurements. The project has no
  equivalent pairs.
- Fujifilm and Kodak technical guides define characteristic curves as
  process-specific density versus log exposure, and distinguish spectral
  sensitivity from spectral dye-density absorption.
- The isolated RF2.C0 spektrafilm control shows that a fuller spectral chain
  can be strong and artifact-clean, while also warning that data-sheet curves
  and heuristic couplers are not measured stock truth.

The project implementation is clean-room and uses no external source code,
profile numbers, LUTs or generated outputs. This is not a novelty claim over
the cited formulations.

## Frozen interpretation

Input and output are explicit D65 linear-sRGB floats in `[0,1]`.

The negative-to-print chain is:

1. positive row-stochastic `capture_matrix` maps digital RGB exposure into
   three virtual emulsion-layer exposures;
2. each layer applies an increasing logistic characteristic curve in
   `log2(exposure + floor)` to produce negative dye density;
3. positive row-stochastic `dye_absorption_matrix` mixes layer densities into
   transmitted-light absorption;
4. Beer-Lambert-style transmission is `10 ** (-density)`;
5. positive row-stochastic `print_matrix` maps transmitted light into three
   virtual paper-layer exposures;
6. a second increasing logistic characteristic curve produces paper density;
7. paper reflectance is `10 ** (-density)`;
8. theoretical all-black/all-white endpoint responses normalize each channel
   once; there is no per-image statistic, auto exposure or clipping;
9. optional strength is a linear-RGB blend with identity in `[0,1]`.

The two density/transmission inversions make the final negative-to-print
mapping increase with scene exposure. Matrices and curves are explicit,
versioned parameters. There is no spatial operation.

## Parameter contract

- all three matrices are non-negative, row-stochastic and have determinant
  `>=0.2`;
- logistic slope is in `[0.2,4.0]`;
- log2 midpoint is in `[-12,2]`;
- maximum density is in `[0.2,4.0]`;
- minimum density is zero in v1;
- exposure floor is exactly `2^-16`;
- theoretical per-channel endpoint span is `>=0.05`;
- strength is in `[0,1]`;
- no implicit clamp is allowed inside the chain.

Invalid shape, range, non-finite parameter, endpoint inversion or unsupported
colour state fails closed.

## Frozen witnesses

Audit one neutral reference and four deliberately distinct, data-independent
parameter variants:

- `neutral_density_reference`;
- `warm_dense_like`;
- `cool_soft_like`;
- `cyan_shadow_warm_highlight_like`;
- `cross_processed_like`.

These are visual descriptions, not film stocks or latent modes. The variants
change only matrices and characteristic-curve parameters specified in the
config.

## Property gates

1. black maps to black and white maps to white within `1e-12`;
2. every 17-cube output is finite and within `[0,1]` without a clamp;
3. minimum finite-difference Jacobian determinant on interior 17-cube probes
   is `>0`;
4. minimum per-channel directional derivative is `>=0`;
5. serialization replay and repeated audit are byte-identical;
6. source arrays are not mutated;
7. strength zero is exact identity; strength one is exact full operator;
8. tiled/partitioned application is byte-identical to full application;
9. invalid parameters and RGB fail closed;
10. every non-neutral witness has uniform-grid RGB RMSE `>=0.03` from identity;
11. minimum pairwise witness RGB RMSE is `>=0.015`;
12. each witness residual after the best global affine RGB fit is `>=0.005`,
    preventing the audit from passing on a matrix-only recolour.

Numerical regularity is not semantic safety. Passing opens only a separately
frozen R2E1 gold/stress style and severe-artifact frontier.

## Branches

- property failure: repair implementation errors only; otherwise close;
- diversity/basic-residual failure: redesign the parameter family before any
  image frontier;
- pass: freeze R2E1 against identity, safe-rich, margin-4 anchor56 and the
  external RF2.C0 control as comparisons;
- R2E1 may search bounded parameters but may not fit current real-film pixels
  or use any comparison output as a teacher;
- no result permits a named-stock, calibrated, measured-response or physical
  authenticity claim.

## DoD

- contract/config committed before code;
- independent scalar logistic/density reference tests;
- analytic/finite-difference monotonicity and endpoint checks;
- two byte-identical audits;
- focused and full CPU tests;
- results propagate with claim ceiling;
- scoped commits pushed; Ultimate Goal remains ACTIVE.
