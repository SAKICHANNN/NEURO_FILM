# U5.R2AA1 Kodak 250D to 2383 Nuisance-Identifiability Contract

Status: **FROZEN BEFORE CURVE ANNOTATION OR NUMERICAL RESULT**

Node: `ULT > U5 > U5.R2 > U5.R2AA1`

DRPT level/mode: L2 / Mode A, exactly one primary writer.

## 1. Question

Can a clean-room spectral chain assembled from the official Kodak VISION3
250D and VISION 2383 graph families retain a strong, non-basic colour
direction when the major unobserved scene-spectrum, exposure-placement,
printer-light, density-to-dye and viewing assumptions are varied
prospectively?

This experiment tests robustness of one explicit Look Approximation
hypothesis. It does not estimate the real photochemical operator.

## 2. Activation and immutable boundaries

AA1 activates only if:

- AA0 decision branch is `bounded_nuisance_pilot_feasible`;
- the five source-file sizes and SHA-256 values in
  `configs/u5_r2aa0_kodak_negative_print_source_decision_v1.json` match;
- official CIE 1931 2-degree CMFs and D50/D55/D65/A illuminants are present
  with exact hashes;
- current stock pixels remain closed for fitting/training;
- no product image, owner anchor, external film profile or LUT participates in
  annotation, parameter choice or threshold selection.

Allowed data are the two technical PDFs, three Kodak setup/topology bulletins,
official CIE tables and a synthetic RGB grid. Everything else is forbidden.

## 3. Source transcription

Run `scripts/extract_kodak_250d_2383_datasheet_graphs.py` and bind the six
exact extracted PNG hashes from AA0.

Human-guided semantic traces are allowed because overlapping black curves and
labels make automatic connected-component tracing ambiguous. Before the first
numerical run:

1. record graph-axis value/pixel pairs;
2. record semantic curve points snapped to exact source ink;
3. render an overlay for every graph;
4. require axis residual at most `2.0` pixels;
5. require every annotation within `2.0` pixels of remaining source ink after
   known axes/grid lines are masked;
6. commit the curve-data file and its hash.

Minimum point counts:

- 12 per negative and print characteristic curve;
- 10 per negative and print sensitivity curve;
- 12 per separated dye curve;
- 12 each for the 250D midscale-neutral and D-min curves.

Pixel proximity proves transcription only. It does not prove physical
calibration or semantic curve identity beyond the recorded human-guided trace.

## 4. Fixed synthetic population

- encoded-sRGB Cartesian grid: 9 levels per channel, `729` colours;
- neutral ramp: 33 encoded levels;
- wavelength support: 380--720 nm at 5 nm;
- input colour state: display-relative sRGB interpreted only as a
  colourimetric coordinate;
- no real raster input;
- seed: `2026072801`.

The smooth bounded RGB-to-reflectance solve and observer-null-space metamer
construction reuse the clean-room H0A numerical method. For every grid colour,
retain:

- one smooth base reflectance;
- one positive feasible metamer;
- one negative feasible metamer.

All three must reconstruct the same D65 colour within the frozen H0A
colourimetric error limits. None is called the original scene spectrum.

## 5. Fixed negative and print chain

### 5.1 Negative exposure and density

Integrate D65, candidate reflectance and each digitized 250D sensitivity
layer. Normalize per layer at 18% neutral. Test exactly three neutral
placements on the published relative-exposure axis:

```text
2.0, 2.5, 3.0
```

Map the resulting log exposures through the digitized 250D channel
characteristic curves. No placement may be selected after output inspection.

### 5.2 Density-to-dye hypotheses

Test exactly two declared mappings:

1. `diagonal_peak_normalized`: assigned channel density maps directly to its
   corresponding peak-normalized yellow/magenta/cyan basis after D-min
   subtraction and neutral normalization;
2. `midscale_nnls_scaled`: non-negative basis scales are solved once from the
   digitized 250D midscale-neutral spectrum, then the same assigned-channel
   density ratios modulate those fixed scales.

The second mapping may reduce transcription mismatch; it is not an identified
analytical dye concentration.

### 5.3 Printer-primary hypotheses

Compute the negative transmittance and expose the digitized 2383 sensitivity
layers through exactly three additive-printer hypotheses:

1. `peak_gaussian_sigma20`: Gaussian primaries centred at the digitized 2383
   layer-sensitivity maxima with 20 nm sigma;
2. `peak_gaussian_sigma45`: the same centres with 45 nm sigma;
3. `tungsten_sensitivity_weighted`: official illuminant A multiplied by each
   normalized 2383 sensitivity layer.

Every family is independently rebalanced on the same 18% neutral negative so
the print characteristic curves hit the fixed H-61B Status-A LAD aims
`[1.09, 1.06, 1.03]`. The balance is a declared neutral timing hypothesis,
not measured printer lights or scene-to-scene grading.

Map print-layer exposure through the digitized 2383 characteristic curves.
Normalize assigned print dye amounts at the LAD point, combine the digitized
2383 visual-neutral CMY dye spectra, and calculate print transmittance.

### 5.4 Viewing hypotheses

Integrate print transmittance with CIE 1931 CMFs under exactly:

- D50;
- D55;
- D65.

Adapt D50/D55 XYZ to D65 using Bradford before conversion to linear sRGB.
These are viewing controls, not substitutes claimed to be the missing xenon
projector spectrum.

## 6. Nominal witness and nuisance decomposition

The nominal witness is frozen as:

```text
smooth base reflectance
neutral placement 2.5
midscale NNLS-scaled negative dye mapping
peak Gaussian sigma45 printer
D55 viewing
```

The complete nuisance ensemble is the Cartesian product:

```text
3 spectra x 3 placements x 2 dye mappings x 3 printers x 3 viewers
= 162 outputs per synthetic colour
```

For each axis, hold the other nominal settings fixed and report paired
Delta-E76 spread. Also report the full-ensemble medoid and maximum pairwise
spread. No nuisance axis may be dropped after viewing results.

## 7. Required controls

Compare the nominal witness against:

1. identity;
2. the existing joint EV/WB/contrast/saturation best-basic fit on the same
   synthetic grid;
3. a per-channel affine RGB fit;
4. a bounded positive 3x3 fit if its existing implementation is available
   without changing that operator's gates.

The primary non-basic residual is nominal output versus the best of controls
1--3. The positive 3x3 is reported separately so an unavailable optional
helper cannot silently change the primary decision.

## 8. Frozen gates

### 8.1 Source and colourimetric validity

- all source, graph and curve-data hashes exact;
- every graph/annotation gate in section 3 passes;
- base reconstruction Delta-E76 median/p95/max <= `.25/.75/2.0`;
- both metamers reconstruct at p95/max <= `.75/2.0`;
- median metamer spectral RMS >= `.01`;
- every result finite.

### 8.2 Visible non-basic effect

- nominal identity Delta-E76 median >= `7.0`;
- nominal best-basic residual median >= `4.9`;
- nominal per-channel-affine residual median >= `3.0`;
- at least 75% of grid colours have best-basic residual Delta-E76 >= `2.0`.

These are synthetic colour-grid salience gates, not product preference.

### 8.3 Nuisance identifiability

For each grid colour, divide paired nuisance spread by nominal identity effect,
with denominator floor `1.0`.

| Nuisance axis | Median ratio max | P95 ratio max |
|---|---:|---:|
| RGB spectral canonicalizer | `.25` | `.50` |
| neutral exposure placement | `.35` | `.75` |
| density-to-dye mapping | `.25` | `.50` |
| printer-primary hypothesis | `.35` | `.75` |
| viewing illuminant | `.25` | `.50` |
| complete 162-member ensemble | `.50` | `1.00` |

All six rows must pass. A strong nominal look does not survive if missing
physical variables can change it by a comparable or larger amount.

### 8.4 Numerical and neutral behaviour

- neutral output Y minimum step >= `-1e-6` for every nuisance member;
- neutral Lab chroma maximum <= `4.0` for the nominal witness and <= `8.0`
  over the full ensemble;
- nominal sampled inner-grid Jacobian determinant > `0`;
- nominal sampled Jacobian spectral norm <= `8.0`;
- raw linear-sRGB absolute maximum <= `4.0`;
- raw outside-cube component fraction <= `25%`;
- two complete reports byte-identical.

AA1 does not gamut-map, clamp, refit or render a photograph. A passing
out-of-cube result may open only a separately frozen bounded explicit-operator
refit.

## 9. Branches

| Branch | Meaning | Action |
|---|---|---|
| `source_or_trace_invalid` | source/axis/curve evidence fails | repair only factual extraction defects before any numerical run |
| `colourimetric_invalid` | RGB reconstruction/metamer construction fails | close; no visual or spectral rescue |
| `basic_collapse` | effect is mostly EV/WB/contrast/saturation/affine | close; do not call it film-specific |
| `nuisance_unidentified` | missing-spectrum/printer/view assumptions dominate | close the datasheet chain |
| `numerically_invalid` | neutral, range or Jacobian gate fails | close; no clamp/gamut-map rescue |
| `stable_nonbasic_candidate` | every gate and repeat passes | open AA2 bounded explicit refit on the synthetic operator only |

No AA1 branch opens real-film pixels, real-image visuals, calibration, LSM,
training or production integration.

## 10. Forbidden actions and claim ceiling

Forbidden:

- fitting or selecting any missing variable on project photographs, stock
  scans, FilmSet, owner anchors or existing style outputs;
- importing an external FPE LUT/profile or proprietary printer spectrum;
- changing trace identity, nuisance variants, nominal member, thresholds or
  controls after the first numerical output;
- neural, local, direct-RGB, clamp, gamut-map or higher-capacity rescue;
- calling a mode, exposure, timing or printer hypothesis observed truth.

Maximum claim:

> Repeatable synthetic-grid evidence that one Kodak-datasheet-constrained
> negative-to-print Look Approximation does or does not retain a strong
> non-basic colour direction under preregistered scene-spectrum,
> exposure-placement, density-mapping, printer-primary and viewing
> hypotheses. No real 250D response, real 250D-to-2383 operator, calibration,
> authenticity, scanner/projector profile, latent mode, product preference or
> production claim.
