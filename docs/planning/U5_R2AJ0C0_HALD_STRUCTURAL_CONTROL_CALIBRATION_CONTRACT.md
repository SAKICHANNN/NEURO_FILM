# U5.R2AJ0C0 — Hald structural control calibration

## Parent and question

Parent `U5.R2AJ0B2` has passed the exact, lane-aware, two-process archive
integrity gate. It authorizes only the design of a synthetic structural
screen. It does not authorize reading any primary CLUT pixel for metrics,
rendering a photograph, or making an aesthetic or stock claim.

`U5.R2AJ0C0` asks a narrower development question:

> Can one fixed Hald parser/interpolator and one preregistered set of
> structural, basic-adjustment, duplicate, strength-path and novelty metrics
> correctly separate analytic positive and negative controls before any of
> the 194 primary CLUTs are measured?

This is a controls-only calibration and conformance leaf. A pass opens only a
separately frozen `U5.R2AJ0C1` primary-bank contract. A failure closes C0 v1;
thresholds may not be rescued after primary results are inspected.

## Exact parent and zero-primary-access boundary

- parent decision:
  `configs/u5_r2aj0b2_haldclut_lane_aware_acquisition_decision_v2.json`;
- parent decision SHA-256:
  `c8007c6abd9495d9b48b33c4ad5718a6ebe83cf2c184d21ea2db5353f65377a3`;
- parent result commit:
  `e0f03f1fbc5e6fcc5695d29b532f8479628c5e3c`;
- retained archive SHA-256:
  `0ffca81f30c72d7bbb85adab0ed98bbdeb0e1034cd6f5986f9f993bb66999dc2`;
- frozen primary-path SHA-256:
  `83b22255e6eea6a99e779c72acf75917481f14d144d9e442b44b6d716256115f`.

C0 must verify the parent decision and its identities but must not open the
archive. Its canonical report must record:

- archive bytes read: exactly zero;
- primary CLUTs decoded: exactly zero;
- primary candidate metrics: exactly zero;
- photograph renders: exactly zero.

## Hald and precision semantics

For Hald level `L`, image side `S=L^3` and 3D cube side `N=L^2`. The flattened
pixel index is:

```text
p = r + N*g + N^2*b
```

Red changes fastest, then green, then blue. A row-major Hald raster therefore
needs the explicit red/blue axis transpose before `cube[r,g,b]` lookup.

All C0 controls use encoded sRGB inputs, encoded sRGB table samples and
trilinear interpolation. The formal blend strength is one. No hidden
linearization, ICC transform, gamma inference, smoothing or output clamp is
allowed. The analytic identity is float64 and exact; quantized identity is a
separate RGB8 Hald control.

The evaluator must retain a source table as `uint8` and gather only the eight
corners needed by each batch before float64 conversion. Probe batches may not
exceed 65,536 rows. The generated `N=144` and `N=256` identity conformance
checks must not instantiate `DenseLUT3D` or another complete float64 copy.

An in-memory, project-generated RGB16 TIFF fixture must preserve exact
samples through `tifffile`, including values above 255. This proves only the
precision boundary. It does not authorize the external TIFF or Negative root
controls as C0 operator input, and Pillow fallback is forbidden.

## Frozen synthetic populations

- style/basic cube: `33^3`;
- residual/strength/novelty cube: `17^3`;
- analytic trilinear Jacobian: the first `2^16` points of an unscrambled,
  three-dimensional Sobol sequence;
- gradients: 4,097 samples on the neutral line, all 12 cube edges and both
  diagonals of each of the six cube faces, for 25 total lines;
- native identity conformance sides: `17`, `33`, `144`, `256`.

The Jacobian is the analytic derivative of the native trilinear cell. It must
be checked against analytic identity, an R/G axis swap and central finite
differences on an interior small-cube population. These probes are a sampled
screen, not a proof about every native cell or continuous topology.

## Frozen controls

The control Hald side is 33. RGB8 tables use `clip -> multiply by 255 ->
numpy.rint -> uint8`.

Expected-safe controls:

1. exact analytic float64 identity reference;
2. quantized RGB8 identity;
3. exposure-only basic adjustment;
4. bounded exposure plus white-balance basic adjustment;
5. contrast-only basic adjustment;
6. saturation-only basic adjustment;
7. joint exposure/WB/contrast/saturation adjustment;
8. one smooth warm non-basic direction at strengths `0.5`, `0.75`, `1.0`;
9. an exact duplicate of warm `1.0`;
10. one distinct smooth cool non-basic direction at strength `1.0`.

The five-parameter basic family is the existing linear-RGB diagnostic:
log-exposure, two zero-sum log-WB coordinates, log-contrast and
log-saturation. It does not contain a learned or global-luma curve.

Expected-negative controls:

1. exact R/G axis swap;
2. hard endpoint clipping;
3. seven-level staircase/posterization;
4. one centre-cell magenta spike in an otherwise quantized identity table.

The pre-contract control-only probe rejected a `0.25` warm strength as a
positive strength-equivalence control because RGB8 quantization reduced its
effect cosine below the already proposed threshold. The control is omitted;
the threshold is not weakened. No primary CLUT metric was read during this
choice.

## Per-operator gates

Safety is conjunctive and precedes style:

| Metric | Frozen gate |
|---|---:|
| finite samples | all |
| output range | `[0,1]` |
| interior input margin | `1/16` |
| endpoint epsilon | `0.5/255` |
| new interior endpoint fraction | `<= 0.01` |
| negative determinant threshold | `< -1e-5` |
| negative Jacobian fraction | `<= 0.005` |
| sampled minimum determinant | `>= -0.05` |
| maximum Jacobian spectral norm | `<= 20` |
| neutral linear-luma reversal | `<= 2e-5` |
| gradient RGB gain p99 | `<= 10` |
| gradient RGB gain maximum | `<= 20` |
| gradient maximum channel second difference | `<= 0.5/255` |
| native maximum adjacent code step | `<= 64` |
| native maximum second code difference | `<= 32` |

An eligible non-basic look must additionally reach:

- median style Delta E76 versus analytic identity `>= 3.0`;
- median residual after the frozen joint-basic fit `>= 1.5`.

Identity and basic controls are expected to be structurally safe but
ineligible as non-basic looks. Failure of one future primary candidate will
reject that candidate only; there will be no whole-bank safety veto.

## Duplicate, strength and novelty contract

The `17^3` encoded-sRGB population defines each effect as `output - analytic
identity`.

1. exact output/signature hashes identify duplicates;
2. positive-scale effect pairs are one strength path only when cosine is at
   least `0.995`, explained energy at least `0.99`, and bilateral normalized
   residual at most `0.10`;
3. each operator is fit once with the frozen joint-basic diagnostic;
4. residual signature is `Lab(output) - Lab(best_basic_output)`;
5. pairwise median residual-signature Delta E76 of at least `1.0` is novel;
6. connected components merge duplicate and strength-equivalent paths before
   novelty selection.

The warm `0.5/0.75/1.0` controls and exact duplicate must form one component.
The cool control must remain a separate component. Basic controls must not
be counted as non-basic novelty. Names, brand strings, stock strings and
filename suffixes are forbidden features.

Future C1 survivor selection is capped at 12 representatives:

1. seed by maximum non-basic residual;
2. repeatedly maximize the minimum residual-signature distance to selected
   representatives;
3. tie-break by style, safety margin, then lexical POSIX path.

The policy is frozen here but is not executed against primary candidates in
C0.

## Automatic pass, branches and DoD

C0 passes only if:

- strict parent identities pass;
- all Hald index, variable-side, interpolation and Jacobian oracle checks
  pass;
- the RGB16 in-memory conformance fixture preserves exact samples and tags;
- every expected-safe structural control passes;
- the expected non-basic controls alone satisfy style/non-basic eligibility;
- each negative control is rejected by its preregistered intended gate;
- exact duplicate, warm strength component, cool separation and basic
  non-novelty expectations pass;
- candidate-order permutation leaves components and representatives
  unchanged;
- two fresh processes write byte-identical canonical reports;
- zero primary/archive/photograph access is recorded.

Branches:

- pass: commit evidence, then freeze C1 before reading any primary CLUT
  metric;
- control/oracle/repeat failure: close C0 v1 and keep C1 forbidden;
- ambiguous control separation: close the metric design; do not inspect the
  primary bank to tune it.

DoD includes focused tests, adjacent evaluator tests, the complete CPU suite,
strict canonical evidence validation, a scoped commit/push and governance
propagation.

## Forbidden and claim ceiling

Forbidden in C0:

- opening or decoding any primary CLUT body;
- rendering or visually inspecting a photograph;
- visually ranking synthetic controls;
- lowering a threshold after primary inspection;
- smoothing, clamping, reducing strength or deleting a failed primary;
- treating preset names as stock/process/scanner/push/pull/exposure truth;
- using the bank as a teacher or training target;
- fitting current real-film pixels, training, routing, LSM or production
  integration;
- distributing the retained external archive.

Claim ceiling:

> Controls-only conformance and preregistration evidence for a sampled,
> encoded-sRGB, trilinear Hald structural screen. No external CLUT has been
> measured, applied to a photograph, shown safe, preferred, authentic,
> calibrated or production-ready.
