# U5.R2AJ0C0B — corrected Hald structural control calibration

## Status and parent

This is a separately versioned correction after C0 v1 closed during
pre-execution review. V1 files and decision remain historical and unchanged.
No primary CLUT metric informed this correction.

Parents:

- AJ0B2 exact lane-aware archive-integrity pass;
- C0 v1 pre-execution contract failure at commit `cdd20cd`;
- zero archive bytes, primary decodes, primary metrics and photograph renders
  in C0 v1.

C0B asks only whether the synthetic evaluator, controls and evidence chain are
internally conformant. It still cannot open the retained archive.

## Probe populations versus legal Hald geometry

The following concepts are separate:

- metric probe cubes have sides 33 (style/basic) and 17
  (residual/strength/novelty); they are analytic sample populations and are
  not Hald rasters;
- a legal Hald level `L` has raster side `S=L^3`, cube side `N=L^2`, and
  `S^2=N^3` RGB samples;
- generated Hald parser conformance uses exact `(L,N,S)` tuples
  `(2,4,8)`, `(6,36,216)`, `(12,144,1728)` and `(16,256,4096)`;
- all quantized control operators use the legal `L=6`, `N=36`, `S=216`
  geometry.

Flat pixel index is exactly `r + N*g + N^2*b`: red fastest, green next, blue
slowest. Row-major Hald raster decode must recover `cube[r,g,b]` exactly.
Generated identity node value is exactly
`(r/(N-1), g/(N-1), b/(N-1))` before RGB8 quantization. The parser must
derive legal `L/N` from square raster side `S` and reject any side that is not
an integer cube of an integer level greater than one. The tiny `N=4` control
is the high-sensitivity flatten/channel/order oracle.

The N=144/256 identity tests retain uint8 storage and only gather the eight
needed corners in batches of at most 65,536. A complete float64 copy is
forbidden.

For every legal identity geometry:

- parser round-trip table equality must be byte exact;
- maximum absolute interpolated channel error versus analytic identity is
  `<= 0.00196078431373`;
- the error is evaluated on the 33-cube, 17-cube, Sobol and gradient probe
  union, in batches, with no output clamp.

## Exact operator and negative-control formulas

All table generation is float64 encoded sRGB, then:

```text
clip to [0,1] -> multiply by 255 -> numpy.rint -> uint8
```

The LUT is applied in encoded sRGB with native trilinear interpolation,
strength 1.0 and no post-interpolation clamp.

The smooth hue controls are:

```text
y = x + 0.75 * strength * x * (1-x) * (M @ x)
```

with exact warm/cool matrices in the config.

Negative controls:

- axis swap: output `(g,r,b)`;
- hard clip, componentwise and before RGB8 quantization:
  - `x <= 0.1 -> 0`;
  - `x >= 0.9 -> 1`;
  - otherwise `x` unchanged;
- seven-level staircase:
  `numpy.rint(x * (7-1)) / (7-1)` before RGB8 quantization;
- single spike: replace exact cube node `(18,18,18)` in the quantized identity
  with RGB8 `(255,0,255)`.

Boundary inclusion and generation order may not vary.

## Frozen basic and Lab semantics

Basic-control generation is fully specified in the v2 config and matches the
project's existing five-parameter family:

- decode encoded sRGB to linear sRGB;
- pivot is median linear luma on the native N=36 encoded grid;
- build exposure, zero-sum log-WB diagonal, contrast about that pivot and the
  luma-preserving saturation matrix in the exact frozen order;
- apply in linear RGB, clip only for encoded output, encode sRGB, then perform
  the one RGB8 quantization.

Evaluation must call
`src.roll2film.baselines.fit_joint_basic_adjustment` and
`src.eval.spectral_film_lut_bank.matched_basic_output` at their pinned source
hashes. The least-squares parameterization, summary vector, bounds, tolerances
and iteration cap are repeated in the config; a locally simplified fit is not
allowed.

Lab is D65 CIELAB using the pinned
`src.eval.velvia_datasheet_witness.encoded_srgb_to_linear`, exact sRGB-to-XYZ
matrix, D65 white and `xyz_to_lab`. No scikit-image Lab substitute, D50
adaptation or input clipping beyond the matched-basic API is allowed.

## Frozen samples, thresholds and expectations

Unchanged controls-only metric populations:

- analytic style/basic probe cube: `33^3`;
- analytic residual/strength/novelty probe cube: `17^3`;
- native analytic trilinear Jacobian: first `2^16` points from an
  unscrambled Sobol sequence;
- 4,097 samples on 25 fixed lines: neutral, 12 edges and 12 face diagonals.

Analytic-Jacobian conformance additionally uses every N=4 cell crossed with
local fractions `{0.25,0.5,0.75}^3`, for 729 fixed points. Identity must match
the 3x3 identity Jacobian and R/G swap its exact permutation Jacobian to
maximum absolute error `1e-12`. A float64 N=4 smooth warm table is checked by
central differences with step `1e-7/(N-1)`, maximum absolute error `1e-7`,
and maximum relative error `1e-6` using denominator
`max(abs(finite_difference),1e-8)`. Both plus and minus probes must remain in
the same native cell.

Safety thresholds remain those chosen before primary access:

| Gate | Value |
|---|---:|
| finite and range | all, `[0,1]` |
| interior margin / endpoint epsilon | `1/16`, `0.5/255` |
| new interior endpoint fraction | `<= 0.01` |
| negative determinant threshold/fraction | `< -1e-5`, `<= 0.005` |
| sampled minimum determinant | `>= -0.05` |
| maximum spectral norm | `<= 20` |
| neutral linear-luma reversal | `<= 2e-5` |
| gradient gain p99 / maximum | `<= 10`, `<= 20` |
| gradient max channel second difference | `<= 0.5/255` |
| native adjacent / second code difference | `<= 64`, `<= 32` |
| style median Delta E76 | `>= 3.0` |
| joint-basic residual median Delta E76 | `>= 1.5` |

Identity and all five basic controls must be structurally safe and non-basic
ineligible. Warm `0.5/0.75/1.0`, its exact duplicate and cool `1.0` must be
structurally safe and non-basic eligible. Each negative must fail at least one
of its exact intended checks.

## Duplicate, strength and novelty conformance

Only structurally safe non-basic eligible controls enter equivalence:

1. exact output/signature hash joins the warm duplicate;
2. positive-scale equivalence requires effect cosine `>=0.995`, explained
   energy `>=0.99` and bilateral normalized residual `<=0.10`;
3. warm `0.5/0.75/1.0` plus duplicate must form exactly one component;
4. exact warm `1.0` is the frozen C0B-only representative of that known
   control component; this does not define the future general C1 rule;
5. cool must form a separate component;
6. residual signature is
   `Lab(output) - Lab(frozen_joint_basic_fit(output))`;
7. only after exact/strength component collapse, warm `1.0` versus cool
   `1.0` must have median residual-signature distance `>=1.0`.

Basic and negative controls are explicitly excluded from component discovery
and future selection. Exact results must be invariant under config order,
reverse order and SHA-256-of-ID order.

The cap-12 primary survivor algorithm is not part of C0B. Its intra-component
representative, normalized safety margin, max-min selection and tie breaks
must be frozen completely in C1 after C0B passes but before primary access.

## Strict evidence

Each single process writes two canonical strict-JSON files:

1. manifest schema
   `u5-r2aj0c0b-hald-structural-control-manifest-v2`;
2. report schema
   `u5-r2aj0c0b-hald-structural-control-report-v2`.

Both bind:

- raw and canonical config SHA-256;
- identical pre-run and post-run raw config SHA-256 to reject TOCTOU;
- exact software commit and tracked-worktree-clean fact;
- exact parent and v1-close identities;
- exact runtime versions;
- ordered control-ID hash and every source-table, output, residual,
  Jacobian, gradient and native-difference signature hash;
- legal Hald/parser/Jacobian/TIFF conformance identities;
- exact probe/Sobol/gradient/native-grid population hashes;
- access counters and forbidden-access booleans;
- state `control-run-only-non-promoting`;
- visual, primary, photograph and promotion booleans fixed false.

Canonical evidence contains no absolute path, process ID, wall clock or
timing. Every schema uses exact recursive key sets and strict JSON scalar types:
Boolean is never accepted as integer, unknown keys fail and non-finite numbers
fail.

The repeat evaluator validates each child independently, requires manifest
bytes equal and report bytes equal, and writes schema
`u5-r2aj0c0b-hald-structural-control-repeat-decision-v2`. A single child can
never contain `automatic_pass` or promote; only the repeat decision may set
`automatic_pass` and `c1_contract_design_allowed`.

The evaluator API accepts no archive path. Tests must patch archive/ZIP and
external data-root opens to fail immediately, so the zero-access ledger is
backed by the call boundary rather than an unrestricted self-report.

Canonical JSON is UTF-8
`json.dumps(sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)`.
Canonical array SHA-256 is computed from a C-contiguous array using the
canonical JSON header `{"dtype": array.dtype.str, "shape": [...]}`, one NUL
byte, then exact C-order array bytes. The config freezes independent expected
hashes for both probe populations, Sobol/gradient/finite-difference probes,
legal identity Hald rasters, the RGB16 fixture and every generated control
table. A two-process agreement is insufficient unless all independent
expected hashes also match.

## Automatic pass and branches

C0B passes only when:

- parent, config, commit, runtime and strict evidence bindings pass;
- all four legal Hald geometries round-trip and interpolate identity within
  their exact quantization bound;
- analytic trilinear Jacobian matches identity, R/G swap and central finite
  differences;
- generated RGB16 TIFF preserves exact uint16 samples/tags without Pillow;
- all expected positive/negative, eligibility, duplicate, strength, novelty
  and permutation expectations pass;
- both child manifest/report pairs are byte-identical;
- access counters are exactly zero for archive/primary/photograph use.

Pass opens only C1 contract design. Failure closes C0B v2 without primary
inspection or threshold rescue.

## Forbidden and ceiling

No archive opening, primary metric, photograph render, visual ranking,
threshold retuning, filename feature, smoothing/clamping rescue,
pseudo-teacher use, film-pixel fitting, training, routing, LSM, integration,
distribution, release or deployment.

Claim ceiling:

> Strict controls-only conformance for a sampled encoded-sRGB trilinear Hald
> structural evaluator. No external CLUT has been measured, rendered,
> visually reviewed, validated, preferred, calibrated or integrated.
