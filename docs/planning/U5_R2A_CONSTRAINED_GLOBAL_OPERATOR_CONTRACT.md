# U5.R2A constrained global colour-operator contract

Date frozen: 2026-07-23
DRPT node: `ULT > U5 > U5.R2 > U5.R2A`
Status: frozen before implementation

## 1. Question

Can the project establish one versioned, replayable global colour-operator
contract that is expressive enough for strongly stylised film-inspired colour,
but remains deterministic, finite, inspectable and numerically bounded?

This leaf does **not** ask whether any current dataset identifies a film-stock
response. Current stock pixels remain closed for fitting, training and latent
mode discovery. It also does not claim that numerical smoothness, invertibility
or gamut bounds imply semantic colour safety.

## 2. Parent evidence and placement

- The deterministic renderer and profile identity boundaries exist.
- `U5.CT1` already provides an orientation-preserving affine operator and
  strictly monotone rational-quadratic channel splines with inverse, Jacobian,
  serialization and dense-LUT bake tests.
- `U5.R1` shows that conventional image metrics do not replace the independent
  severe-artifact veto, and the conditioned SCIS route is closed.
- `RF2.S0`, `SF1.0B` and `SF1.3B` prohibit rescuing unidentified data by adding
  capacity.

The implementation belongs under `src/roll2film/` as a research operator
primitive. It must reuse the existing affine/spline implementation and extend
the existing LUT module instead of creating a parallel renderer. Product
integration under `U2.2/U2.3` remains a separate leaf.

## 3. Standards and method boundary

The ACES Common LUT Format specifies trilinear and tetrahedral interpolation
for `LUT3D` nodes and defines tetrahedral interpolation in its appendix:
https://docs.acescentral.com/clf/specification/

OpenColorIO exposes tetrahedral interpolation explicitly for 3D LUTs and warns
that callers should request a specific interpolation when stability matters:
https://opencolorio.readthedocs.io/en/v2.5.0/api/enums.html

Image-adaptive 3D LUT and SepLUT work establish LUTs and cascaded 1D/3D LUTs as
useful explicit photo-enhancement representations:

- https://arxiv.org/abs/2009.14468
- https://arxiv.org/abs/2207.08351

These sources justify the representation and interpolation choice. They do not
establish film authenticity, stock identity or semantic safety for this
project.

## 4. Frozen operator

The versioned order is:

```text
declared linear-sRGB input in [0, 1]
  -> orientation-preserving affine transform
  -> three strictly monotone rational-quadratic curves
  -> tetrahedral 3D LUT
  -> finite output in the declared [0, 1] headroom
```

Rules:

1. clipping is never implicit;
2. every stage declares one working space and domain;
3. invalid shape, non-finite data, domain excursion or inconsistent metadata
   fails before an output is returned;
4. serialization is versioned and deterministic;
5. the source array is never mutated;
6. the LUT audit records residual amplitude, first and second residual-axis
   differences, neutral-axis error and every tetrahedron's affine Jacobian
   determinant;
7. constraints are enforced by validation, not described only as a training
   regularizer;
8. no property in this leaf is a semantic or severe-artifact certificate.

Frozen machine-readable authority:
`configs/u5_r2a_constrained_global_operator_v1.json`.

## 5. Competing hypotheses

- `H1 representation pass`: the existing affine/spline core plus an exact
  tetrahedral LUT can satisfy identity, replay, domain and numerical-property
  gates without a production-path change.
- `H2 implementation defect`: interpolation, tetrahedron ordering, domain
  handling or serialization fails a golden/property test. Repair only the
  implementation; do not loosen a valid frozen gate.
- `H3 contract insufficient for safety`: a smooth, neutral-preserving,
  in-gamut, positive-Jacobian operator still maps a blue probe strongly toward
  purple. This is an expected counterexample and proves that numerical
  constraints do not replace `U4/U5.R1` visual/severe evaluation.

## 6. Required evidence

The implementation report must include:

- exact config hash and software commit;
- deterministic identity and golden vectors;
- monotone-curve roundtrip and positive-derivative result;
- vectorized tetrahedral output against an independent scalar reference;
- exact serialization replay;
- source nonmutation and invalid-input fail-closed tests;
- LUT residual, first/second-difference, neutral-axis and per-tetrahedron
  Jacobian summaries;
- two byte-identical audit reports;
- the frozen blue-to-purple counterexample passing the numerical contract while
  exceeding its semantic-colour-shift threshold;
- focused tests, complete CPU suite, compile and diff checks.

## 7. DoR, DoD and branches

### DoR

- current worktree and remote are synchronized;
- current stock fitting/training/LSM flags remain false;
- no other writer owns the files;
- contract and config are committed before implementation.

### DoD

All frozen property gates pass, the non-safety counterexample is reproduced,
the implementation stays isolated from renderer/profile/schema defaults and
the result is propagated to tracker, board and agent log.

### Branches

- valid automatic pass: close `U5.R2A` as a numerical representation pass and
  allow only a separately frozen `U5.R2B` global-frontier design;
- valid property failure: close or repair the primitive without fitting data;
- counterexample fails because it is rejected numerically: replace only with a
  preregistered counterexample that passes the unchanged intended contract;
- counterexample passes: record the required negative result; never call the
  operator safe;
- any severe artifact in a later real render: reject the candidate or policy;
  never relax the veto.

## 8. Explicit exclusions

- no current real-film pixels, FilmSet targets or owner anchors are fitted;
- no stock, process, scanner, exposure, push/pull or latent-mode inference;
- no training, neural router or direct RGB generator;
- no renderer, CLI, profile, recipe-schema or default integration;
- no calibration, stock-response or authenticity claim;
- no release, deployment, paid compute or participant action.

Ultimate remains active after this leaf regardless of pass or fail.

## 9. Pre-result contract repair

Before any audit result was produced, the first focused property run exposed
two specification defects. The counterexample matrix did not preserve the
neutral axis, and the first-difference wording could count the identity grid
step rather than the identity-plus-residual field. The matrix was replaced by
a convex, row-sum-one transform and the wording was made explicit. All numeric
thresholds, the blue-probe shift gate and every branch rule remain unchanged.
