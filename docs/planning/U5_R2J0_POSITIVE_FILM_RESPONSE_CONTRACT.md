# U5.R2J0 positive-film-response operator contract

Date: 2026-07-26

Node: `ULT > U5 > U5.R2 > U5.R2J0`

Status: **frozen before implementation or audit results**

## Question

Can a compact positive-film functional form—two bounded colour matrices around
three independent log-exposure response curves—produce multiple clearly
non-basic, continuous and artifact-resistant colour directions without the
extra negative-to-print inversion that dominated the failed I0/I1 route?

This is an architecture witness, not a stock fit or product selection.

## Evidence and non-duplication

The official *Emulating Emulsion* project describes a
`matrix -> three sigmoids -> matrix` RAW-to-scanned-positive-film model with
about 30 parameters. SF2.6R confirms that its measurements, fitted parameters,
code and reusable data licence are not public. This leaf therefore copies no
parameters, code, LUT or output. It implements the published mathematical
family independently and uses only the original synthetic witnesses frozen in
the config.

The leaf is distinct from:

- R2E0, which models a colour-negative plus a second paper-density stage with
  three matrices and two response layers;
- U2.2B/I0/I1, whose negative-to-print composition produced a strong green
  neutral-axis cast and became weak/basic after that cast was removed;
- RF2.C0, which remains an external spectral Look Approximation control.

## Frozen operator

For D65 linear-sRGB input `x` in `[0,1]`:

1. `e = M_capture x`;
2. `h = log2(e + 2^-16)`;
3. each virtual film layer applies
   `r_c = A_c / (1 + exp(-k_c (h_c - m_c)))`;
4. `y_raw = M_scan r`;
5. responses to theoretical `[0,0,0]` and `[1,1,1]` inputs normalize each
   output channel to exact black and white;
6. optional strength is a linear-RGB interpolation with identity.

Both matrices are non-negative and row-stochastic with determinant at least
`0.2`. Slopes are in `[0.2,4]`, midpoints in `[-12,2]`, maximum responses in
`[0.2,4]`, and each raw endpoint span is at least `0.05`. There is no clipping,
spatial operation, per-image statistic, auto exposure or learned RGB output.

## Frozen witnesses

- `neutral_positive_reference`;
- `warm_highlight_like`;
- `cool_dense_like`;
- `cyan_shadow_warm_highlight_like`;
- `cross_bias_like`.

Names are visual descriptions only. They are not stocks, modes, exposure
states, illuminants, processes or scanner profiles.

## Numerical and diversity gates

The immutable 17-cube audit requires:

1. exact black/white normalization within `1e-12`;
2. finite unclamped output in `[0,1]`;
3. strictly positive interior finite-difference Jacobian determinants;
4. non-negative per-channel directional derivatives;
5. exact serialization replay and two-run report identity;
6. exact strength-zero identity, full-strength parity and partition parity;
7. no source-array mutation and fail-closed invalid input/parameters;
8. every non-neutral witness RGB RMSE from identity at least `0.03`;
9. minimum pairwise witness RGB RMSE at least `0.015`;
10. residual after best global affine RGB fit at least `0.005`.

Passing these gates establishes only a useful bounded nonlinear representation.
It opens a separately frozen real-image gold/stress frontier. It does not
establish aesthetic value or severe-artifact safety.

## Branches

- implementation defect: repair without changing the config;
- representation/diversity failure: close this exact family and do not tune
  after seeing the result;
- pass: freeze one real-image frontier using the existing gold/stress set,
  inherited style/non-basic/clipping gates and safe-rich/R2E1 comparators;
- any future fit requires new rights-cleared paired evidence and a separate
  preregistered stock/group design;
- no result opens training, LSM, production integration or a named-stock claim.

## DoD

- contract/config committed before implementation;
- isolated reusable module plus independent scalar references;
- two byte-identical formal audits;
- focused and complete CPU tests;
- result and claim ceiling propagated;
- scoped commits pushed; Ultimate Goal remains active.
