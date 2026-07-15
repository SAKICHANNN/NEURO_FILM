# Roll2Film CT5 baseline and internal evaluator contract

> Date: 2026-07-15
> Node: `ULT > U5.CT5`
> State: frozen before decoding internal-dev target payloads

## Purpose

Determine whether a pooled explicit L2 colour operator adds anything beyond
ordinary exposure, white balance, contrast, saturation and strong classical
unpaired transfer at matched observed style strength. This is a FilmSet recipe
experiment, not evidence of physical film or named-stock fidelity.

The machine-readable authority is
`configs/roll2film_ct5_baselines.json`. Any change to candidate membership,
sampling, folds, metric direction or promotion logic creates a new experiment
ID and invalidates confirmatory use of the old internal-dev results.
Generation-v1 config SHA-256 is
`b85e1a58cb07ce4138fb45a514051630700cfbc949c1798ca2024a97c50090c1`.

## Access boundary

- fitting may read only `source_train.jsonl` and `target_train.jsonl`;
- source and target identities and duplicate clusters are disjoint;
- `internal_dev_lockbox.jsonl` is opened only by the evaluator after this
  contract is committed;
- internal-dev clusters are deterministically divided 50/50 into pilot and
  confirmatory folds before target metrics are computed;
- pilot may set implementation tolerances and strength strata; candidate
  parameters are fit from training manifests, never paired pilot images;
- confirmatory targets are not used for tuning;
- `final_628_lockbox.jsonl` is forbidden to every CT5 process and remains
  unopened until the complete CT5-CT7 policy is frozen.

Every loader verifies the four frozen manifest hashes. An unexpected role,
hash, content/cluster overlap, missing ICC conversion or final-lockbox path
fails closed.

## Candidate ladder

The initial CPU-safe ladder is fixed in this order:

1. identity;
2. exposure-only, WB-only, contrast-only and saturation-only;
3. joint WB + contrast + saturation (`best-basic` family);
4. Lab mean/std;
5. per-channel quantile/histogram transfer;
6. Gaussian/Bures affine transport;
7. deterministic sliced/iterative-distribution transport;
8. pooled affine-plus-monotone-spline L2 operator;
9. paired per-image oracle, evaluator-only and never deployable.

The basic family is fitted against the same unpaired target pixels and receives
the same image-equal sampling budget as every unpaired candidate. Sliced OT is
allowed to expose outlier, off-support and speckle failures; those are evidence,
not reasons to silently add local smoothing.

Image-adaptive LUT, SepLUT, NILUT, CanonCGT, StatLUT and learned selectors are
explicitly `not-run` in generation v1. They require a new frozen implementation
and compute contract; their absence must not be misreported as a win.

## Matched-strength rule

Style strength is measured independently of recipe fidelity. Primary strength
is median Delta-E 2000 from the input; secondary evidence includes mean
Delta-E, chroma displacement, luma-quantile displacement and linear-RGB RMS.

Pilot results define frozen strength strata. Comparisons are made within those
strata. Candidates are not post-hoc weakened by arbitrary RGB interpolation to
manufacture a favorable match. A later strength path must be valid inside its
operator family and receive a new contract/hash.

`best-basic` is the eligible basic candidate with the best pilot recipe
fidelity inside the candidate's strength stratum. A nonlinear/classical method
must report residual gain over this adversary; merely increasing saturation or
contrast cannot count as style discovery.

## Evaluation order

1. fail closed on non-finite output, role/hash/access violation, unstable
   operator, invalid Jacobian or undeclared clipping;
2. record automatic artifact canaries and full-resolution worst cases; these
   do not replace visual severe adjudication;
3. require the frozen style floor relative to matched `best-basic`;
4. compare hidden recipe fidelity using image/cluster bootstrap confidence
   intervals;
5. report paired-gap closure to the evaluator-only oracle;
6. prefer the simplest survivor. If pooled L2 cannot beat a classical/basic
   method, it is not promoted.

The recipe-fidelity primary endpoint is mean Delta-E 2000 to the hidden target.
Linear-RGB RMSE, sRGB PSNR and SSIM are secondary. Metrics are image-equal and
bootstrap duplicate clusters, not correlated pixels. Output strength, fidelity
and artifacts remain separate scorecards.

## Stop and promotion rules

- any confirmed severe artifact blocks promotion regardless of fidelity;
- confirmatory improvement over matched `best-basic` needs a cluster-bootstrap
  95% lower bound above zero;
- style strength must be at least the frozen matched-basic floor;
- identity collapse, saturation-only equivalence or no paired-gap closure stops
  the method claim;
- one winning pooled/classical operator is a valid product/research outcome;
  ML routing is not opened without at least two substantive modes and an Oracle
  gap;
- the official 628 is not a development fallback and cannot be opened after an
  ambiguous internal result.

## Required outputs

Generated ignored outputs must include the config/code/manifest hashes,
environment, exact sampled content IDs, pilot/confirmatory membership, fitted
operator bundles, per-image metrics, bootstrap summaries, failure masks and a
decision record. Public example images or weights remain release-review gated.
