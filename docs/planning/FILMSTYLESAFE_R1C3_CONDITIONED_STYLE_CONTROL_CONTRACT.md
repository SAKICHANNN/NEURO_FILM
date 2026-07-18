# U5.R1C3 conditioned style-control contract

Date: 2026-07-18

Node: `ULT > U5.R1 > U5.R1C3`

Status: frozen before implementation

## Question and parent evidence

Can an explicit, per-pair low-capacity colour canonicalizer remove legitimate
global style from SCIS without absorbing the three bound A0 proxy-severe
failures?

R1C2 establishes only that SCIS v0.1 reaches 3/3 sensitivity at zero false
positives against the hard-negative/external subset. With all five non-severe
controls, its threshold is set by scheme 53 and sensitivity falls to 1/3.
Schemes 53/55/56 are mandatory style/strength-path controls; excluding them is
forbidden.

## Competing hypotheses

- H1 — global-style explanation: a robust global colour relation explains the
  53/55/56 response while localized proxy-severe residuals remain.
- H2 — spatial-interaction explanation: the 53/55/56 response is dominated by
  texture/spatial interactions that a global relation cannot remove.
- H3 — absorption failure: the canonicalizer also explains the localized
  failures, improving false positives only by destroying sensitivity.

## Frozen candidates

For every source/candidate pair independently, predict candidate Lab `a,b`
from source Lab using deterministic robust ridge regression. This is pairwise
canonicalization, not corpus/model training.

Capacity ladder:

1. affine basis: `1, L, a, b`;
2. quadratic basis: affine plus `L^2, a^2, b^2, La, Lb, ab`.

Lab inputs are scaled to approximately unit range. Fits use a deterministic
maximum 65,536-pixel spatial sample, ridge `1e-4`, Huber delta `0.04`, four
fixed IRLS iterations and no outcome-dependent parameter search. SCIS v0.1's
frozen smooth mask, component, residual-threshold and sparse-density formula
is applied to the canonical residual.

## Variables and controls

- primary variable: canonicalizer capacity (`none/v0.1`, affine, quadratic);
- positives: all three existing A0 proxy-severe members;
- negatives: all five existing non-severe members, including 53/55/56,
  RF2.C0 external style and legitimate halation/bloom;
- capacity negative control: affine must be reported even if quadratic wins;
- absorption control: every positive must remain above the maximum negative;
- deterministic/nonmutation/failure controls for fit inputs and parameters.

No member, role, label, threshold, operator or source may be added after result
inspection. No sibling 09/53/55/56 regression artifact is promoted to a new
independent sample in this leaf.

## Frozen decision rules

Evaluate strict `score > max(non-severe score)`.

- pass: the simplest candidate with 3/3 sensitivity at zero false positives
  across all five non-severe controls, and both hard-negative/external scores
  below all positives;
- weak-pass: 2/3 with all style controls retained; close calibration and allow
  only a separately frozen minimal failure-mechanism diagnostic;
- fail: 0/3 or 1/3, or any 53/55/56 exclusion is required; close the
  conditioned candidate;
- absorption failure: any positive falls at or below the maximum non-severe
  after a candidate that otherwise reduces style scores; classify H3 and
  forbid capacity expansion;
- invalid: inventory/hash/shape/finite/determinism/baseline reproduction fails;
  repair evidence only and do not interpret scores.

Quadratic may be retained only if affine is a valid fail/weak-pass and
quadratic is a full pass. If both pass, affine wins. If neither passes, do not
add cubic bases, LUTs, spatial networks, learned features or label-conditioned
thresholds.

## Required evidence

- reproduce the R1C2 baseline summary from the unchanged inventory;
- record every member's baseline/affine/quadratic score and fit diagnostics;
- repeat run must be byte-identical after volatile-field normalization;
- focused tests, complete CPU suite, compile and diff checks pass;
- config hash, software commit, inventory identity and claim ceiling recorded.

## Allowed branches

- full pass: retain only the simplest explicit A0 candidate and open a
  separately frozen resolution/generalization stress design; do not promote a
  safety gate;
- weak-pass: one minimal mechanism diagnostic may be designed;
- fail/H3: close R1C3 and keep SCIS descriptive/unpromoted; return to product
  artifact veto plus future A1 evidence gate;
- invalid: repair the exact evidence/implementation defect, then rerun the
  unchanged contract.

## Forbidden fallback and claim ceiling

Forbidden: dropping 53/55/56, training, hidden A1 population, recruitment,
paid annotation, tuning after A0 results, neural features, stock/authenticity
claims, population risk bounds or automatic production veto integration.

Claim ceiling: A0 proxy-label evidence about whether one fixed low-capacity
pairwise colour canonicalizer separates the already bound controls. It is not
human confirmation, a validated severe-artifact detector or a product gate.
