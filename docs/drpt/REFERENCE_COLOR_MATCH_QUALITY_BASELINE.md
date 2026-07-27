# NFCM-A4 Known-Operator Quality Baseline

## Question

Can the v1 single-reference safe-Lab recipe recover one known global
photographic look across different image content?

This is a product-quality falsification leaf, not a film-stock authenticity
test. The target images were produced by the same existing deterministic
Velvia-look renderer, so the hidden target look is held constant while image
content changes.

## Frozen inputs

- Reference: styled target image `01` from
  `outputs/eval/baseline_current/velvia_50/after`.
- Sources: neutral images `01`, `02`, `03`, `05`, `07` and `09` from
  `outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/inputs`.
- Known targets: the corresponding six images from the same
  `baseline_current/velvia_50/after` run.
- Recipe: one fit from reference `01`, reused unchanged for all six sources.
- Output evidence is intentionally ignored under
  `outputs/reference_color_match_quality/velvia_known_operator_v1`.

The same-content `01` row is a positive control. The other five rows are the
cross-content test.

## Results

Pixel diagnostics compare each neutral source and reference-matched output to
its same-content known target in D65 CIELAB. Positive improvement means the
reference match reduced median Delta E76.

| Image | Source to target median | Match to target median | Median improvement | Match p95 |
|---|---:|---:|---:|---:|
| 01 positive control | 9.78 | 3.93 | +59.8% | 8.41 |
| 02 | 9.47 | 9.83 | -3.8% | 21.96 |
| 03 | 10.30 | 16.91 | -64.1% | 23.39 |
| 05 | 7.30 | 9.58 | -31.4% | 16.19 |
| 07 | 7.03 | 11.24 | -59.8% | 17.42 |
| 09 | 8.11 | 22.35 | -175.4% | 25.02 |

The positive control passes, but all five cross-content rows regress. Gamut
compression touched 21.1% to 63.2% of pixels in four of those five rows and
50.4%/63.2% in the two strongest failures.

The committed `evaluate_known_operator_batch` harness reproduces the six rows
and reports one improved / five regressed samples, aggregate median improvement
`-45.6%`, worst improvement `-175.4%`, and maximum newly introduced encoding
boundary fraction `10.9%`. Gamut-adjusted fraction and new encoding-boundary
fraction are separate diagnostics and must not be conflated.

The product guard freezes two independent v1 tail thresholds:

- maximum gamut-adjusted fraction: 25%;
- maximum newly introduced encoding-boundary fraction: 5%.

On this slice it accepts the same-content positive control and rejects all five
cross-content failures. Images `02`/`03` fail both gates, `05`/`07` fail the
new-boundary gate, and `09` fails the gamut-adjustment gate. Rejected files are
delivered as identity fallback with the candidate diagnostics and rejection
reasons preserved.

This is a safety result, not look-recovery success. Returning the source avoids
the visible failure but does not satisfy the requested aesthetic match.

### Six-reference matrix

The same six known-look targets were then rotated through the reference role,
producing 36 full-resolution reference/source combinations:

| Slice | Count | Improved | Regressed | Guard accepted | Regressions accepted | Improvements rejected |
|---|---:|---:|---:|---:|---:|---:|
| Same-content positive controls | 6 | 6 | 0 | 5 | 0 | 1 |
| Cross-content | 30 | 5 | 25 | 3 | 0 | 2 |

Same-content median improvement is `+60.0%` with worst `+39.7%`.
Cross-content median improvement is `-92.2%` with worst `-344.9%`.

Within this bounded slice, the guard has zero false acceptance among the 25
known regressions. It is conservative: one same-content control and two
cross-content improvements are also rejected. The rejected same-content
control needs an 11.8% new-boundary allowance; raising the current 5% threshold
that far would also admit known regressions at 7.7% and 10.7%. Therefore the
threshold is retained as a safety veto rather than relaxed to optimize recall.

This matrix is not an independent photographic preference study: all targets
come from one existing deterministic Velvia-look run. It calibrates failure
detection and demonstrates the content-dependence problem; it does not prove
general user preference or real-film fidelity.

### Full-covariance Gaussian/MKL challenger

A stronger non-ML distribution baseline was evaluated on the exact same 6x6
matrix. For each reference/source pair it fits the unique symmetric
positive-definite Gaussian optimal-transport map in D65 Lab, uses the same 85%
luma strength, then applies the existing source-relative gamut compression and
the frozen product guard.

| Algorithm | Same-content median | Cross-content improved | Cross-content median | Cross-content worst | Guard-accepted cross-content |
|---|---:|---:|---:|---:|---:|
| v1 channel mean/std | +60.0% | 5/30 | -92.2% | -344.9% | 3 |
| Gaussian/MKL covariance | +65.2% | 3/30 | -109.5% | -332.2% | 2 |

The covariance map improves same-content fit but worsens held-out-content
recovery and produces 27/30 cross-content regressions. This is the expected
failure of treating each scene's full colour covariance as photographic style.

Decision: reject full-covariance Gaussian/MKL as a product algorithm. It remains
a research comparator only. Adding histogram iterations, higher moments or
more aggressive distribution matching is not an allowed rescue without new
source-use/content-invariance evidence.

Autonomous visual inspection agrees with the metric direction:

- image `02` becomes too dark and warm;
- image `03` loses the target's cooler ground/foliage relation and raises the
  flower scene globally;
- image `07` desaturates the blue sky and darkens the architecture;
- image `09` raises shadows and introduces a cyan/grey cast instead of the
  target's deep blue-black response.

No geometry or texture rewrite was observed, as expected from the pointwise
safe-Lab renderer. The failure is colour-operator identification, not spatial
generation.

## Decision

The v1 algorithm is rejected as the final photographic matcher. It remains a
safe deterministic challenger behind an identity fallback and a
product-contract baseline only.

The evidence distinguishes two properties:

1. same-content fitting can move toward a target;
2. reference-image global moments do not identify a shared look across
   different content.

Therefore parameter tuning, stronger global moment matching, or declaring the
safe-Lab recipe a champion is forbidden. A promotable challenger must estimate
a content-independent grade through a canonical pivot or an equivalent
identified representation, then emit a bounded explicit operator.

## Promotion gate

A challenger may replace v1 only if it:

- uses exactly one reference and one frozen operator/grade identity across N
  sources;
- improves the held-out cross-content aggregate over both neutral source and
  v1 without relying on the known targets at inference;
- passes severe-artifact, clipping/gamut, neutral, skin, sky, highlight and
  batch-consistency gates;
- preserves deterministic recipe replay and the existing transactional file
  boundary;
- retains the `reference-look` claim ceiling;
- reports synthetic/known-operator evidence separately from real-world blind
  aesthetic preference.

The promotion harness is pure `WorkingImage` evaluation: targets are consumed
only after rendering, shape/colour state/gamut violations fail closed, and the
reported automated metrics never promote a candidate without the independent
visual/aesthetic gate.

### Machine-executable promotion adjudication

The A4 boundary is now executable rather than prose-only. The streaming runner
fits all six reference recipes, evaluates the 30 cross-content rows one at a
time, and then renders a frozen structural-colour probe through every recipe.
It therefore does not hide reference-dependent tail risk behind one selected
recipe and does not retain 30 full-resolution candidates in memory.

The formal result reproduces the earlier matrix:

- 5/30 improved and 25/30 regressed;
- median improvement `-92.1932%`;
- worst improvement `-344.8730%`;
- maximum new-boundary fraction `21.8555%`;
- two of six recipe probes pass and four fail;
- worst recipe-probe new-boundary fraction `12.6946%`;
- no tone reversal or plateau was detected;
- worst neutral-chroma p95 is `7.6740`;
- worst skin/sky/foliage hue-rotation p95 is `19.8072` degrees.
- zero of six recipes pass shared-colour cross-context consistency;
- worst shared-colour median/p95/maximum drift is
  `60.8633/77.3292/80.1493` Delta E76.

The promotion decision is `rejected` for known-operator improvement rate,
median, tail, new boundary, photographic-probe new boundary and all three
context-invariance tails. No blind review is opened. This separates the
content-identification failure from the absence of geometry rewriting: a
pointwise renderer may be structurally clean yet still be the wrong
photographic look and may map the same colour differently in another photo.

The ignored report is
`outputs/reference_color_match_quality/promotion_v1/report.json`, with
content-bound report ID `2f7b8b2c...736411` and file SHA-256
`40d52826...213a1`. Its strict JSON Schema is
`configs/schemas/reference_match_promotion_report_v1.schema.json`.
Absolute paths are retained as local provenance but excluded from report
identity; reference/source/target file hashes are included instead.

### Strong batch-consistency correction

Earlier evidence established deterministic replay, source-order independence
and use of one immutable recipe. Those are necessary operational invariants,
but they do not establish album consistency. The v1 renderer recomputes Lab
normalization statistics for every source, so identical pixels can receive a
different mapping when their unrelated surroundings change.

The frozen v1 context probe embeds one exact shared chart into two images:
dark/cool surroundings and bright/warm surroundings. A fixed pointwise
operator produces byte-identical shared-region output and zero Delta E76.
The baseline fails every recipe by a large margin. Therefore documentation
must distinguish `deterministic/order-stable batch execution` from
`shared-colour context invariance`; only the latter can support a strong
batch-consistency product claim.

The main neuro-film W1 task owns the current reference-identifiability research.
This branch will consume a committed passing descriptor/head through an
adapter; it will not duplicate the W1 experiment. The standalone D-PCT task
continues to own RAW/HDR/video and portable media execution.
