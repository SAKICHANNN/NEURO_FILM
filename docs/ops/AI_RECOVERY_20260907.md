# Early model recovery: descriptive smoke, not promotion

## Fixed ClassNeg photographic review (2026-09-07)

Parent: learned RGB pilot below. Freeze global checkpoint bf0dcb11...cdd9,
ClassNeg index1, full strength; no optimizer or parameter editing. Actual AO7
source manifest contains17 rows (not historical shorthand16); include all.
Raw SHA intersection with the prior20 transfer pool is empty. These sources
are held out from the current learned pilot, not all historical project work;
author/scene independence unknown. Existing CC0 encoded1600px inputs only.
Reuse old decoder output rather than silently changing input processing.

Config `configs/ai_classneg_photo_review_v1.json` freezes identity, learned,
current safe-rich Portra and simple1.15 contrast comparisons, unblinded
autonomous full1600px review. No false blind/population/stock claim.0 severe
and12/17 wins vs each control opens native-resolution tests only. AO6 remains
an additional required comparison; this does not replace it. Any failed/unclear
visual gate closes promotion without tuning this cohort. No data downloads.
Mode A / AI-ML primary, L2; only new config/runner/tests plus this note/log.
Existing models/defaults/foreign files untouched. Additive commits are rollback.
Status: freeze DONE; implementation/review IN_PROGRESS; full12-24MP DEFERRED.

## Learned complete RGB LUT pilot (prospective, 2026-09-07)

The fit-only oracle below establishes large representational headroom but folds;
new learner predicts a bounded9^3 RGB LUT from input and style only, not targets.
Reuse the previous small64px context CNN; replace36-param head by2187 LUT-node
residuals on top of learned style-global LUTs. Unit-cube node projection yields
bounded trilinear output, not a no-artifact guarantee. Train with L1 plus .001
second-difference smoothness and .01 squared hinge on cell-centre determinant
below .1. The soft penalty is not proof of whole-cell/global invertibility.

Same64 prior fit images. New16 development images are SHA-ranked later
source_train duplicate groups, excluding all80 prior AI paired rows and3
target-only preflight IDs. All original lockbox/test roles remain untouched.
This is fresh relative to these AI pilots, NOT necessarily all historical work
or an independent-corpus evaluation. Same explicit ICC conversion throughout.
Freeze the new manifest before any of its evaluation pixels; all three final
checkpoints before evaluation. Three from-zero arms: global, image-conditioned,
cyclic wrong-content image-conditioned (within64 fit, same style).600 Adam steps,
lr.003/batch4/192px, final checkpoint only;30min time cap and no cloud/download.
Also evaluate unchanged prior triangular conditional checkpoint on these16.

Decision: numerical progress requires >=25% median L1 reduction vs prior
triangular and >=36/48 wins, >=20% median reduction vs wrong-target. Conditional
must additionally improve >=10% over new global to be preferred; otherwise retain
global as simpler learned control. Neither numerical gate is photographic value.
Inspect fixed first3 evaluation IDs x3 styles for both real-supervision arms,
plus existing3 CC0 transfers; severe artifact veto precedes appeal judgement.
Record all cell-centre fold fractions and output boundaries. No product/weight
release or stock claim. No step/loss/grid rescue on this evaluation cohort.

Mode A; AI/ML primary. Owned new config/model/runner/test with matching
`ai_conditioned_rgb_lut` names, this note/log; reuse source decoder/LUT/evaluation
math. No legacy defaults or foreign edits. Scoped commits provide rollback;
all outputs in one new P-backed root. Status: configuration DONE;
implementation/training/evaluation DONE. A positive result
only opens independent photographic-value work, not product promotion.

New manifest metadata lock SHA256
`c65c6b6db647a637fe512508bcc4656bb8352f00f482c967ac44594bc5314326`;
no new evaluation pixels read at lock time. Model/role/regularizer unit checks
and synthetic CUDA update precede training; no new external assets/dependencies.

### Learned RGB result (not product promotion)

Code `d2ef0f023`; three from-zero600-step runs completed in105.39s total. Report
SHA256 `2c14e1f3e23015c85fffa994b7b4c2e49bb7e47d9aa965b18d5d4591e6c62436`.
All64 fit groups unchanged;16 new evaluation groups have zero overlap with the
old80 and exclude3 preflight IDs. This does not certify all-project historical
independence. Three checkpoints were saved before first new evaluation pixels.

| Arm | Mean L1,48 rows | Median gain vs prior | Wins vs prior |
| --- | ---: | ---: | ---: |
| Learned global RGB LUT | .00926825 | 59.48% | 48/48 |
| Image-conditioned RGB LUT | .00980514 | 58.22% | 48/48 |
| Wrong-content conditional | .21834097 | control | control |
| Prior conditional triangular | .02233304 | baseline | baseline |

Both numerical-progress gates pass; conditional vs global median gain -6.85%
fails preference for added conditional complexity. Keep the learned global
as the simpler supervised development candidate, not a handcrafted LUT. Global
checkpoint SHA `bf0dcb11c1b6c5f7286b056e8610f27d43d980a56741093f8c0aa3d535d0cdd9`;
conditional `c6f406d6a87327ca183f2746e81d18b47bb8cee0fc0114f01f9f1662c1420afa`.
Wrong-target gain is95.50%/95.29%; this is recipe correspondence, not film truth.

Autonomous unblinded visual review completed for fixed3 evaluation images x3
styles x2 learned arms at512px and3 previously used CC0 transfers x3 x2 at768px
maximum, with originals. ClassNeg targets for evaluation00/02 also compared.
No conspicuous geometry/text rewrite or severe false-colour breakup in this
limited review. Global/conditional are visually close; all three style hashes
are distinct for every transfer (no prior-style-collapse pattern). ClassNeg
offers clearer contrast/palette separation; Cinema is still washed/mild and
Velvia produces purple sky/yellow indoor saturation, not uniformly appealing.
These are developer observations, NOT blind preference or established film feel.

Maximum cell-centre nonpositive-Jacobian fractions remain3.90625% global and
3.515625% conditional; maximum new boundary component fractions .608317% and
.555547%. Soft penalty did not prove safe colour topology. Do not hide these
behind small L1 or promote at512/768px.16 focused/parent tests PASS, Ruff/diff
PASS; artifacts total33.54MiB, no download/cloud/production edits. Retain all
checkpoints and old negatives. Next: one frozen candidate's broader photographic
value and full-detail risk assessment, not more loss/step/LUT-size tuning on
these16. Full-resolution, independent preference, texture, product integration,
release rights and eventual12-24MP replay remain unproved. The existing private
FilmSet research permission is not a public-weight/product-release clearance.

## Fit-only operator-capacity diagnostic (2026-09-07)

Parent: paired pilot below misses conditional preference and remains visually
weak. Is a better explicit colour fit possible, or is the target unsuitable?
Mode A, AI/ML primary writer, L2 bounded local diagnostic. No live training
process observed; only desktop GPU clients. P has82.61GiB free. No new payload.

Use only first6 `paired_fit` identities in the exact existing paired manifest,
all3 styles. Do not read16 evaluation or any lockbox/test rows. Same embedded ICC
conversion as accepted paired pilot, resize192. Optimize separate per-image/style
parameters using even8x8 checkerboard blocks; score odd blocks only after final
600 steps. This within-image pixel partition is NOT independent generalization.
Two identity-initialized arms: original36-parameter triangular vs projected
unit-range9^3 RGB LUT using existing trilinear renderer; Adam .01, LUT curvature
penalty .001, no other tuning. Parameter budgets differ intentionally: this tests
representational headroom, not a matched-budget architecture advantage. Record
train/odd-block L1 by channel, identity baseline, new boundary and cell-centre
Jacobian determinant diagnostic. Bounded LUT is not guaranteed injective/safe.

Headroom signal requires >=20% median odd-block L1 reduction and >=12/18 wins
over independently optimized triangular. Otherwise no capacity-based expansion.
Even a signal only opens a separately designed learner, not this target-reading
oracle at inference. Inspect fixed first2 source IDs x3 outputs against targets;
no image/style cherry-pick. FilmSet remains internal digital-recipe supervision.

| Step | Status | Verification / commit | Rollback |
| --- | --- | --- | --- |
| Config/question | DONE | Commit before new optimization | Scoped revert |
| Runner/tests | DONE | `b4020dcad`,12 focused/parent tests PASS | New-only files |
| Capacity run/review | DONE | Report below; fixed first2 x3 x2 arms reviewed | Retain evidence, no promotion |

Owned: `configs/ai_paired_operator_capacity_v1.json`,
`scripts/audit_ai_paired_operator_capacity.py`, dedicated test, this note/log.
Reuse LUT and ICC primitives unchanged; no product/default/old evidence changes.
Run caps20min; failure leaves partial output, no auto-restart or optimizer rescue.

Completed in116.59s; report SHA256
`e97a5d4e8df0c1e035c12e6eec9d32b73ce381003c2c9662c0b1132524b83f7e`.
LUT vs triangular odd-block mean L1 .00240004 vs .01730262; wins18/18,
median/worst relative gain .850548/.750491. RGB mean errors fall from
[.025044,.010501,.016363] to [.002711,.001861,.002628]. This demonstrates
representational/optimization headroom under the fixed budgets, not that the
red-only restriction alone caused all error (parameter count also changes).
LUT cell-centre nonpositive determinants occur in17/18 fits (0..3.7109% cells),
with119 new boundary components across all images. These are risks, not a proven
severe photographic defect or full-cell safety assessment. At192px, the fixed
12 oracle outputs and6 targets show the LUT closer to target colour; no broad
film/safety claim. Existing training/evaluation roles stay unchanged. A learned
full-RGB operator needs fresh image evaluation and direct folding regularization;
do not deploy the target-reading per-image oracle.

## Paired digital-recipe AI control (prospective, 2026-09-07)

Target-only visual preflight: first3 Cinema target SHA-ranked content identities
in existing `target_train.jsonl` (`fujifilm_x_t4_11`, `20210718-DSCF9752`,
`CNRAW_25`), all three variants (9 images) hash-verified/viewed. ClassNeg has
coherent contrast/colour changes, Cinema softer; Velvia often oversaturated.
This is sufficient for a *learning control*, not proof of film appearance.
Manifest SHA `740019507339bb156a7b73061571209145447a3774f1c5319dbc4f9555c0db4f`.

New AI-product development scope, NOT Roll2Film/CT pair-blind research:
explicitly pair a bounded subset of the old source-training identities with
their distributed-train digital targets. Original role files are immutable;
new paired exposure is recorded and must not be passed off as pair-blind later.
No internal_dev_lockbox or official test628 access. Existing licence permits
internal research only; no public weights/examples or product redistribution.
These are Capture One recipes, never actual-film/stock-calibration labels.

Mode A, no production changes. New manifest builder/trainer/tests under existing
scripts/models conventions. SHA-rank source_train, take one per frozen duplicate
cluster; first64 fit, next16 same-corpus development evaluation. All3 styles.
Freeze manifest before pixels; verify file hashes and sRGB ICC before training.
Three arms: global learned triangular experts, small image/style-conditioned
CNN predicting the same36 parameters, identical CNN with cyclic wrong-content
targets (within fit pool, same style). No direct neural RGB output.
Each600 Adam steps/lr.003/batch4 at192px, seed20260907, final checkpoint only.
Freeze checkpoints before evaluation pixels. Report per-image/style L1 and
identity-relative improvement, global-vs-conditional and wrong-target controls;
inspect fixed first3 evaluation IDs and three existing CC0 transfer photographs.
Prefer conditional only if median L1 improves >=10% over global and wrong-target
on development evaluation without conspicuous severe artifacts; otherwise keep
the simpler result as a control, not a product winner. No loss/step/strength tuning
on these evaluation results. Full-resolution/independent validation remains later.

Manifest frozen at `outputs/ai_paired_recipe_pilot_v1/manifest.json`, SHA
`0e7603d551791337510850f620bdefdf66eabd44e70e249ac5e7dc7e0ba62439`.
Source/target pairs are explicitly new supervised roles; no old role file changed.
Six model tests PASS. Initial prescore test incorrectly demanded historical
scalar/batched float32 byte equality (11/648 values, max1.19209e-7); this new
kernel is a formula-equivalence comparison at2 float32 eps, not a historical
byte-replay claim. Test corrected before first paired pixel/optimizer access.
Status: manifest/implementation/training DONE, conditional preference gate FAIL. Small local
GPU run, no cloud or new dataset download. Scoped commits provide rollback;
all old model runs and unrelated worktree changes stay intact.

Pre-training source correction: attempt `run/` stopped on first input header
before any decoded array/optimizer/evaluation pixels; all256 fit files carry
embedded `Adobe RGB (1998)`, not the assumed sRGB. Empty attempt directory
retained. V2 uses explicit LittleCMS relative-colorimetric conversion from the
embedded profile to sRGB8 for both input/target, never assigns sRGB to unlabelled
pixels. Conversion is an explicit gamut/quantization limitation, not sensor
truth. Source pair hash-lock, roles, model, steps and loss unchanged. New result
root `run_icc_v2`; RGB profile validation precedes decode. Original target-only
preflight is not a numerical colour-managed comparison.

### Paired pilot closure and transfer review

Accepted training code `814b387e3`, all three final checkpoints frozen before
the16 development-evaluation targets. Report SHA256
`84b620a5bdaa5b675c2a2f4efecf4535000fc450c52f0e3270bf3e16bf6be9e8`.
Across48 image/style rows, mean L1 is global .023265, conditional .021924,
wrong-target .069723, identity .030595. Conditional wins39/48 against global,
but median relative gain .0701676 misses the frozen .10 rule. Per-style medians
are Cinema .113726, ClassNeg .070168, Velvia .042001; no passing-stratum rescue.
Against identity median gain .304498, worst -.180226. Paired supervision is
informative relative to mismatched targets, but this does not establish film
appearance, independent generalization, or a conditional product winner.

Fixed first3 evaluation images x3 conditional styles reviewed at512px, plus all
9 conditional transfers on existing CC0 rows6/7/8 at maximum768px. No conspicuous
severe false-color/geometry failure in this narrow review; outdoor styles remain
similar and film appearance insufficient. This is autonomous development review,
not full-size safety or blind population preference. Transfer report SHA256
`4f18ae7fa0e691e37b2527df4d5498284b4ef4e6546dcc179bec657cc2d75f75`.
Global/wrong-target transfer PNGs are retained controls, not all visually
adjudicated. The first oversized six-image tool response was truncated and not
counted; Cinema/Velvia images were subsequently reviewed in batches of3.

Stop this fixed pilot without changing steps/loss/gates on its evaluation rows.
Keep global as a learning control, not a product selection. Before another
learner, inspect fit-only residual representability: this triangular operator's
red output for fixed parameters depends only on red, unlike a general colour
transform. This is a structural restriction, not a demonstrated cause of failure.
Any follow-up must separate supervision value from operator capacity, use new
evaluation identities, and avoid treating digital presets as film truth.
Eight model/source tests PASS; no production integration, new downloads or
independent-test reads. Existing project structure and foreign diffs preserved.

## Texture supervision pilot (prospective, 2026-09-07)

Parent: learned colour pilots below did not establish film appeal. New question:
can the already-cleared 19 Italy JPEG references supply transferable *display
texture* supervision, rather than another scene-palette objective? This is not
emulsion/grain calibration or a reopening of the historical Commons grain pool.
No new downloads or independent-test access. Native JPEG scale only.

Mode A, AI/ML primary. Allowed: new `src/eval/photo_texture_reference.py`,
`scripts/audit_ai_photo_texture_reference.py`, config and focused tests; existing
recovery/log notes. Production, old runs and foreign diffs remain untouched.
Rollback is scoped reversion; generated diagnostics remain negative evidence.

1. DONE: hash-lock all 19; sort SHA, first12 development-fit and remaining7
   same-author diagnostic images (not independent population confirmation).
   Extract up to8 disjoint64px patches per image on a64px grid. Require less
   than1% pixels with any channel outside [.03,.97], smoothed-luma gradient
   p90 <=.01; rank by this gradient with coordinate tie break. Remove least-square
   RGB planes. Record residual standard deviation, 2-D spectra, lag1 correlation,
   JPEG8 boundary/interior residual-difference ratio and within-image dispersion.
   Inspect source patches and amplified residuals, not only a noise score.
2. CLOSED_NO_FIT: only if >=12 images have >=4 eligible patches and visual
   support is not dominated by edges/blocking, freeze a small learned texture
   model and white-noise control. Otherwise stop extraction-as-supervision;
   do not raise thresholds, fit contaminated patches or call them film grain.
3. NOT_OPENED: any learned model must compare source-image-disjoint residual
   statistics and actual photos against colour-only and equal-power white-noise
   controls before independent/product work. No additional manual strength.

Noise-estimation methodological reference: Liu/Tanaka/Okutomi ICIP2012
https://doi.org/10.1109/icip.2012.6466947 explains weak-texture selection and
scene-contamination risk. Our transparent plane-residual diagnostic is NOT
their PCA/Gaussian-noise estimator and cannot identify true noise from texture.
Support is an engineering gate, not proof of a film-domain distribution.

Result at `b98faca44`: 121 patches, 15/19 images have at least4, numerical
support passes. Report `outputs/ai_photo_texture_reference_v1/report.json` SHA
`bbe4caeb9c505afe2558e03e7a058753d26fa15368a57f776d23cd7f80bee69a`.
All four patch sheets inspected. Several fit images contain mostly sky-like
noise, but06/10/23/07 retain strong scene structure. Diagnostic04 has masonry,
25 has obvious texture/blocking,16/05 include strong surface/edge residuals,
26 is visibly block-compressed with boundary/interior ratio3.28-4.57. Thus the
fixed extraction is not reliable unified texture supervision across this set.
Do not manually prune patches, retune selector or train its contamination into
grain. No texture model trained; no product pixels changed. Four synthetic
extraction tests PASS (plane rejection, white-noise statistics, disjoint selection,
invalid input). This rejects this JPEG residual-supervision pilot, not learned
texture generally. The original scan appearance is still a style reference.

Next source check: existing FilmSet target-train examples only, internal research
under the existing licence boundary. Do not inspect official test628 or recover
paired targets through the pair-blind loader. FilmSet is independently authored
digital Capture One recipe supervision, not physical film. First inspect target
appeal; do not launch training merely because pairs exist. Any paired experiment
requires a separate explicit role freeze and cannot be presented as unpaired.

## Single-reference diagnostic result, 2026-09-07

Run from `aac6e4244`, all three predetermined reference arms completed.
Reports under `outputs/ai_single_reference_photo_pilot_v1`:

- ref_01: `c49c46ff988774f884910af780e860a640d2b5b164f4df4ec978e3ee9037c640`
- ref_07: `d07dfbf88141597b187e669ab862e22add50a3ffb024464647685d5adb712a8e`
- ref_13: `4b3b761ce369c46d36329d7f295e0d13e53a123bc7990205984588cc71fef943`

All nine triangular transfer outputs, nine corresponding simple controls,
three originals and three averaged-reference triangular outputs were inspected
at their 768px development resolution. The first attempted bulk image display
was truncated and not counted; subsequent bounded displays covered the set.
Ref01 is cooler/greener, ref07 warmer with more contrast, ref13 mildly pinker.
The triangle avoids the conspicuous false-colour failure of the unconstrained
cube on these views, but remains a modest colour adjustment. The simple arms
show similar reference-dependent colour changes (ref13 also lifts shadows).
No clear credible-film/appeal advantage is established; do not select a winning
reference from these nine pictures or promote to product. Averaging alone is
not sufficient to explain the weak outcome under this fixed operator/budget.

This is unblinded autonomous development inspection, not independent evidence,
population preference or a full-resolution severe-artifact pass. The source07
sideways orientation is already in the original, not introduced by learning.
Five structural/source tests pass. No model or product parameters changed.
Next substantive question is the supervision's ability to distinguish film
appearance from scene palette/illumination, not more reference/strength search.
Retain the learned models and outputs as negative controls.

## Reference-average diagnostic, frozen before execution

Same structural operator/features/120step budget/source roles; vary only target
from all19 averaged feature moments to one reference at a time. First3 source
SHA-ranked images are03/09/18 (manifest indices1/7/13), chosen without output
scores. `ai_single_reference_photo_pilot_v1.json` fixes these three and requires
all9 transfer outputs reviewed. Same simple arm reruns for each target; no
best-reference routing or postscore palette choice, no promotion from this
small development diagnostic. If distinct but unappealing, averaging alone is
not a sufficient explanation. Existing averaged results remain immutable.

## Structural successor, before photo training

Executed development successor at `bf2f7c3d5`,120 updates/arm, same feature
targets and source roles. Report SHA256
`33c2eb8a328c12d57fb22389efd01eced82411f1e6fe863628cce5e324431ee9`
under `outputs/ai_structural_photo_pilot_v1`. All3 triangular transfer outputs
visually inspected: no obvious preceding broad false-colour/contour corruption
at768px, but effect too weak to establish credible film appearance. NO PROMOTION;
full-resolution/independent gates remain unopened. This proves neither universal
artifact safety nor inability of all learned operators; one fixed global feature-
average fitting run is insufficient. Do not increase strength after inspection.

Structural tests3 PASS (identity/endpoints,12 extreme-parameter Jacobians,
nonzero gradients/invalid values). Shared training harness default path preserved;
new structural option changes only candidate operator and omits a lattice-only
regularizer. Initial lint flagged an immediate-use loop closure and import
spacing; shell still ran. Post-run correction explicitly binds callback defaults
and formats import; no numerical-rule/parameter changes, current Ruff PASS.
Future launch commands must check exit codes before commit/run, not chain through
a failing validation. This operational correction does not relabel old run lint.

Add private `src/models/color_lut/triangular_photo.py`, with zero-initialized
36 learned parameters: each output logit is its input logit plus a bounded bias,
nine bounded tanh basis functions of that same channel, and lower-triangular
cross-channel terms using preceding original RGB channels. No spatial resampling.
For each own-channel logit derivative, sum of absolute basis derivative bounds
is <=.5, hence derivative lies [.5,1.5]. Triangular Jacobian has positive diagonal
on the open RGB cube, so determinant is positive. Exact endpoints are handled
analytically, not by clipping all near-endpoint inputs. This is a new explicitly
specified structural response to the measured folding; not rescue of prior LUT
weights, and not proof of photographic quality or novelty. Unit tests must cover
identity, endpoints, extreme parameters, positive Jacobians and gradients before
any photographic fitting. Production API and historical runner remain unchanged.

## Deep-photo pilot v1 (frozen before training)

Result: completed at implementation `94e82a8ae`,120 fresh updates per arm.
Report `outputs/ai_deep_photo_pilot_v1/report.json` SHA256
`9fe6677103fbd745d37736b55963ea351c87fccb67993f601d1eaf6d59fdb266`.
All3 transfer LUT outputs inspected at768px: strong false-colour regions and
posterized/contoured building, leaf and wall textures. Severe-artifact veto FAIL;
simple row06 remains weak, remaining simple rows not needed to reject LUT.
Do not promote, reduce strength, tune this run or call it an NLUT paper failure.

Read-only postscore finite-difference diagnostic (not a preregistered gate):
33^3 RGB probes on [.01,.99], forward h=1e-4 using saved parameters and unchanged
executor:4138/35937 negative Jacobian determinants, min-4.89016,max17.37087.
Bounded output alone demonstrably does not prevent local orientation reversals.
This supports a representation-risk diagnosis, not complete causal attribution
of every visual artifact. Next bounded-learning hypothesis should structurally
exclude folding/large derivatives, not merely change feature scorer or tune
smoothness after these images. Weights and all negative outputs retained.

Reuse the verified feature model, all19 development references and source
rows0..5 / transfer6..8; no independent data consumed. New
`ai_deep_photo_pilot_v1.json` fixes120steps/arm,192px,Adam.01,9cube residual
LUT versus learned simple logit-affine. Four-layer reference channel means and
population standard deviations are averaged equally across images. Each layer
style error is normalized by its frozen reference squared moment energy+1e-6;
content relu4_1 error by input feature squared energy+1e-6. Weight style1,
content.1,parameter-neighbour smoothness.02. This normalization is our explicit
development variant, not exact NLUT reproduction. Encoder frozen; only operator
parameters optimized. All images rendered at768px for initial review; no texture
generation or hand palette. Stop fixed run if no convincing visual advantage;
do not use success of numerical optimization as the product gate.

## Next supervision decision: deep photo features, not text or RGB marginals

Preparation executed successfully at `a30b26cc0`. Exact LICENSE/net/weight Git
blobs match; 80,108,872 bytes total under `data/ai_models/nlut_feature_v1`.
Manifest SHA256 `d817f7ec074dbea827781188bdb2e5015cf137d5cd0e56ebb327341e57088142`.
Strict weights-only state loading succeeds without unsafe pickle fallback.
Frozen four-layer features for synthetic 64x64 input are 64x64x64,
128x32x32, 256x16x16, 512x8x8; finite nonzero input gradients and no trainable
encoder parameters. No photograph read; no learned film output yet. Ruff and
compile pass. Model/source available locally for the next bounded learning run.
Official finetuning defaults content/style weights1/1,40iterations and lr1e-4
apply to its pretrained network, not automatically to our zero-initialized LUT.
Freeze a separately identified development experiment rather than claim those
defaults or full NLUT reproduction.

Bounded model preparation now frozen in `ai_nlut_feature_source_v1.json`:
80.1MB published NLUT feature model + exact source + license, Git blob verified
before weights-only loading, no unrestricted pickle fallback. Oxford primary
VGG page explicitly releases original models under Creative Commons Attribution;
NLUT README covers algorithm with MIT and publishes normalized weights. This
supports private comparative development with attribution, not a claim that the
normalized conversion was independently reproduced or cleared for product
redistribution. No decoder/full RGB generator, full NLUT checkpoint or dataset
is downloaded. A finite synthetic forward smoke opens experiment implementation,
not photographic efficacy. Existing source code is loaded only after exact blob
verification and review; freeze/verify artifact signatures in a local manifest.

2026-09-07 read-only official implementation audit identifies NLUT
<https://github.com/semchan/NLUT> revision
`66e271dd5740282d1b0f16d42ed5ca3f763ce818` as a concrete comparison method.
Exact `net.py` blob `e2872c99aa56b33ab973031acf29d78bfef21f2c` uses fixed VGG
relu1_1/relu2_1/relu3_1/relu4_1 features. Style loss is summed feature mean and
population-standard-deviation MSE; content loss is relu4_1 activation MSE.
The third moment is computed but commented out of the loss. It is not CLIP
text supervision, RGB histogram matching, or a full Gram-matrix loss.

Official repository README calls the algorithm MIT and its LICENSE contains
the MIT permission text, with a surprising inherited Friendika copyright.
The tree includes `models/vgg_normalised.pth`, 80,102,481 bytes, Git blob
`18a783877329319e5a640ae20f7b98c64e5f5572`; checkpoint redistribution/training
provenance must be kept distinct from code permission. No weight downloaded
or executed in this audit. Full pretrained NLUT is externally hosted and has
not been audited or obtained. Do not claim exact NLUT reproduction yet.

Available user torch cache was inspected read-only: ConvNeXt-tiny, DINOv2-small,
MobileNetV2, RAFT-small and an unidentified scaled_offline checkpoint, no VGG.
Do not silently substitute these for the named official feature extractor or
load the unidentified checkpoint. Existing train_unet.py uses torchvision VGG19
features, but presence of code does not prove a local matching checkpoint.

Next ready action is bounded VGG provenance/acquisition verification followed
by a fixed deep-feature-supervised LUT comparison on the licensed development
references, with no production imports. A pointwise LUT still cannot synthesize
grain or halation; feature supervision is only an appearance-learning hypothesis.
The previous RGB pilot does not prove all global LUTs incapable of film colour.
Any later learned texture component needs its own identifiable objective and
content/artifact tests, not arbitrary hand-added grain to declare success.

## Actual-photo supervision intake, 2026-09-07

### Photo-supervised pilot result

Executed committed implementation `867988a13`: 150 fresh Adam updates per arm,
no checkpoint reuse, output `outputs/ai_photo_distribution_pilot_v1`.
Report SHA256 `1e4cd6d8e9364260e051ab867ca7a73db01cef8ace4e61c9d66e409f805a52ee`.
Both arms started distribution loss .00714744. At step100 LUT distribution
loss .00063522 vs simple .00120565; this proves fitting, not visual superiority.
All three transfer rows' originals/LUT/simple were visually inspected at768px.
No obvious prior CLIP-style green/magenta contour corruption at this resolution.
However LUT mainly lightens/desaturates the building, shifts the foliage/wall,
and cools indoor light. It does not establish convincing film character or a
clear photographic advantage over the simple learned control. NO PROMOTION;
independent/full-resolution validation not opened. This exact fixed run stops,
without strength/loss/palette rescue. Missing texture/context in a pooled RGB
distribution is a plausible limitation, not a demonstrated causal conclusion.
Next method must learn more than global colour-distribution matching.

Implementation lint initially flagged an unused noqa, but the shell proceeded
to commit/run. Removed only that comment afterwards; current Ruff/compile pass.
The execution commit is preserved and no claim of pre-run lint pass is made.

Photo-distribution pilot frozen before fitting: 19 references inspected as a
contact sheet, natural travel scan appearances, repeated bridges/architecture
and lighting confounds explicitly retained as a single development group.
QA dHash minimum within=16, against historical stored signatures=20; these are
heuristics, not proof of independence, and historical pixels remain unread.
New `ai_photo_distribution_pilot_v1.json` fixes 150-step equal-image sampled
RGB sliced-quantile distribution learning, 64 seeded projections, versus a
learned simple logit-affine control and identity. Shared 9-cube residual LUT
and simple arm use the prior explicit bounded executor, no CLIP objective,
no hand palette. This tests actual-photo supervision, not methodological novelty.
Spatial content cannot be invented by this pointwise executor, but false colour
and contour artifacts remain possible and require visual veto. Frozen source
development rows0..5 and transfer rows6..8 are reused development, never a final
independent result. Run once, preserve parameters/losses/images and do not rescue
the fixed run by changing strengths, palette, losses or references afterwards.

AI-first child after the failed CLIP-only pilot: acquire the 19 explicitly
CC-BY-4.0 photographs on Nick Rudzicz's Italy 2024 page, with a 64MiB total cap.
The page states Nikon FM-2 / Portra400. This is author/scan appearance supervision,
not identified Portra response. All 19 are development references in one group;
no random within-series split is an independent test. Old A3U 2023 references
and sealed assessment remain untouched. Exact new URL search of configs,
docs/data and docs/evidence found no matches before this lock.

Sequence: freeze config -> bounded source/rights snapshot + image hashes ->
check duplicates and view references -> separately specify photo-supervised
learning and controls before fitting. Abort on rights/URL drift, decode failure
or budget excess; retain explicit partial state, never substitute another image.
No training or production change is authorized by acquisition success alone.
Candidate learning must beat identity/simple controls visually after artifact
review; photo-distribution loss is not a photographic-quality result.

Owner: main single-agent. New acquisition script/config and this recovery note
only, plus minimal agent log; foreign dirty files are excluded. Revert the scoped
code commit for rollback; preserve source manifests and historical evidence.
Data layout: source.html, manifest.json and ID.jpg under the configured P-backed
repo-relative directory; manifest carries bytes/SHA256/dimensions/URL/role and
license attribution. Verify bounded reads, create-only destination and JPEG
integrity before recording completion. Intake is DONE; learning and
independent validation are NOT_STARTED.

Result: 19 JPEGs / 11,797,708 bytes. Manifest SHA256:
`7a7cf18824151f656cdd94e82abcdd4e4186c2167f273dbd32d27bd61da8ff06`.
JPEG verification and historical nicknick URL/byte duplicate checks pass.
Perceptual overlap remains unchecked, sealed pixels unread. Image04 visually
inspected only: natural building/sky scene with visible scan texture; not full
source QA or learned-candidate evidence. Parser/hash tests 2 PASS; Ruff/compile
PASS. Initial inline shell smoke had a quoting SyntaxError before execution;
replaced with pytest tests. No training or production change.

The current user asks for AI-learned film character without supplied paired
captures. Manual creative-v6 changes remain interrupted and untouched.

## Live evidence

- `scripts/train_neural_lut.py` constructs supervised targets by calling
  `pipeline_color_baseline.style_transfer`; checkpoint
  `outputs/neural_lut/challenge_color6_s800_b12_init/model.pt` exists. Its
  800-step training loss is teacher approximation, not film authenticity.
- `outputs/neural_film_lut_v2/scheme_a_distilled_v1/metrics.json` records
  20 images and teacher `outputs/eval/color_engine_challenge/film_response_v1_s1p0`.
  `fit_distilled_film_lut.py` fits curves/residual LUT and context gain from that
  teacher. This is data fitting, but not independently learned real-film style.
- The numpy model SHA256 is
  `1be4dce9b99bb9565997e22b6265c87fdbfe11f85ac1d6155f7359f45e2059a0`.

## Executed recovery

Unchanged `scripts/evaluate_distilled_film_lut.py`, existing `.venv`, first three
historical manifest rows, Ektar/Portra/Velvia configuration IDs, strength 1.0,
max-side 768, historical output margin 4. Output directory was checked absent
before execution: `outputs/ai_recovery_20260907` (47 files, 20,561,301 bytes).
No training, new downloads, assessment-set reads, model changes or product changes.
Summary SHA256 `36b3bce76a6ac144c58ed4ba863b5201b385e25c4a45987a26585595f63ab809`.

Ektar and Velvia contact sheets visually inspected: snow scene, fruit/chart and
flowers. Changes mainly appear as contrast/chroma differences. This inspection
does not establish compelling film character, full-size safety or user appeal.
Portra executed but was not separately visually adjudicated. Same historical
training-era images are intentionally reused for recovery, never held-out claims.
Legacy clipping margin/SSIM are reported by the old runner, not current promotion
criteria. This run does not establish which early algorithm was the user's best.

## Next useful action

### First new semantic-supervision learning pilot: rejected

Frozen implementation/config commit `d09b953ce`, script
`scripts/run_ai_clip_lut_pilot.py`. Official local CLIP ViT-B/16 SHA256 matches
the public model URL `5806e77c...df416f`; MIT software license and research
limitations read from https://github.com/openai/CLIP. No downloads or paid run.
Model is frozen and backpropagates image-text loss into a shared17-cube field;
the competing simple control learns six diagonal-color parameters. Final RGB is
an explicit logit displacement, not a generated image. Both60 steps, seed fixed.
Inputs are nine existing CC0 rawpixls derivatives: six training, three transfer
development rows. These are not new independent assessment or real-film targets.

Identity, finite-bound and gradient smoke passed before learning. Two local CUDA
fits and27 renders completed;30 artifacts /18,159,362 bytes. Report SHA256
`c4ae5e50ed58d33ab7e75506eeeca764aedaae34ca1b0df7bfccc3f2cc385292`
under `outputs/ai_clip_lut_pilot_v1`. Model/evaluator unchanged; no prompt sweep.

Visual inspection of all three LUT transfer results rejects the candidate:
station facade, wall/shadows and interior show conspicuous false-color speckle,
contours and patches. Original/station simple-control inspected too; simple
control has a pervasive purple/red cast. Falling CLIP loss does not establish
film appeal. Bounded output did not prevent severe color artifacts. No promoted
candidate, no full-resolution work, no parameter/prompt/step rescue. Remaining
simple transfer rows need not be adjudicated to reject the LUT candidate.

This is a concrete learned baseline failure, not evidence that all AI methods
fail. Next requires a photographic style observation or stronger justified
supervision and structure, not more text-score maximization. Official CLIP model
card explicitly does not establish general deployment readiness.

### Current method/source triage (2026-09-07, no new payload)

| Primary source | Decision for next learner |
| --- | --- |
| https://github.com/ns144/3D-LUT | Direct precedent: CNN-predicted LUT with unpaired GAN/CycleGAN/StarGAN. GitHub API reports no license; not adopted as code/data/weights. Architecture alone is not our novelty. |
| https://github.com/EtonMu/deep-analog | Latest 2026 reference-LUT work, but existing registry records no usable permission/checkpoints; procedural supervision is not physical-film evidence. No adoption. |
| https://github.com/Ry3nG/SA-LUT | Do not reopen R2Q0/BL12: existing source/runtime and resource failures remain authoritative. |
| https://openaccess.thecvf.com/content/WACV2025/papers/Li_D-LUT_Photorealistic_Style_Transfer_via_Diffusion_Process_WACV_2025_paper.pdf | Score learning with explicit final LUT is relevant, but R2R1 published-trajectory orientation/range failures remain closed. Not a ready checkpoint. |
| https://github.com/jonathangranskog/lut_generation | MIT engineering precedent for text-guided LUT optimization, not established film supervision or a paper-quality result. Separate base-model rights and reward-hacking risks; not selected merely because runnable. |

Exa discovery followed by official repository reads and local closed-route
deduplication. No image/model/data downloads or external messages. This is a
decision matrix, not an exhaustive literature survey. A new learner needs legal
style observations or a justified learned supervision signal, not another
hand-authored teacher. Generic look claims do not authorize reopening sealed
physical-stock evaluations. No candidate promoted or new training yet.

### Three-checkpoint diagnostic closure

`scripts/audit_early_lut_conditioning.py` is a read-only CPU probe on one fixed
64x64 synthetic RGB ramp. No training or photographs are used by the diagnostic.
Results: smoke style-LUT span .0090916; s300 and s800 span exactly0. Smoke logits
range -1.65..2.84; s300 -20.21..23.24; s800 -73.12..154.51. Softmax saturation is
observed; causal attribution to learning rate, target diversity or preprocessing
is not established by comparing distinct runs.

The training script minimizes only L1 against safe-rich targets, reports loss on
the same examples and has no condition-effect validation. Training uses padded
128px squares while the old evaluation consumes unpadded 1600px images: another
known mismatch to avoid in new experiments, not a proven sole collapse cause.

Unchanged smoke checkpoint inference also completed on the same three inputs.
Its metrics record only Portra/Velvia, six examples and60 steps. Ektar was
additionally executed but is an **untrained ID**, excluded from quality claims.
Velvia sheet inspected: still visually near-input. Summary SHA256
`fc58c54d5b01fd67bf36a638f98dc69d0f87f824d9abc5b0adcd02546ed404c2`, under
`outputs/ai_recovery_20260907/neural_smoke`. No safety/full-size promotion.

This closes the bounded old-LUT recovery pass: do not expand old-weight tests or
repair pseudo-teacher fitting as the new film-learning method. Next audit legal
non-paired style supervision and official learned operator implementations;
retain these recovered outputs as bland learned baselines.

### Neural s800 recovery and conditioning failure

Executed unchanged `evaluate_neural_lut.py` with
`outputs/neural_lut/challenge_color6_s800_b12_init/model.pt`, first same three
historical inputs at their 1600px source size, Ektar/Portra/Velvia IDs and legacy
output margin4. CUDA inference completed nine outputs; 28 artifacts / 31,297,190
bytes under `outputs/ai_recovery_20260907/neural_s800`. This differs in resolution
from the preceding 768px numpy recovery; do not compare their metrics directly.

Checkpoint SHA256:
`83c21d2121ab9bb2a3e99ffa49653a947ae16c1ccddb0222a7a152f098919899`.
Summary SHA256:
`7a9e69c9a71605fe7c7de9c112ca8c40adefafe22a5afb360f40b96f707d0dd9`.
For each input, all three style output PNG hashes are identical:

- 01: `77b815776025e5311df93c42a182661749e36d1c404f97a6e3d21fc786c40c83`
- 02: `703dad793922e41b7476f0b52bf464b9d8704280b68a44b54e0aed8e95c64366`
- 03: `a47ad16647b119fdfab1bdfffb2b00edb22d72a534f628b719bab537c47cedfd`

Read-only CPU probe on input01 through the exact loaded encoder: every one of
the eight checkpoint style IDs returns one-hot weight at zero-based basis7
of12, all other weights zero; maximum cross-style LUT difference is exactly0.
Thus this checkpoint exhibits collapsed conditioning on this probe. This is not
proof of the training cause, nor a universal statement about all images/models.
Ektar sheet inspected: very small visual change. SSIM .99902-.99968 is not a
style-success measure. No promotion, retraining or checkpoint modification.

The documented owner anchors 01/09/53/55/56 are recovered in
`render_filmcase_anchor_set.py` as deterministic pipeline recipes, explicitly
anchor-inspired rather than exact historical replay. They do not identify a
preferred AI checkpoint. Keep that distinction instead of retroactively labeling
those preferences as neural-model success.

Recover neural checkpoint inference and trace the genuinely preferred early
outputs separately. Compare on the same development photographs before deciding
what is reusable. Keep diffusion-era output evidence, but do not restart final-RGB
generation. A new learned route must change the supervision, not merely distill
another hand-designed palette. Independent assessment remains sealed.
