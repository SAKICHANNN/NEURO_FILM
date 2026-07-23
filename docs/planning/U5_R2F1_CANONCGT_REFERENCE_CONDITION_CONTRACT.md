# U5.R2F1 CanonCGT reference-condition safety contract

Date frozen: 2026-07-23

Node: `ULT > U5 > U5.R2 > U5.R2F1`

Status: **frozen before any nine-reference frontier inference**

## Research question

Can the fixed public CanonCGT E2E checkpoint use a small immutable bank of
unpaired, rights-audited A0 photographic-film images as reference conditions
to produce visibly stylized, non-basic and reference-sensitive explicit-LUT
renders without a confirmed severe artifact on the existing nine gold
inputs?

This tests one plausible non-generative ML route. It does not test stock
authenticity or identify a digital-to-film operator.

## Parent evidence

- U5.R2D/D1/D2 show why raw statistics and imperfect canonicalizers cannot
  identify an unpaired operator.
- U5.R2E1 retains one strong deterministic B0 density challenger, which is a
  comparator rather than teacher truth.
- U5.R2F0 verifies the Apache-2.0 CanonCGT source at commit
  `229a7d3ec24af19ea5d97bb136f0317762bb9261` and the 20,914,369-byte E2E
  checkpoint at SHA-256
  `916f7ad5028d3fef51cf915bb51febc9508e7a378c0babe9ff3d3992b8c594f7`.
- The checkpoint predicts two 17-cube LUTs. It is runnable and strong on the
  official sample, but the predicted LUTs are not bounded and the public SSL
  file is only a placeholder. F1 therefore uses E2E only.

## Immutable reference conditions

The exact nine entries in
`configs/u5_r2f1_canoncgt_reference_condition_v1.json` are three independent
groups from each of three already-acquired source/category pools. Every local
hash matches its acquisition manifest and every entry has a retained CC BY
rights record.

The category strings are provenance only. They are not accepted stock truth.
The current Commons/YFCC pools failed stock identifiability and remain closed
for training and operator fitting. F1 uses each image once as a runtime
reference condition; it never uses a reference as a target, pair, loss,
teacher, fitted parameter source or named-stock expert.

## Fixed execution

- Load only the fixed E2E checkpoint with `weights_only=True`.
- Do not train, fine-tune, optimize, retrieve presets or update batch
  statistics.
- Use the nine frozen R2B gold inputs and nine references for exactly 81
  input/reference outputs per pass.
- Decode to float32 encoded sRGB in `[0,1]`.
- Obtain the canonicalizer and restyler 17-cube LUTs from the frozen model.
- Verify that applying those two LUTs explicitly reproduces the model's
  low-resolution output within `1e-6` maximum absolute error.
- Produce final full-resolution RGB only by applying the two predicted LUTs;
  no neural decoder or spatial RGB generator is allowed.
- Clip only once for RGB8 serialization while recording every pre-clamp
  range and out-of-range fraction.
- Repeat the complete 81-output pass and require byte-identical manifests,
  LUT hashes and output hashes.

The external checkout stays isolated under ignored outputs. Production
modules may not import it.

## Required audit

For every input/reference pair record:

- source, reference, model/config/checkpoint and output hashes;
- both LUT hashes, minima, maxima and out-of-range node fractions;
- canonical image and raw final minima, maxima and out-of-range fractions;
- official-vs-explicit LUT replay error;
- dimensions, device, seed and software commit.

Any non-finite value, hash mismatch, missing parameter, unexpected forward
parameter, replay error above `1e-6`, output resize or repeat mismatch
invalidates the run before visual interpretation.

## Automatic gates

Reuse the R2B definitions:

- gold median style Delta E76 from input `>= 7.0`;
- gold median non-basic residual after the frozen joint
  EV/WB/contrast/saturation affine fit `>= 4.9`;
- worst-gold new hard clipping `<= 0.5%`;
- worst-gold raw-final out-of-range fraction `<= 0.5%`.

The complete reference bank must also achieve median pairwise output Delta
E76 of at least `2.0` across matched inputs. Otherwise conditioning is too
weak to justify a reference bank even if one global-looking result passes.

LUT-node excursions are always reported but are not themselves a promotion
gate: only the composed raw final and serialized result determine clipping
risk. Large excursions remain explicit product-risk evidence.

## Frozen shortlist and visual veto

Among automatic survivors, retain at most one reference from each provenance
bucket, ordered by non-basic residual, style and reference ID. Admit at most
three candidates.

Build three deterministic blind rounds containing input, safe-rich,
margin-4 anchor56, the retained U5.R2E1 density challenger and the shortlisted
ML candidates. If sheet width requires separate matched panels, keep the
same randomization seed and private mapping.

Comparator paths are fixed by the byte-identical R2B and R2E1 manifests
recorded in the frozen configuration; they remain comparators, never targets.

Inspect every shortlisted ML candidate at full resolution on all nine gold
images. Sample 11 remains the explicit red-speckle/posterization regression.
Also inspect faces, smooth gradients, saturated reds/blues, text, fine
texture, highlight clipping, colour islands and output-wide casts. One
confirmed severe failure vetoes that reference policy.

Autonomous review is B0 development evidence, not an owner vote,
independent-human result or population preference.

## Branches

- **Integrity or replay failure:** reject the adapter/run and fix only the
  implementation defect; do not interpret images.
- **Reference diversity below threshold:** close the bank as effectively
  global/basic.
- **No automatic survivor:** retain CanonCGT as an external negative control.
- **Survivors with severe failures:** reject those policies; never weaken the
  veto.
- **Safe but basic/bland:** retain as a control, not a challenger.
- **Safe, non-basic and visually useful:** retain at most one B0 generic
  reference-conditioned ML challenger.
- **LUT or raw-range risk incompatible with a bounded product:** retain only
  offline research evidence even if images look attractive.

No F1 outcome establishes stock response, calibration, latent modes or
production eligibility. Failure does not stop Ultimate.

## Allowed actions

- write an isolated adapter, runner, focused tests and ignored evidence;
- use the existing local GPU without paid compute;
- perform two bounded 81-output passes and autonomous visual review;
- commit and push scoped contract, implementation and result evidence.

## Forbidden actions

- train or fine-tune CanonCGT or any other network;
- acquire the paper's FiveK/Lightroom preset training targets;
- call these nine images paired data, targets or stock truth;
- infer exposure, EI, illuminant, push/pull, process or scanner labels;
- modify thresholds, references or shortlist after inference;
- integrate external code/checkpoints into production or redistribute them;
- use a neural spatial decoder to generate final RGB;
- fit current real-film/community pixels, open LSM, release or deploy.

## Definition of ready

- exact source/model/config/reference hashes present and verified;
- nine independent reference groups and nine frozen gold inputs available;
- no conflicting writer or dirty file overlaps;
- contract and configuration committed and pushed before inference.

## Definition of done

F1 closes only with two exact passes, complete LUT/range/replay evidence,
automatic metrics and reference-sensitivity result, deterministic shortlist,
three blind rounds when applicable, all-nine full-resolution veto notes,
focused and complete tests, a branch decision, claim-ceiling propagation,
scoped commit and push.
