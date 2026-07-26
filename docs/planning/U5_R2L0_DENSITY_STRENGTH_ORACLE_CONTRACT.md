# U5.R2L0 — Density Strength Oracle Contract

**Status:** frozen before implementation and formal report generation  
**DRPT level:** L2  
**Parent:** U5.R2E1 retained B0 cyan-shadow/warm-highlight challenger  
**Primary writer:** current Codex Goal session

## Question

Does the immutable U5.R2E1 cyan-shadow/warm-highlight operator have enough
per-image bounded-strength headroom to justify a later input-side hard policy,
or is the fixed strength `0.50` already the useful product ceiling?

This is an evaluator Oracle, not a deployable router. It may inspect the
already-rendered `0.65` candidate's per-image clipping metric and fall back to
the already-rendered `0.50` output. A deployable policy cannot use this
after-the-fact information unless a later, separately frozen method can
reproduce the decision from legal inference-time state.

## Epistemic status

The node was motivated after exploratory inspection of the completed E1
development report showed heterogeneous `0.65` clipping. It is therefore
`B0_development`, not hidden confirmation. Its thresholds are product-value
screens, not statistical population claims.

E1 remains immutable. In particular, this node does not rewrite E1's
`per_image_adjustment_allowed=false`, shortlist, strength bank, metrics, visual
decision or retained `0.50` challenger. It creates a new conditional research
child using only immutable E1 evidence.

## Immutable inputs

- exact E1 automatic report and render-pass-1 manifest, both hash-pinned;
- witness `cyan_shadow_warm_highlight_like` only;
- fixed baseline `0.50` and challenger `0.65`;
- exact 41-image A0 development set: nine gold and 32 stress;
- unchanged per-image new-hard-clipping ceiling `0.005`;
- no new pixels, image fitting, stock labels, owner votes or learned features.

## Oracle rule

For each image:

1. select `0.65` only when its already-computed new-hard-clipping fraction is
   at most `0.005`;
2. otherwise select `0.50`;
3. never interpolate strengths or pixels;
4. preserve and verify the selected E1 output hash.

The fixed `0.50` output is the comparator. Style gain is the selected output's
existing median Delta E76 from input minus the fixed comparator's value.
Non-basic residual and clipping remain separate diagnostics.

## Frozen automatic gates

- all 41 source rows and both candidate rows are present exactly once;
- all parent hashes, output hashes, sample IDs and splits verify;
- at least 50% of images select `0.65`;
- mean style gain is at least `1.0` Delta E76 overall;
- mean style gain is at least `1.0` on gold;
- median style gain is at least `1.0`;
- every selected output stays within the per-image `0.005` clipping ceiling;
- two formal reports are byte-identical.

These gates test whether an Oracle gap exists. They do not establish visual
safety, preference, deployability or a usable predictor.

## Visual veto

If the automatic gates pass, inspect every policy-selected output at full
resolution, with explicit attention to:

- ID11 red-speckle/posterization regression;
- faces, text, smooth gradients and saturated highlights;
- banding, blocks, seams, clipping plateaus and unstable colour speckles.

Any confirmed severe artifact closes the Oracle product branch regardless of
style gain. Contact sheets may aid triage but do not replace full-resolution
inspection. Results are autonomous visual B0 evidence, not owner, population or
human-panel preference.

## Branches

- **No automatic Oracle gap:** close per-image strength routing and retain fixed
  `0.50`.
- **Automatic gap but severe failure:** close; clipping is proven insufficient
  as a safety policy.
- **Automatic gap and severe-clean:** retain only Oracle feasibility and freeze
  U5.R2L1, the simplest inference-time hard policy. Start with deterministic
  analytic state; a small bounded parameter predictor is allowed only if the
  simple policy fails and a leakage-safe development/confirmation split exists.
- **Later policy cannot reproduce the Oracle:** close routing and retain fixed
  `0.50`.

## DoD

- hash-validating evaluator and targeted tests;
- two byte-identical reports and an exact selected-output manifest;
- automatic decision plus full-resolution severe review if opened;
- full CPU suite;
- tracker, board, implementation plan, AGENTS and agent-log propagation;
- scoped commits and pushes.

## Claim ceiling

B0 autonomous development evidence that a fixed film-inspired density Look
Approximation may have heterogeneous clipping-limited strength headroom on the
frozen 41-image set. No learned routing, general safety, preference,
digital-to-film identification, named-stock response, calibration,
authenticity or production claim.
