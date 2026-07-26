# U5.R2J1 positive-film real-image frontier contract

Date: 2026-07-26

Node: `ULT > U5 > U5.R2 > U5.R2J1`

Status: **frozen before rendering or inspecting results**

## Question

Does any fixed J0 positive-film witness produce a visibly strong, non-basic and
artifact-safe photographic look that is preferable to both safe-rich and the
retained R2E1 density challenger?

## Frozen evidence and candidates

- immutable provisional set: 9 gold and 32 stress images;
- five exact J0 witness parameter sets;
- strengths: `0.10`, `0.15`, `0.20`, `0.25`, `0.35`;
- 25 candidates, with no result-dependent interpolation, extension or retune;
- no effects, per-image adjustment, resampling, fitting or learned output.

The lower strength range is frozen because J0's uniform-grid identity RMSE is
already `0.383–0.416` at full strength. It is not evidence that any selected
strength will pass.

## Automatic gates

Inherit R2B without modification:

- gold median style Delta E76 at least `7.0`;
- gold median residual after frozen joint EV/WB/contrast/saturation fit at
  least `4.9`;
- worst-gold new hard clipping at most `0.5%`;
- report worst-stress clipping separately.

The shortlist keeps at most one strength per witness, selects the maximum-style
surviving strength for that witness with lower-strength tie-break, then ranks
at most three representatives by non-basic residual and style.

## Visual gate

Only automatic survivors may enter:

- three deterministic blind rounds;
- all nine gold images visible in every round;
- columns include input, `bland_safe_rich_control`, retained
  `cyan_shadow_warm_highlight_like__s50` from R2E1, and the J1 shortlist;
- mapping remains private until all round choices are recorded;
- every shortlisted J1 render is then inspected at full resolution on all nine
  gold images;
- ID11 red-speckle/posterization regression is explicit;
- any confirmed severe artifact rejects that candidate regardless of style.

A J1 candidate can be retained as a B0 challenger only if it has no confirmed
severe gold artifact and demonstrates coherent visual value against both
comparators. This is autonomous development evidence, not owner or population
preference.

## Integrity and branches

- source, config, comparator-manifest and output hashes are verified;
- two render manifests and two metric reports must be byte-identical;
- automatic failure forbids visual review and closes J1;
- visual aesthetic loss closes the candidate even if metrics pass;
- severe failure vetoes the candidate;
- a retained challenger does not replace safe-rich, become a stock profile or
  open training/router/production integration;
- no result may be described as Velvia or a measured positive-film response.

## DoD

- contract/config committed before render;
- isolated evaluator with focused tests;
- two deterministic full render/evaluation passes;
- gated blind and full-resolution review if applicable;
- complete CPU suite;
- evidence/decision propagation and scoped push;
- Ultimate Goal continues to the next ready leaf.
