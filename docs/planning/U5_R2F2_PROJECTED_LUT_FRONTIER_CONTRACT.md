# U5.R2F2 projected-LUT safety frontier contract

Date frozen: 2026-07-23

Node: `ULT > U5 > U5.R2 > U5.R2F2`

Status: **frozen before any projected LUT or F2 image result**

## Research question

Can a data-independent projection of the already-frozen CanonCGT two-LUT
evidence remove the measured clipping/range risk while preserving visible,
non-basic and reference-sensitive style?

F2 changes no model, reference, image, learned parameter or F1 conclusion.
It is a new B0 development algorithm leaf, not post-hoc rescue of ref08.

## Parent evidence and immutable inputs

- F1's exact 81-pair manifest is fixed at SHA-256
  `77c25ad0d75c9ec5d0eca66a60a133b6e6a0d48f719e6121d3540fc53bfc3f7c`.
- Its two LUTs exactly replay every output, so all F2 renders can be computed
  without loading or running a neural network.
- F1 reference sensitivity passes at median pairwise Delta E76 `8.5322`, and
  six references pass style/non-basic floors, but zero raw policies pass
  clipping/range.
- Raw F1 LUTs contain negative corresponding-channel grid steps and negative
  sampled cell determinants. These are existing F1 diagnostics, not F2
  results.

The exact nine references, nine gold inputs, source hashes, LUT hashes and
three comparator manifests remain inherited and immutable.

## Fixed projection bank

For every one of the 81 frozen input/reference LUT pairs, evaluate exactly
three policies:

1. `clip_only_cap100`: clip every LUT node to `[0,1]`; no structural
   contraction. This is a bounded-node negative control.
2. `safe_contract_cap100`: clip nodes, then find the largest identity
   contraction coefficient at or below `1.0` for which the complete prefix
   remains structure-safe.
3. `safe_contract_cap075`: the same projection with coefficient capped at
   `0.75`.

This yields exactly 27 reference/policy candidates and 243 output images per
pass. No candidate, reference or coefficient may be removed after results.

## Projection algorithm

Convert the public `[3,B,G,R]` array to a project `[R,G,B,3]` LUT. Let `I` be
the exact 17-cube identity and `C = clip(raw, 0, 1)`. For coefficient `a`,

`L(a) = I + a * (C - I)`.

For structure-constrained policies, search only the contiguous prefix from
identity to the fixed cap. Use 33 fixed prefix samples, then 24 deterministic
bisection iterations at the first unsafe boundary. A LUT is structure-safe
only when:

- corresponding R/G/B node steps are each at least `1e-7`;
- all six project tetrahedral cell Jacobian determinants are at least
  `1e-8`;
- every node is finite and in `[0,1]`.

If safety is non-monotone over the sampled prefix, stop at the first unsafe
sample rather than jumping to a later island. Record the chosen coefficient,
minimum steps and determinant for both LUT stages.

`clip_only_cap100` must still report the same structural diagnostics. Because
the experiment requires structure-safe promotion, it cannot survive if its
diagnostics fail.

## Rendering and integrity

Apply both projected LUTs in encoded sRGB using the existing deterministic
project trilinear renderer. Save once as RGB8 PNG without output resize,
effects, per-image adjustment or further clipping policy.

Require:

- all inherited source and LUT hashes;
- all projected LUT hashes and parameter records;
- finite full-resolution output with identical dimensions;
- exactly 243 records;
- a second complete pass with byte-identical manifest, projected LUT and
  image hashes;
- zero model import/inference and zero training.

## Automatic gates

Every candidate must pass:

- all projected LUTs structure-safe;
- gold median style Delta E76 `>=7.0`;
- gold median non-basic residual `>=4.9`;
- worst-gold new hard clipping `<=0.5%`;
- worst-gold raw-final out-of-range fraction exactly `0`.

The surviving bank must also retain matched-input median pairwise reference
Delta E76 `>=2.0`. A projection that makes every reference look the same
closes even if individually safe.

## Frozen shortlist and visual veto

Within each reference, retain only the passing policy with greatest non-basic
residual, then style, then minimum applied alpha, then ID. Next retain at most
one reference per inherited provenance bucket and at most three total.

Only that shortlist may enter three blind rounds against input, safe-rich,
margin-4 anchor56 and the retained E1 density challenger. Inspect every
shortlisted candidate at original resolution on all nine gold inputs,
including the ID11 red-speckle/posterization regression.

One confirmed severe artifact vetoes the candidate. Autonomous vision remains
B0 development evidence, not independent-human or population preference.

## Branches

- **No automatic survivor:** close projected CanonCGT for current product
  value; retain F1/F2 as evidence that style alone is insufficient.
- **Structure-safe but bland:** close the route rather than weaken the style
  floor.
- **Style-safe but reference-insensitive:** close reference routing; retain
  only a global control if independently justified.
- **Automatic and visual pass:** retain at most one generic B0
  reference-conditioned challenger; do not call it a stock expert.
- **Severe visual failure:** reject that policy without relaxing any gate.

No result opens training, current-pixel fitting, LSM, stock claims or
production integration. Failure never stops Ultimate.

## Allowed actions

- implement isolated NumPy projection/render/evaluation code and tests;
- read immutable ignored F1 LUT evidence;
- write two bounded projected output passes and review evidence;
- commit and push scoped results.

## Forbidden actions

- run or modify CanonCGT, train/fine-tune any model or download new weights;
- select only ref08 or tune coefficients from F2 outcomes;
- use references as pairs, targets, teacher truth or stock labels;
- fit projection parameters on current images;
- change gates or candidates after projected rendering;
- integrate into production, release, deploy or redistribute external assets.

## Definition of done

F2 closes only with exact double-render evidence, complete projection and
structure records, unchanged automatic metrics, reference-sensitivity result,
the frozen shortlist, blind/full-resolution review when allowed, focused and
complete tests, branch decision, claim-ceiling propagation, scoped commit and
push.
