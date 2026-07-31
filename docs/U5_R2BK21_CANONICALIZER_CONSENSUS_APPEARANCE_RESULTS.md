# U5.R2BK21 Canonicalizer-Consensus Appearance Results

Date: 2026-07-31  
Decision: `close_frozen_consensus_policy`

## Result

BK21 tested a hard appearance-matching policy after BK20 proved that unpaired
sets cannot identify the hidden transform. Two independent objectives fitted
the same bounded O0 explicit flow:

- 48 fixed sliced-quantile projections;
- 144 fixed RFF-MMD features.

The policy never averaged operators or RGB. It selected one whole candidate
only when held-out cross-objective appearance improved, the two operators
agreed within the frozen full-cube threshold, and structural gates passed.

Two complete CUDA float32 runs are byte-identical at
`90ea9ca31e0bf6867a091abdce726fe045b578ad32d5372b4cd8e12722013fcd`.

## Automatic evidence

| Scenario | Policy | Operator disagreement | Cross-objective improvement | Outcome |
|---|---|---:|---:|---|
| material shared look | hard sliced-quantile candidate | 0.01081 | 97.15% | expected pass |
| appearance already matched | identity fallback | 0.00507 | -21.13% | expected pass |
| content-confounded reference | hard sliced-quantile candidate | 0.05007 | 64.98% | **failed negative control** |

The generated true look was material at `0.13134` RGB RMSE from identity.
Every retained structural gate passed: sampled output stayed in
`[0.03008, 0.96837]`, minimum Jacobian determinant was `0.14591`, maximum
Jacobian norm `2.05950`, inverse error `1.72e-8`, and replay error zero.

## Interpretation

The failure is informational, not numerical. Both appearance objectives can
agree and strongly reduce held-out distribution loss while jointly following
a content-distribution shift. Their agreement is therefore a useful
sensitivity diagnostic but not sufficient evidence that the learned look is
content-independent.

The frozen `0.06` disagreement threshold is not tightened after seeing the
failure. No third loss, larger flow, optimizer extension or threshold rescue
is allowed.

The next distinct algorithm question may test a content-cell-balanced
worst-case gate whose cell identity is defined independently of colour
appearance. Real use additionally requires rights-cleared
same-look/different-content connectivity. Until then, reference-only output
remains a film-inspired appearance match with exact global/identity fallback,
not a digital-to-film operator, stock response or latent mode.
