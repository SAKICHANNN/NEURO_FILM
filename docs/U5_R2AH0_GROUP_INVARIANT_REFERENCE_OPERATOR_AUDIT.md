# U5.R2AH0 — group-invariant reference-operator source/method audit

Date: 2026-07-28

Decision: `open_synthetic_group_invariance_development_only`

## Question

W1 established that fixed output-only statistics remain content dominated,
even when four unrelated observations share one known synthetic look. P46's
independent external-core diagnosis likewise reports universal overcorrection
from a joint source/reference global fit. Neither result tests the following
distinct hypothesis:

> If training explicitly supplies many groups in which the bounded operator
> is held fixed while content and basic nuisance vary, can a
> permutation-invariant predictor learn a group representation that recovers
> the operator while rejecting content?

This is a supervised synthetic inverse-problem experiment. It is not a claim
that an unpaired film scan identifies its physical digital-to-film operator.

## Primary-source findings

Five exact primary sources were retained as ignored audit inputs:

| Source | Exact local bytes | SHA-256 | Relevant boundary |
|---|---:|---|---|
| Zaheer et al., *Deep Sets*, NeurIPS 2017 | 1,990,691 | `ae1e9a4655b8ada5292b3dd3558dd7e18b05913106ebc36bf9f609c917f163c0` | sum-decomposable permutation-invariant set functions |
| Lee et al., *Set Transformer*, ICML 2019 | 4,877,237 | `965aee34a80935c775156bf96c059e74a226282cbcaee69eb6f9cf6f2603525d` | attention can model interactions inside sets, but adds unnecessary first-pilot capacity |
| Ganin et al., *Domain-Adversarial Training*, JMLR 2016 | 5,670,840 | `801be7cae7acf3ec0df604d40e6129cb9f594e09e6891370b3f7bd2ebef40403` | a discriminative gradient-reversal head can explicitly penalize nuisance information; it is not an RGB generator |
| Bardes et al., *VICReg*, 2021 | 873,759 | `c227c290d5eb2ba459fd502de3b4e1da36a0ea4c88989b7d8153e211751e9e77` | invariance alone can collapse; variance and covariance terms are explicit anti-collapse controls |
| Amos et al., *Meta Optimal Transport*, ICML 2023 | 10,139,141 | `a8e01e0684bf62b9d9b7090cd5dc528711f81a9bbf03c5f1dfec8d8c7815194d` | amortized models can predict continuous colour-transport solutions from measures, but published colour transfer consumes both source and target measures and optimizes canonical OT, not a reference-only film operator |

Canonical source URLs:

- <https://papers.nips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf>
- <https://proceedings.mlr.press/v97/lee19d/lee19d.pdf>
- <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>
- <https://arxiv.org/pdf/2105.04906>
- <https://proceedings.mlr.press/v202/amos23a/amos23a.pdf>

The sources justify architectural primitives, not film truth. In particular,
Deep Sets gives permutation invariance, not content invariance; VICReg avoids
representation collapse, not causal identifiability; DANN can suppress a
declared nuisance label, not prove all nuisance is absent; Meta OT solves a
chosen transport problem and does not identify the real photographic process.

## Clean-room candidate

The candidate is a hierarchical Deep Sets predictor:

```text
unordered RGB samples per final reference
  -> shared point MLP + symmetric mean
  -> shared reference MLP
  -> symmetric mean over 1 or 4 same-look references
  -> bounded O0 velocity-grid parameters
  -> existing deterministic cube-preserving O0 renderer
```

Training may use only newly generated episodes with known O0 operators:

- direct velocity-grid and fixed-grid output losses;
- same-operator/different-content embedding invariance;
- VICReg-style variance/covariance anti-collapse terms;
- an explicitly labelled content-family gradient-reversal diagnostic;
- identity episodes and a continuous strength path.

The predictor never emits final RGB. It emits one bounded explicit stationary
velocity field that the already validated O0 integrator renders. Content
features used for leakage testing are separate from the predicted operator.

## Why this is not a W1 capacity rescue

W1 froze a hand-designed descriptor and ridge/retrieval family on a fixed
32-direction, eight-content design. AH1 must:

- use new seeds and a new online episode population;
- optimize a new group-invariance objective rather than enlarge W1's
  descriptor/regressor;
- keep all W1 reserved confirmation directions and seeds unread;
- compare against the frozen W1 fixed-descriptor/global/identity controls;
- reserve an entirely new operator-family and content-generator confirmation;
- fail if operator recovery improves while a held-out content probe remains
  above its frozen ceiling.

The falsifiable novelty is supervised nuisance variation and explicit
representation invariance, not network size.

## Risks and required controls

1. **Synthetic-prior shortcut.** A model can learn correlations between the
   operator generator and content generator. Operator, content, nuisance and
   batching streams must be independent, with crossed combinations.
2. **Content leakage.** Same-look embedding agreement is insufficient.
   Held-out linear and nonlinear content probes, label shuffles and
   same-content/different-look controls are mandatory.
3. **Identity hallucination.** Identity references must remain near the exact
   zero velocity field on unseen content and nuisance.
4. **Strength splitting.** The `53/55/56` regression remains one direction
   with continuous strength.
5. **Family overfit.** Development success cannot open images. It first needs
   untouched confirmation on a distinct operator generator and distinct
   content distribution.
6. **Renderer safety.** Every prediction is rechecked through O0 range,
   determinant, spectral norm, inverse and exact replay gates. A bounded
   coefficient alone is not a safety certificate.
7. **Real-film identifiability.** Synthetic operator labels do not exist for
   current real scans. Even a confirmation pass remains a computational
   mechanism result until a separate evidence-eligible repeated-look data
   design exists.

## Decision

One development-only AH1 contract is feasible on the local RTX 5070 Ti
Laptop. It may train a small discriminative parameter predictor on generated
sets and render only through O0.

AH1 must close without architecture/seed/loss/threshold rescue if:

- it fails operator recovery against identity/global/W1 controls;
- content or nuisance remains decodable above the frozen ceiling;
- identity false positives or strength-path splitting fail;
- any O0 structural/replay gate fails.

No photographs, current film pixels, W1 confirmation seeds, external code,
generative model, direct neural RGB, stock/mode claim or production
integration opens.
