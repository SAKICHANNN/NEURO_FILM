# U5.R2S4 Diversified-Distribution Explicit-Operator Development Contract

Date: 2026-07-26

Status: **preregistered, pending U5.R2S3 repeated closure**

## Parent and activation

This is a conditional child of `U5.R2S3D`, not a rescue of its fixed pooled
losses. It may run only if two byte-identical S3 reports close on the frozen
`distribution_fail` branch. Another S3 outcome requires a new adjudication;
this contract then remains unexecuted.

The runner enforces this boundary by loading
`configs/u5_r2s3_unpaired_distribution_operator_pilot_decision_v1.json` and
requiring both `decision_branch=distribution_fail` and
`repeat_report_sha256_equal=true`. Missing or mismatched evidence fails before
any synthetic observation is generated.

S3 report A currently shows the motivating failure: both pooled objectives
reduce their training losses while failing held-out distribution, hidden
operator and replicate gates. That observation is provisional until the
second report completes.

## Research basis and exact epistemic boundary

[Shrestha and Fu, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8bb5a934785817f752e7f9322f9b4d54-Abstract-Conference.html)
show that marginal distribution matching can admit measure-preserving
automorphisms, and establish identifiability under a sufficiently diverse set
of corresponding conditional distribution pairs.

[Shrestha and Fu, ICML 2025](https://proceedings.mlr.press/v267/shrestha25a.html)
also show that implementing this idea with flows is non-trivial: their
identifiable flow construction uses private nonlinear interpolants and a
bilevel/two-stage formulation. A naive conditional flow objective is not
automatically their method.

Accordingly, S4 tests only a small, project-specific candidate:

> Can one shared, bounded `4x4x4` colour flow recover a known synthetic style
> more reliably when fixed RFF-MMD is matched over several correctly
> corresponding content-condition distributions instead of one pooled
> marginal?

It is not a reproduction of either theorem. A pass would not prove that the
current film data has valid conditions or that real unpaired operators are
identified.

## Generated observations

Four latent content cells are generated around preregistered RGB palette
centres: neutral `[.22,.22,.22]`, red-biased `[.72,.24,.20]`, green-biased
`[.22,.68,.28]` and blue-biased `[.24,.38,.76]`. These deliberately separated
synthetic supports test the sufficient-diversity mechanism; they are not a
proposal to classify real scenes by colour. Each scene draws two to four
components with `.08` mean jitter and component standard deviations in
`.04-.10`.

For each style, observation replicate and cell:

- three neutral scenes and three different target scenes are sampled;
- each scene contributes 128 RGB samples;
- neutral and target domains share only the generated cell identity and
  content-cell distribution, never a scene, palette, sample or pixel;
- one hidden analytic diffeomorphic style transforms only the independent
  target-domain samples;
- no paired samples or oracle coefficients reach optimization.

The cell identity is generated truth. It is not inferred from RGB, a scene
model, CLIP or post-hoc clustering.

## Frozen comparison

All methods see the same observations and fit one shared explicit flow:

1. pooled RFF-MMD-192 discards cell identity;
2. conditional RFF-MMD-192 averages equally over four correct cell pairs;
3. shuffled-condition RFF-MMD-192 uses fixed target permutation
   `[1, 3, 0, 2]`.

The primary must pass held-out conditional-distribution, hidden-operator,
A/B replicate, style, range, Jacobian, inverse, coefficient and exact-repeat
gates. It must also reduce median oracle error by at least 25% relative to
both pooled and shuffled controls. The shuffled control must fail its oracle
or replicate gate; otherwise cell identity has not discriminated the intended
operator.

Optimization may use deterministic CUDA float32 to avoid the RTX 5070 Ti
Laptop's prohibitive float64 throughput. Exported flows remain evaluated with
the existing float64 renderer. The S3 API default and all frozen S3 evidence
must remain unchanged.

## Branches

- primary passes all gates: open only an untouched synthetic confirmation,
  and only after a separate real-data auxiliary-condition feasibility
  decision;
- distribution passes but hidden operator fails: close as non-identifying;
- primary fails or does not beat pooled: close without capacity rescue;
- shuffled negative passes: close because conditions are not discriminating;
- structure or repeat fails: invalidate and debug only that boundary;
- synthetic pass but real-condition feasibility fails: retain mechanism
  evidence and close real-data application.

## Real-data stop

No current project pool has established corresponding, style-independent
auxiliary content conditions in both a neutral-digital domain and a target
film-stock domain. Before any real-pixel experiment, a separate DoR decision
must establish rights, stock evidence, cross-domain condition support,
group-held-out connectivity, leakage controls and independence from scene
colour, uploader, scanner, source and geometry.

CLIP/scene embeddings, raw or low-frequency RGB, appearance-inferred
day/night, uploader, scanner, border, aspect ratio, resolution, date and
post-hoc clusters are forbidden substitutes. If this DoR cannot be met, the
correct result is a data gap—not training.

The complete machine-readable freeze is
`configs/u5_r2s4_diversified_distribution_operator_development_v1.json`.
