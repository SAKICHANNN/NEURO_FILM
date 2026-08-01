# U5.R2BN8-BN9 historical preference evaluator

BN8 establishes that the sealed autonomous blind records are sufficiently connected to ask a bounded selector question: 108 source-experiment units across 62 sources, eight experiments, five cohorts and ten canonical explicit arms. This is historical autonomous evidence, not owner or population preference.

BN9 then tests one frozen, interpretable pairwise utility model. It consumes only independently computed source/candidate distribution statistics, never candidate, experiment or camera identity, and holds out whole sources, experiments and source cohorts. The two formal reports are byte-identical at SHA-256 `3e1679cd03f6b328d5b4499269c50883457936d8852f53daa28ce4a9daaf866f`.

- Leave-one-source-out accuracy is `53.70%`, versus `48.15%` for the experiment-global arm baseline: a `5.56` point gain.
- Source interactions add only `0.93` points over candidate-only features, below the frozen `2` point gate.
- Leave-one-experiment-out and leave-one-cohort-out gains over their random-choice baselines are only `1.70` and `0.77` points.
- The source-bootstrap gain lower bound is `-2.96` points and the grouped permutation p-value is `.614`.
- The 251 candidate assets retain zero confirmed severe artifacts, but safety does not repair absent transferable selection value.

Decision: close historical preference learning and retain the fixed global incumbent. Do not rescue this consumed evidence with a larger classifier, new embedding, feature search or relaxed gates. The result directly answers the proposed case-based intuition: these current historical blind choices do not contain stable cross-family evidence that photos with similar measured appearance should receive the same explicit look. Continue with a mechanism-distinct explicit or physical renderer and require new independent evidence before reopening routing.
