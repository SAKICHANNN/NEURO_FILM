# U5.R2BQ2B FiveK pairwise compatibility learner

BQ2B closes on development evidence before confirmation. Two runs are
byte-identical at SHA-256
`22a543633f03c3815f82deca5b17bedb1f2b9f61a8b8218225d39c3b47bade44`.
The fixed source-only descriptor, 32-component development-only PCA and
closed-form ridge ranker fit 246 rows from 19 camera groups; 135 rows from five
other camera groups are held out. The ranker only chooses one already-fitted
explicit operator, with pooled-global OOD fallback.

Filtered targets pass every development gate: mean error improves 8.52% over
the fit-only pooled global, the win rate is 66.67%, and the ranker improves
2.68% over frozen tone-layout nearest retrieval. Aligned Expert targets retain
a real but insufficient signal: 5.30% mean gain, positive group bootstrap and
safe tails, but only 54.81% wins versus the frozen 55% requirement. More
importantly, aligned mean error is 0.60% worse than nearest retrieval and wins
only 46.67% head-to-head.

The confirmation population was not decoded or scored by the corrected formal
runs. One superseded pre-formal runner attempt had eagerly decoded it and is
retained only as an explicitly invalid artifact; its report was not inspected
or used for tuning. The model receives no capacity, feature or threshold rescue. This closes
the present hard case-router family despite the strong BQ1v2 evaluator Oracle.
The next legal algorithm leaf is mechanism-distinct: predict a bounded explicit
operator from source descriptors under group-held development, while measuring
style magnitude so improved error cannot be obtained merely by averaging back
toward a bland global transform. FiveK remains a digital-retouch architecture
control, not film or stock evidence.
