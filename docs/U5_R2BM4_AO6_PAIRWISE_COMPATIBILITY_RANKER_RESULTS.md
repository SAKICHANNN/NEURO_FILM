# U5.R2BM4 pairwise compatibility ranker

Two complete reports are byte-identical at
`0e9ddaec4f6a1caf4debf278170f6293a77d41e95b0f2271547116f7aa84364e`
(stable ID `8aa17d2c430cd0b3c0ef74447ff89087bff11198170a2dfc85c6c7cc2a94f0f3`).

The small ridge ranker is nested by query group, predicts only one hard case
ID, and never sees held targets or generates RGB. It recovers zero median BM1
Oracle gain, improves 1/17 sources and zero outer folds, and has a 2.292x worst
tail. Its advantage over shuffled training labels is 3.27%, below the frozen
5% gate. Eleven sources are non-fallback, so this is not an all-fallback
artifact.

BM1 remains meaningful as an evaluator Oracle: cases can outperform a medoid
and a continuous strength path. BM2-BM4 show that the current 17-source set
does not identify a transferable router. Do not add capacity or tune features
on this population. Future hard-case routing requires materially larger,
independent grouped cases; until then use contextual AO6/global fallback and
advance a mechanism-distinct explicit colour or physical algorithm.
