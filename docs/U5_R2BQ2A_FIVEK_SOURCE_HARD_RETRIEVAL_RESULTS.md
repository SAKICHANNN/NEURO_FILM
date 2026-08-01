# U5.R2BQ2A FiveK source-only hard retrieval

BQ2A closes after two byte-identical reports at SHA-256
`230133781aa6431b015f2b6dd45ff2c97e5f68ba531b308cd28266622b70b4a5`.
Whole-camera development admits both spatial-photometric and tone-layout
nearest-case retrieval. The frozen rule selects tone-layout without using
confirmation targets.

On the 128-row confirmation split, tone-layout improves aligned Expert mean
error by 5.06%, closes 15.16% of the evaluator-Oracle gap, and passes its
bootstrap, tail, OOD, random-case, support and concentration gates. It wins
76/128 rows (59.375%), one row below the frozen 60% requirement. The filtered
variant passes all gates with 7.45% mean improvement, 61.72% wins and 19.26%
Oracle-gap closure.

This is useful positive evidence that source appearance predicts some case
compatibility, but it does not authorize the nearest-neighbour selector. The
descriptor and threshold are frozen and receive no rescue. The next leaf may
train one group-aware pairwise compatibility model on development targets,
then execute confirmation once with hard Top-1 selection and the pooled global
fallback. Dense blending, learned final RGB, film, stock, calibration,
preference and product claims remain closed.
