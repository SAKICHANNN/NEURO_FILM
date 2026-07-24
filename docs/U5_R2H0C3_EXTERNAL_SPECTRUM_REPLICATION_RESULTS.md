# U5.R2H0C3 external measured-spectrum replication results

Date: 2026-07-24  
Decision: **external replication failed; close broad hard-spectrum prior**  
Parent: `U5.R2H0C2`

## Outcome

The H0C2 CAVE hard Top-1 result does not replicate on independent USGS
measured reflectances. At the frozen D65 Lab radius of Delta E76 `1.0`, hard
CAVE retrieval wins only `6.52%` of selected external queries. The smooth
bounded reconstruction is much more accurate, and hard retrieval points in
the wrong direction in every official USGS material chapter.

This is a strong source-generalization failure, not a low-power or threshold
edge case. The broad idea that an RGB/Lab-near measured material supplies a
generally useful hidden spectrum for the synthetic film witness is closed.

Formal report:
`outputs/u5_r2h0c3_external_spectrum_replication/formal/report.json`, SHA-256
`ed263e7ef6c4815eb3fa8f545b3a0ec7dc4c6967d0deb7b8b70d521e15c8bfdf`.

## Integrity and population

- official USGS `ASCIIdata_splib07a.zip`: 21,812,828 bytes;
- official/local MD5: `bfe74068d85811e52e5e07d017720a17`;
- source SHA-256:
  `d232645740869a82aafcad5839448c50b1dc72965ce042d1374f29b7a798a91c`;
- 1,732 eligible AREF queries, 1,592 filename-derived sample groups and seven
  official chapters;
- fixed bank: the unchanged 5,951-row / 31-scene CAVE population;
- USGS contributes zero bank rows and no feature, threshold or model fitting;
- two complete runs are hash-identical;
- 1,732-row query lineage CSV SHA-256:
  `dc2e38a2984af8932f222d24de08681a6570484558822952989253c768b28c1c`.

The source/parser rejected 679 error-bar files, 683 non-AREF records, 25
records without full visible overlap, five out-of-bounds reflectances and 19
relative-Y exclusions. Nothing was clipped into eligibility.

## Frozen-policy result

| Metric | Smooth | Hard CAVE Top-1 |
|---|---:|---:|
| selected median error Delta E76 | 0.2714 | 1.5610 |
| selected p95 error Delta E76 | 4.3308 | 5.6898 |
| selected mean error Delta E76 | 0.7757 | 2.2328 |
| selected maximum error Delta E76 | 8.7580 | 15.8368 |

The fixed threshold selects 184/1,732 queries (`10.62%`) across 175 sample
groups and all seven chapters. Maximum sample-group share is `1.09%`; maximum
chapter share is `48.91%`. All support gates pass.

Performance fails decisively:

- hard win rate: `6.52%`, bootstrap 95% interval `3.24%--10.33%`;
- hard loss rate: `93.48%`;
- median relative reduction: `-475.08%`;
- group-bootstrap reduction interval: `-661.01%--363.89%`;
- evaluable chapters with lower hard median: `0/4`;
- all seven chapter point medians favour smooth;
- complete fallback policy p95: `6.5652`, worse than smooth `6.3530` by
  `0.2122`.

Absolute hard median/p95 limits happen to pass, but they cannot compensate for
failed comparative, bootstrap, chapter-direction and full-policy gates.

## Interpretation

H0C2's within-CAVE improvement was source-conditional. CAVE image-cell
reflectances and USGS material spectra have different sampling and smoothness
structure; close D65 Lab is insufficient to transfer the missing spectral
degrees of freedom between them. In the external population the smooth
bounded solution is already very accurate (`0.596/6.353` median/p95 across all
queries), while an arbitrary CAVE metamer introduces error.

This does not prove spectral modelling is useless. It proves the tested
source-agnostic hard prior is not justified. RGB-to-spectrum would require an
independently validated estimator and application distribution; the current
project has neither, and H0A's theoretical non-identifiability remains.

## Branch decision

Close the broad cross-source empirical-prior branch:

- do not retune the Delta-E radius;
- do not add USGS records to the bank and rerun the same-source problem;
- do not blend Top-K spectra;
- do not train a neural RGB-to-spectrum model as confirmatory rescue;
- do not use H0C2 as teacher truth for film or photographic pixels.

No RGB spectral-estimation, film fitting, photograph rendering, visual
shortlist, stock/calibration, production or LSM work opens from H0C3. Ultimate
remains active and returns to another independently justified explicit film
algorithm or deterministic product leaf.

## Reproducibility

- config SHA-256:
  `429cf0d828ebb27976ce02d704afd562f37491313ac285281902790800b4c95a`;
- source decision SHA-256:
  `5c3d78a4c7377788c1c2df8c4706be2b96bd7ab913369b96cd2ad4613dc421fb`;
- software commit: `29ea56d23cc176048ce339201cc7ce3ec7581536`;
- exact runs: 2/2;
- focused H0A/H0C1/H0C2/H0C3 tests: 15 passed.

## Claim ceiling

One failed cross-source replication of a fixed hard measured-spectrum prior
under a synthetic datasheet-prior witness.

