# SF2.9B ColorReference batch-family result

Date: 2026-07-28

Status: source and frozen confirmatory gates pass; measurement signal only

## Result

The official ColorReference archive contains 51 Ektachrome-compatible K3
charges and 49 Velvia-50-compatible V3 charges with spectral data. All 100
archives (29,856,706 bytes) pass download, ZIP CRC, deterministic parse,
288-common-patch and 41-point 380--780 nm spectral checks. Two formal source
reports are byte-identical at SHA-256 `5f4f392c...a00550`.

The analysis split was committed before fitting. Calendar years ending 8/9
form the 13-charge confirmatory set; the other 87 charges alone fit pooled
patch aims, normalization, PCA and logistic regularization.

| Confirmatory metric | Result |
|---|---:|
| Spectral balanced accuracy / ROC AUC | 1.000 / 1.000 |
| Lab balanced accuracy / ROC AUC | 1.000 / 1.000 |
| Exact fixed-score stratified permutation | 1 / 1,716 (`p=.000583`) |
| Calendar-year-only balanced accuracy | .536 |
| Batch-error-only balanced accuracy | .357 |
| Spectral gain over best nuisance | .464 |

Two pilot reports are byte-identical at SHA-256
`97110652...50caa`; stable evidence ID is `b845b3ad...52b72`.

## Interpretation

The target-family measurement signal is stable across held-out production
charges and is not explained by the frozen year-only or reported
manufacturing-error controls. This materially strengthens the evidence that
one averaged scanner-target profile is insufficient for these two families.

It does **not** identify a photographic appearance operator. These are
manufactured calibration targets, not common camera exposures. The separation
may combine film material, recorder compensation, IT8 aim adjustment, process
and target manufacture. Ektachrome is only a compatible product family, not
one exact stock.

No renderer fitting, stock learning, latent-mode study or product integration
opens. The next bounded research leaf asks whether family-specific
nonnegative optical-density spectral bases improve held-out-charge
reconstruction over one shared basis and wrong-family controls. Any retained
basis remains a target-family physical prior, not calibrated dye layers or a
digital-to-film transform.

Evidence is frozen in
`configs/sf2_9b_colorreference_batch_family_decision_v1.json`.
