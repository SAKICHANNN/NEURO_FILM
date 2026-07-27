# U5.R2W1 Reference-Look Identifiability Development Results

Date: 2026-07-27

Decision: **paired upper bound only; output-only reference recovery fails**

## Reproducibility

- software commit:
  `b72b594f5203952aeeabcbd877ad0cadfce6a283`;
- config SHA-256:
  `6599b99e82d21d7dc5e502e2880a83e238fcd15be74ae4c4f220bc7ca0fc0e26`;
- two complete reports are byte-identical at
  `9b42e9a8edca5a6034e4033d71b37bbe229a53378e4fe031df7c45cf7e8dec68`;
- all four stdout/stderr logs are empty;
- reserved confirmation families and seeds were not accessed;
- no image or visual shortlist was generated.

A host crash interrupted the first B attempt before any report was written.
Its empty logs are not an experimental result. After restart, B was rerun on
the same HEAD and configuration and exactly reproduced A.

## Information-regime result

| Regime / selected practical method | Operator RMSE median / p90 | Key failure |
|---|---:|---|
| Seen, single output-only reference / direction-strength retrieval | `.11946 / .14838` | error, replication, retrieval, strength, content and identity controls |
| Seen, four output-only references / direction-strength retrieval | `.05774 / .08531` | only 38.54% direction retrieval; loses baselines; content and identity controls |
| Unseen, single output-only reference / descriptor ridge | `.06464 / .10488` | p90, replication, family, strength, content and identity controls |
| Unseen, four output-only references / descriptor ridge | `.06631 / .08191` | loses baselines; content and identity controls |
| Unseen global-mean control | `.04657 / .10172` | control, not a look-identified method |
| Unseen identity control | `.04979 / .12487` | control |
| Exact paired-reference upper bound | `.02498 / .07282` | passes |

The paired upper bound reduces median pair MSE from `.0026098` to
`.00001484` and passes the frozen `.07/.10` operator gates. The bounded
operator family and optimizer therefore have sufficient synthetic headroom.

Output-only reference information is the failure. Four references improve
absolute error relative to one reference, but the best unseen four-reference
ridge remains 42.40% worse than the global-mean operator on median error and
33.18% worse than identity. It cannot be promoted merely because its absolute
`.06631/.08191` values pass the standalone operator gates.

## Shortcut and identity controls

The fixed representation remains dominated by content:

- descriptor-ridge content-group balanced accuracy is `98.44%` versus
  `25%` chance;
- direction-strength retrieval content-group balanced accuracy is `85.16%`
  versus `25%` chance;
- nuisance-family probes pass, so this is specifically a content failure
  under the frozen synthetic design;
- on identity references, the four-reference ridge produces median grid RMSE
  `.06783`, and direction-strength retrieval produces `.06037`, both far
  above the frozen `.01` false-positive gate.

This is the central adjudication. Multiple same-look references can make
operator error look numerically acceptable while their representation still
encodes the scene and invents a transform when no look exists.

## Strength-path control

The `53/55/56` negative behaves correctly. Descriptor ridge retains minimum
pairwise direction cosine `.99797` with strength-order Spearman `1.0`;
direction-strength retrieval reaches `1.0` for both. The experiment therefore
does not mistake one operator direction at several strengths for several
categorical modes.

This local control pass does not rescue the failed output-only information
regime.

## Branch decision

The formal branch is `paired_upper_bound_only_passes`.

W1 closes without:

- a larger, semantic or neural descriptor;
- retuned banks, ridge families, thresholds, groups or seeds;
- reserved confirmation access;
- a visual candidate;
- real images, film pixels or film/stock claims.

The result does not prove that every possible output-only reference method is
impossible. It proves that this frozen fixed representation lacks the
required content invariance, and that capacity cannot legitimately be added
as a rescue on the same evidence.

## Next distinct question

Only U5.R2W2F0 global explainability opens. FilmSet supplies aligned input and
Capture One recipe outputs, so W2F0 may ask whether each recipe is at least
coherent in the same bounded global-operator family:

```text
aligned FilmSet input / recipe output pairs
  -> best basic, shared O0 and per-pair O0
  -> held-out recipe coherence and spatial-residual controls
```

W2F1 output-only reference recovery remains closed. W2F0 is Capture One
recipe mechanism evidence only, not film, stock, calibration or unpaired
operator identification.
