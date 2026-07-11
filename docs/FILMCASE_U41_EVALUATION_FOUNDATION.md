# FilmCase U4.1 Evaluation Foundation

This is the evaluation foundation, not a statement that U4.1 is complete.
It provides a replayable, fail-closed path for the severe-artifact veto before
any FilmCase candidate can be promoted.

## Components

- `src/filmcase/evaluation.py`: freezes local source identity, checks category
  coverage, records per-sample severity decisions, and reports a stress-set
  Wilson interval.
- `scripts/freeze_filmcase_evaluation_set.py`: reads the existing ignored
  union-40 source index and writes only
  `outputs/filmcase/u41_provisional_eval_set.json`.
- `tests/test_filmcase_evaluation.py`: proves provisional, missing-coverage,
  uncertain-review and final-pass behavior.

The contract fails closed. A candidate can pass gold only when the set is
explicitly `final`, every required category is covered and available, every
gold sample has exactly one non-uncertain adjudication, and no gold sample is
severe. Metrics, clipping, hashes and a passing stress rate never bypass that
requirement.

## 2026-07-11 local seed result

The union-40 seed froze 8 available gold samples and 32 stress samples. Its
gold sample IDs are `01,05,08,09,11,18,21,29`; all have source SHA-256 values
in the ignored frozen-set output. It covers deep shadow, fine detail, foliage,
high key, saturated objects, sky and text/logo.

The local FilmSet input `DSCF01200 iso1600.png` was then visually verified as
a clear child-face stress input and added as `FS_FACE_01`. Its local Kaggle
metadata snapshot declares FilmSet `MIT`; the frozen record carries both image
and metadata hashes and explicitly remains `research-only` and
`filmcase_reference_eligible=false`. It fills face coverage for internal
content/artifact evaluation only. It cannot act as an unpaired reference,
case-memory asset, stock truth, or release evidence.

The resulting 9/32 set now covers every required category, but its status is
still `provisional` and it cannot prove a zero-severe-artifact result: each
candidate still needs complete three-pass visual reviews and any escalation
must reach original-resolution adjudication. This intentionally preserves the
known red-highlight failure at ID 11 as a discriminating stress/gold seed
rather than allowing a clipping-only check to conceal it.

To complete U4.1, a future local/cleared source manifest must add the missing
required categories, freeze their input hashes, and receive independent
recorded visual adjudications for every gold candidate. No image collection,
private-image upload, or external recruitment is initiated by this work.
