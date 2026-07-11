# FilmCase U4.2 Blind-Audit Foundation

The style/appeal evaluator is an **autonomous visual-evidence protocol**, not
a learned aesthetic model and not a population-preference study. It does not
promote a candidate until U4.1 has a final severe-artifact gold set.

## Fixed protocol

For every frozen sample and candidate, run exactly three blind rounds. Each
round assigns a fresh deterministic anonymous label (`A` …) to each candidate.
The review order is severe artifact, then style strength (1–5), then appeal
(1–5). Candidate IDs are absent from the public sheet and only appear in the
ignored private mapping after review closure.

The pre-registered interpretation is:

1. two or three `severe=yes` votes veto that candidate/sample;
2. one `severe=yes`, any `uncertain`, or an incomplete pass requires
   original-resolution adjudication;
3. adjudication that remains inconsistent is `AMBIGUOUS` and cannot support
   promotion;
4. only survivors may be compared by median style and appeal, with a Pareto
   decision instead of a combined score.

`01/09/53/55/56` are the frozen owner anchors; a neutral/bland control must be
included in every actual run. Matched saturation/contrast controls are required
when a candidate claims palette rather than generic effect strength.

## Implemented isolation

- `src/filmcase/vision_audit.py` builds deterministic permutations and applies
  the rule above from recorded reviews.
- `scripts/build_filmcase_blind_audit.py` writes a public review sheet and a
  private label map only under ignored `outputs/filmcase/`.
- `tests/test_filmcase_vision_audit.py` covers deterministic hiding, `2/3`
  veto, one-vote escalation and duplicate-record rejection.

The first local dry run created 24 sheet/mapping rows: 8 provisional union
seed samples × 6 columns (`01/09/53/55/56` plus bland control) × 3 rounds.
It contains no review result and cannot complete U4.2. Final U4.2 needs the
U4.1 final set, normalized same-input anchor renders, nuisance controls and
recorded repeated visual reviews.
