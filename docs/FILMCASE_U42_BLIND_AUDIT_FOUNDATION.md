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
  private label map only under ignored `outputs/filmcase/`. Given the
  normalized replay manifest, it also writes anonymous assets such as
  `assets/round_1/sample_11/A.png`, with no candidate directory name exposed.
- `tests/test_filmcase_vision_audit.py` covers deterministic hiding, `2/3`
  veto, one-vote escalation and duplicate-record rejection.

The first local dry run created 24 sheet/mapping rows: 8 provisional union
seed samples × 6 columns (`01/09/53/55/56` plus bland control) × 3 rounds.
It contains no review result and cannot complete U4.2. Final U4.2 needs the
U4.1 final set, normalized same-input anchor renders, nuisance controls and
recorded repeated visual reviews.

## Normalized color-only replay

`scripts/render_filmcase_anchor_set.py` now replays the five anchors plus the
bland control on the same frozen gold inputs. It defaults to gold only (9
inputs × 6 candidates = 54 renders), disables grain, records input/output
hashes and preserves the recovered union-40 color/luma/gamut parameters. The
outputs are explicitly `anchor-inspired normalized replays`, not claims of
pixel-identical historical reproduction.

On PowerShell, quote comma-separated `--samples` and `--candidates` arguments
so it does not split or numerically normalize identifiers such as `01`.

A directed, non-blind sanity inspection confirms that the known union ID 11
red-lit bicycle/ColorChecker case remains discriminative: normalized
`09/56/53` all exhibit conspicuous red highlight speckling on the metal frame,
whereas the bland safe-rich control suppresses it at the cost of palette
strength. This observation is a rubric/failure-gallery check only; it is not a
substitute for the blinded three-pass verdict.

## EXP-VIS-00 single-case evidence

On the red bicycle/ColorChecker failure case (ID 11), the controlled
`anchor56` chroma-gamut variant initially removed visible red speckling but
created 26.06% new hard clipping because it omitted the mandated output margin.
The same variant with `output_margin=4` visibly retained smooth red highlights,
reported zero new clipping and an output range of `[4,251]`. Its L-SSIM was
0.9647 and its mean chroma fell from 30.93 to 21.46, so the result is a
trade-off candidate, not a free safety win.

This is only a one-image, directed visual and diagnostic check. It permits the
margin-bounded chroma variant to enter the frozen nine-sample replay and blind
review; it does not promote it, nor does it demonstrate that style salience is
preserved across the gold set.

The permitted replay completed on 2026-07-11: 9 gold inputs × 7 columns
(five anchors, bland control and unpromoted challenger) = 63 color-only
renders, followed by 189 anonymous assets across three blind rounds. The
replay manifest and review assets remain ignored local evidence. No rating has
yet been entered or unblinded.
