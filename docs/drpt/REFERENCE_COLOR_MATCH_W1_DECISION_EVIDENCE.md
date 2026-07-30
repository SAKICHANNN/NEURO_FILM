# Repeated W1 Reference-Look Decision Evidence

## Decision

The main neuro-film task completed two byte-identical formal W1 development
runs. This branch independently accepted them through the previously committed
read-only receiver and froze the result as:

```text
status: development-route-closed
branch: paired_upper_bound_only_passes
delivery_algorithm: identity
product_integration_open: false
decision_id: f8661315f5ec8e2d334cad11ca52511e221b4443166429376b301c2e8c0af60e
```

## Provenance

- external software commit:
  `b72b594f5203952aeeabcbd877ad0cadfce6a283`;
- required implementation ancestor:
  `77df1b9004aa0d155ce1b662202db476ec5f677e`;
- external report A/B SHA-256:
  `9b42e9a8edca5a6034e4033d71b37bbe229a53378e4fe031df7c45cf7e8dec68`;
- W1 config SHA-256:
  `6599b99e82d21d7dc5e502e2880a83e238fcd15be74ae4c4f220bc7ca0fc0e26`;
- reserved confirmation accessed: false.

The receiver revalidated the five pinned source files at the report's exact
commit and recomputed the decision branch. No external module, mutable report
or uncommitted main-task file was imported.

## Scientific interpretation

The paired neutral/styled upper bound passes at operator RMSE median/p90
`.02498/.07282`, proving that the bounded explicit operator and optimizer have
headroom. Output-only reference information fails:

- seen single: `.11946/.14838`;
- seen four: `.05774/.08531`, only 38.54% direction retrieval;
- unseen single: `.06464/.10488`;
- unseen four: `.06631/.08191`, worse than global mean and identity;
- content-group balanced accuracy reaches 98.44% versus 25% chance;
- identity references invent a `.06783` transform against the `.01` gate.

This closes the current fixed output-only operator-recovery route. It does not
prove that every perceptual colour-match product is impossible. It does forbid
claiming that a larger descriptor, semantic embedding or neural encoder has
been justified as a rescue on the same evidence.

## Product propagation

- current statistical and CFSM algorithms remain unpromoted;
- ordinary delivery remains identity;
- no confirmation or visual candidate opens;
- stock/film/calibration claims remain forbidden;
- a next candidate must either add a genuinely new information regime or
  explicitly target bounded perceptual Look Approximation rather than hidden
  operator recovery.

## Verification

- dedicated W1 receiver/parser tests: `11 passed`;
- colour-match/preprocess focused tests: `177 passed, 909 deselected`;
- complete CPU collection: `1049 passed, 1 skipped, 36 failed`;
- all 36 failures are the unchanged isolated-worktree ignored-output/CRLF-hash
  baseline; no colour-match test failed;
- compileall and `git diff --check`: pass.

Implementation commit: `9cf7d6c` (`feat: freeze repeated W1 matcher decision`).
