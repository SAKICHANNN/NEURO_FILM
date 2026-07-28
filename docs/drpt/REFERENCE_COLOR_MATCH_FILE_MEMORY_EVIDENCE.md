# P154 Reference-Match File Memory Evidence

## Result

The P151-P153 lifetime changes preserve every durable artifact and reduce the
measured 6 MP file-path peak, but the improvement is below both preregistered
memory thresholds. P154 therefore closes as
`measured-improvement-below-frozen-gate`, not as a memory-readiness pass.

The frozen contract is
`configs/reference_match_file_memory_v1.json` at commit
`b4975cc4d062cb7c9f224496ef700fbd2ee11b27`. The harness is
`scripts/audit_reference_match_file_memory_v1.py` at commit
`7c059e7667ae5c70e1c1239eb8a49f0f4dd6a20f`. Its ignored raw report has
SHA-256 `41e28f788cfb9905d568835934c1da4afc7fef92770b5952b1a58eb598134dca`;
the durable result is
`configs/reference_match_file_memory_decision_v1.json`.

## Measured matrix

The exact order was baseline, candidate, candidate, baseline. The baseline is
P150 `34259086fbbd6e479753a9f72bf77f2e0d463e93`; the candidate is the
P151-P153 functional payload
`de57918e23568ad6321262903addfac7f2349d3c`.

| Run | Revision | Peak process-tree RSS |
|---:|---|---:|
| 1 | baseline | 1,383,059,456 B |
| 2 | candidate | 1,353,474,048 B |
| 3 | candidate | 1,316,970,496 B |
| 4 | baseline | 1,395,191,808 B |

- baseline median: `1,389,125,632 B`;
- candidate median: `1,335,222,272 B`;
- reduction: `53,903,360 B` (about 51.41 MiB);
- candidate/baseline ratio: `0.9611961951041157`.

The frozen gates required at least `67,108,864 B` reduction and a ratio no
higher than `0.92`. Both fail. The thresholds were not changed after observing
the result.

## Behavior and integrity

All four fresh workers completed within the timeout. The 20 ms monitor covered
the complete venv-launcher/worker process tree. Every run:

- returned default `identity-fallback`;
- produced output SHA-256
  `2fdb6f01c158c096f49d7e2108f61b6b706f0df873160ba17616068bf320c26d`;
- produced recipe SHA-256
  `70a357489958a32bf28f9e7c59a28dfe0bd6350b3cdca499cd75cfce00877bce`;
- reproduced normalized semantic-report SHA-256
  `ef0fa5a01c50808b837c7f2f1b7429526e06c1421884ea94018af76a3c6fabec`;
- left zero transaction staging temporaries and zero observed workers alive.

Both detached worktrees were removed successfully. The preflight observed about
33.65 GB available physical memory, about 48.24 GB free disk, and no unrelated
process above the frozen 8 GiB competitor threshold.

## Interpretation

P151-P153 remain correct lifecycle improvements: weak-reference regressions
prove the obsolete reference, source and rejected-candidate objects are no
longer retained across their next stage, and this independent process result
measures a smaller peak. The whole-path 6 MP peak does not fall enough to claim
the file adapter memory target is closed.

The current render code retains multiple full-frame Lab/RGB arrays and invokes
full-frame safe-Lab/gamut operations. That is a code-path observation, not yet
phase-attributed RSS evidence. Any later optimization must measure phases or
provide an exact bounded execution path; P154 cannot be relabeled by lowering
its gate.

This result does not change A1/A4/A5, product promotion, producer compatibility,
RAW/HDR/video support, FilmFX ownership, main renderer memory evidence, or any
target-platform claim.
