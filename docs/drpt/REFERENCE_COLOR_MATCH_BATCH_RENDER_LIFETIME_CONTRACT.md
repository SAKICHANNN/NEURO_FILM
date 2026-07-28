# P161 — encoded render lifetime contract

P160 passes its 24 MP batch envelope but shows the preceding rendered
`WorkingImage` remains live into the next source load. P161 may change only
that obsolete Python-object lifetime.

## Frozen change

- Baseline workload and artifacts are the exact P160 contract and formal run.
- After one rendered image is encoded and its scalar diagnostics/safety fields
  are appended, release the enclosing render result before loading the next
  source.
- Do not change fitting, rendering, safety, encoding, ordering, report,
  transaction, schema, rail or producer behavior.

## Frozen proof

- A weak-reference regression must prove each rendered image remains alive at
  encoder entry and is collectible before the next source load.
- The focused file suite and the full non-manifest color-match suite must pass.
- A fresh baseline/candidate comparison reuses the exact P160 inputs and
  ordered three-source workload in interleaved order
  `baseline,candidate,candidate,baseline`.
- Ordered outputs, recipe and normalized report must remain exact across every
  run; every source remains identity fallback and cleanup remains complete.
- Candidate median RSS must fall by at least 200 MiB and be no more than 0.92
  of baseline median. Candidate/baseline median wall ratio must be at most
  1.05.

## Claim ceiling

A pass proves only removal of one encoded-output lifetime overlap on the local
Windows/Python SDR PNG batch. It does not establish arbitrary-N constant
memory, target-platform, media-rail, quality or product readiness.
