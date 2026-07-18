# U1.6G4B Row-Stage Invariance Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4B`

**Decision:** closed / frozen byte-parity gate failed

## Question

Can the committed `shape-stable-global-resample-v1` coarse stage be built from
bounded source-row batches while reproducing the materialised G1-v1 coarse and
reconstructed bytes for every legal chunk height?

## Reproducible evidence

- software commit: `34320e9e89ba30e81b304262ad3c1a7b9f372c7e`;
- config: `configs/u1_6g4b_row_stage_invariance_v1.json`;
- config SHA-256: `e005e1639f8aa47a1ce13296aca6d7e601b4f4a1658d1c9f9446a0c28ceaa0cb`;
- script: `scripts/audit_u1_6g4b_row_stage_invariance.py`;
- ignored formal report: `outputs/u1_6g4b/formal_report_34320e9.json`;
- report SHA-256: `e24e990849fd730b66dfe83bcc09b42c10a96b53be5ceead2cffeb99b1b3a0fe`;
- two independent executions produced byte-identical reports.

The diagnostic uses the existing G1-v1 horizontal area reducer and direct
coarse Gaussian. It changes only the number of source rows supplied to the
horizontal reduction call.

## Gate result

| Case | Chunk rows | Coarse result | Reconstructed result | Largest source span |
|---|---:|---|---|---:|
| scalar `257x389`, sigma 52 | 1 | fail, max `4.19e-9`, 1,201 values | fail, max `3.73e-9`, 71,678 values | 9 rows |
| scalar `257x389`, sigma 52 | 5 / 17 | byte-identical | byte-identical | 41 / 137 rows |
| scalar `193x277`, sigma `(40,70)` | 1 | fail, max `3.73e-9`, 330 values | fail, max `4.66e-9`, 37,068 values | 13 rows |
| scalar `193x277`, sigma `(40,70)` | 3 / 11 | byte-identical | byte-identical | 35 / 125 rows |
| RGB `83x107`, sigma 38 | 1 / 4 / 9 | byte-identical | byte-identical | 8 / 27 / 58 rows |

Every reader request remained bounded below the full source height. The
resource condition passed, but the required coarse-byte and reconstructed-byte
conditions failed.

## Interpretation

`np.tensordot` selects batch-shape-dependent floating-point reduction kernels
for the horizontal area operation. The mathematical weights and source pixels
are unchanged, yet the float32 reduction order changes with row-batch height.
Passing chunk sizes are therefore accidental and cannot be selected as a
portable deterministic policy.

This result does not falsify bounded row staging in general. It falsifies exact
row-batched construction of the already committed G1-v1 bytes using its current
batch-sensitive reducer. The frozen G1-v1 result and implementation remain
unchanged.

## Branch decision

Close G4B. Do not weaken byte parity, choose a lucky chunk size, or silently
rewrite `shape-stable-global-resample-v1`. Open U1.6G4C as a separately
versioned, chunk-invariant deterministic area reducer. That version must pass
fresh numerical, visual/severe-artifact and compatibility gates before any
halation executor can consume it.

## Claim ceiling

This is a numerical determinism and invariance diagnosis only. It proves no
effect parity, visual quality, physical halation, renderer integration,
streaming decode, total-memory bound, stock response or 100MP readiness.
