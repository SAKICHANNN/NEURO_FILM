# RF2.S0 Gold archive-display matrix transplant results

## Decision

Close direct automatic transplant. The six-roll BlueNeg archive-display matrix
does not preserve enough non-basic style on ordinary digital inputs and
violates the frozen clipping budget. Reducing strength makes it progressively
blander without repairing the worst clipping case.

This is a useful negative result: RF1.4B1 identified a real archive processing
mapping, but that mapping is not the missing digital-to-film look.

## Reproducibility

- evaluator commit: `ba62ca2cacf6a35614fb2574fcc2c6548f53d28d`;
- 143 repository tests pass;
- two byte-identical executions;
- report SHA-256 `1cdd2ab4e5ab3beb12e0a5afa14d460312f20ec8a6b9b206d6c6daa110abedab`;
- 47 archive pairs/six rolls set the operator and OOD support;
- 9 frozen digital gold plus 32 stress images only evaluate the transplant.

## Gates

OOD coverage is not the reason for rejection. The archive-only threshold is
2.5722; 8/9 gold and 78.125% of stress images are eligible, meeting the frozen
8/9 and 75% gates.

| Strength | gold median style Delta E76 | residual after matched basic | worst gold raw clipping |
|---:|---:|---:|---:|
| 1.00 | 6.2512 | 1.9425 | 8.700% |
| 0.75 | 4.6904 | 1.4583 | 8.594% |
| 0.50 | 3.1274 | 0.9784 | 8.374% |
| frozen requirement | >=7.0 | >=4.9 | <=0.5% |

At full strength, the non-basic residual is below even the existing safe-rich
control's 2.1452 anchor diagnostic. The worst failures are digital sample 21
(8.70% clipping), sample 09 (7.13%) and sample 11 (1.69%). Sample 11 is also
the sole OOD gold input, consistent with its known red-highlight stress role,
but samples 21 and 09 fail despite being OOD-eligible.

## Visual and architecture consequence

The preregistered protocol forbids visual rendering after an automatic gate
failure, so no candidate contact sheet was generated and no visual preference
claim is made.

Do not repair this branch with a neural model or a stronger nonlinear LUT. Its
problem is not capacity: the learned transformation is an archive restoration/
display adjustment and becomes mostly EV/WB/contrast/saturation on digital
inputs. The next named-stock research must obtain multi-source, stock-labelled
positive scans suitable for stock-internal retrieval or explicit target
statistics, with source/content/scanner controls. It must not reuse the failed
preview-to-proxy transplant as teacher truth.

Machine decision: `configs/real_film_gold_matrix_transplant_decision_v1.json`.
