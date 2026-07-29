# U5.R2AV1 — Filmr external-control results

## Decision

Close the pinned Filmr Velvia-labelled control at the automatic gate. It is
strong and non-basic, but its worst new hard clipping is far above the frozen
budget. The protocol therefore forbids visual candidacy and parameter rescue.

## What was tested

- upstream: `W-Mai/filmr` release `v0.13.2`, commit
  `fee388e213dd77bfe6977b63071b1c136213373a`;
- exact Windows release ZIP SHA-256
  `48d8421e...bb70d6` and CLI SHA-256 `6af3dedb...64639c`;
- official `fujifilm-velvia-50` preset exported through the CLI;
- only four stochastic grain amplitudes set to zero. Curves, matrix, spectral
  parameters, exposure, white balance, halation, vignette and output mode were
  unchanged;
- all 9 frozen gold and 32 stress photographs, rendered twice.

The two committed-code runs produce the same report SHA-256
`583f665d...84d4d`. All 82 outputs and exposure receipts repeat exactly.

## Automatic result

| Metric | Result | Gate |
|---|---:|---:|
| Gold median style Delta E76 | 15.9384 | >= 7.0, pass |
| Gold median non-basic residual Delta E76 | 5.1711 | >= 4.9, pass |
| Worst gold new hard clipping | 2.0589% | <= 0.5%, **fail** |
| Worst stress new hard clipping | 2.9315% | report-only |

Gold ID 09 is the worst gold clipping case; stress ID 10 is the worst overall.
All images use the same `Exposure time: 1.0000s`, so this run does not provide
evidence for useful image-adaptive exposure.

## Evidence boundary

The implementation is a genuinely separate code family from spektrafilm, but
its stock presets are not measurement-calibrated profiles. The inspected
Velvia response uses parameterized Gaussian spectral peaks, a generic
error-function curve, hand-authored interlayer/matrix terms, and a CIE-derived
surrogate called an sRGB camera sensitivity. The repository does not provide
row-level primary-source lineage for these complete parameters.

Accordingly, the only valid retained claim is:

> A generic physical-inspired external control can be much more stylized than
> a saturation-only baseline, yet still fail the project's severe-first output
> safety budget. Physics vocabulary and a stock name do not establish a safe
> or calibrated stock operator.

No visual review, tuning, fitting, training, profile import or product
integration opens.

Authority:

- `configs/u5_r2av1_filmr_external_control_v1.json`
- `configs/u5_r2av1_filmr_external_control_decision_v1.json`
- ignored formal report:
  `outputs/external_controls/filmr_v0132_velvia50_nograin_v1/automatic_report.json`
