# RF1.4A stock-preview identifiability results

Date: 2026-07-16

Decision: close the post-negation-preview stock-signal route as structurally
insufficient or nuisance-confounded

## Reproducibility

At software commit `f5a7bbd845dfd29d61181e03554d40a19f26936d`, two complete
runs were byte-identical:

- report: `outputs/real_film/stock_pilots_v1/rf1_4/report.json`
- SHA-256: `6ca752d0ddfed89d70f25843e7da6adb2936e8bb47e917caedeae85f899982c9`
- full-resolution display-operator fit: not performed
- holdout/vote/permutation unit: physical roll

## Structural result

GA100 and Konica were closed before feature interpretation. Both contain a
one-frame roll and only one content cell with at least two frames. GA100's roll
counts are 13/1/2; Konica's are 4/5/3/3/1/2/4.

NPH400 and Gold100 form the only comparable clique: 103 frames and 11 rolls.
This admission is only sufficient to run the shortcut audit; it does not prove
content matching.

## Frozen descriptor results

| Descriptor | Held-out-roll accuracy |
|---|---:|
| Primary center-80 RGB distribution | 0.727 (8/11) |
| RGB mean/std only | 0.727 (8/11) |
| Luminance only | 0.818 (9/11) |
| Full-frame RGB distribution | 0.818 (9/11) |
| Center-60 RGB distribution | 0.727 (8/11) |
| Per-channel-standardized RGB distribution | 0.727 (8/11) |
| Date/content/dimensions/border shortcut | 0.909 (10/11) |

For the primary descriptor, the count-preserving roll-label permutation test
gave `p=0.191`; its null 95th percentile was 0.818. The observed result does
not reject the roll-level null. It also ties the simple global-colour baseline
and loses to the nuisance shortcut.

## Interpretation

The current BlueNeg previews do not provide evidence for a generalized named
stock character. A larger classifier could improve archive recognition by
learning date, location/content, border/crop, exposure or scanner/render cues,
but that is exactly the failure mode this gate forbids.

This result closes preview-only learning for all four pilots:

- GA100 and Konica: structurally unidentified;
- NPH400 and Gold100 preview comparison: null not rejected and nuisance wins;
- no physical-density, stock-response, display-colour or `S2` claim;
- no capacity escalation, GPU model or frame-random rescue.

The surviving child is Gold100's separately verified display-proxy lane. It
requires a new preregistered paired-transform consistency test and does not
inherit a positive result from this failed preview audit.

Machine-readable decision:
`configs/real_film_stock_identifiability_decision_v1.json`.
