# SF1.0B connected stock identifiability results

Date: 2026-07-16
Decision: **close the current four-cell pools for stock learning; preserve them as aesthetic/failure evidence**.

## Frozen design

The audit uses 104 hash-verified files in four cells: Commons Ektar100/UltraMax400 and YFCC Ektar100/Velvia50. It holds out complete photographer groups, gives each author-by-stock unit one vote, and compares RGB distribution against luma, grayscale HOG, geometry/border, low-frequency RGB and per-image-standardized RGB. It uses 499 count-preserving group permutations and 4,000 class-stratified paired group bootstraps.

The report ran twice with identical SHA-256 `a5877346c8a56edb7ca338f564bb7750dc3d439d1a017a02127e956cbd7a68d8`.

## Results

| Contrast | RGB BA / p | Strong controls | Decision |
|---|---|---|---|
| Commons Ektar vs UltraMax | 59.82% / 0.280 | 4x4 RGB 80.36% / 0.024; standardized RGB 73.21% / 0.094 | global stock colour fails; spatial scene-colour signal is not attributable |
| YFCC Ektar vs Velvia | 65.71% / 0.188 | luma 75.71%; HOG 72.14%; geometry 75.71%; 4x4 RGB 89.29% / 0.002 | content, tone and acquisition shortcuts dominate |
| Ektar Commons vs YFCC source | 48.57% / 0.592 | geometry/border 92.86% / 0.016; HOG 75.71% | strong source fingerprint confirmed |

Neither same-source stock edge passes the preregistered gate. In both cases RGB fails the 70%/p<=0.05 criteria and fails to beat the strongest nuisance control with a positive paired confidence interval. The YFCC edge is especially invalid because luma, grayscale content and geometry already predict the label.

The apparent low-frequency and standardized-colour performance is useful evidence for a content-aware retrieval product, but it is not evidence of a stock response. A larger network would increase shortcut capacity and is forbidden.

## Shared-author check and next gate

The retained Commons Ektar/UltraMax cells share only one author. The retained YFCC Ektar/Velvia cells share none. The raw YFCC15M text pools share three UIDs, but all relevant current pages fail the frozen live CC-BY check.

The only justified data expansion is the complete YFCC100M metadata index, because the 7.35M subset is only about 7.4% of the 99.2M-photo population and a larger index may yield enough same-author multi-stock groups for an author-controlled experiment. SF1.1 is metadata-only. It stops unless the exact-stock filtered index has at least 12 shared Ektar/Velvia author UIDs, each stock has at least 100 permissive rows and 30 UIDs, and a later live-rights preflight projects at least five usable shared authors. No image download is allowed in SF1.1.
