# U5.R2H0A Velvia 50 datasheet spectral witness results

Date: 2026-07-24
Decision: **numerically invalid; exact witness closed without visual review**
Parent: `U5.R2H0`

## Outcome

The official-data-only witness is capable of a very strong nonlinear colour
effect, but display-sRGB does not contain enough spectral information to make
that effect stable. The frozen numerical contract closes before any project
photograph is rendered.

Two complete 729-colour evaluations are byte-identical at the report/array
level. The formal report is
`outputs/u5_r2h0a_velvia_datasheet_witness/formal/report.json`, SHA-256
`de0bb65699a19554b6da3090bfe3829e41b72c203a56c01c552692e282ab08ac`.

## Integrity

| Check | Result |
|---|---:|
| exact repeats | 2/2 |
| synthetic colours | 729 |
| non-neutral analysis colours | 714 |
| neutral-ramp samples | 33 |
| wavelengths | 69, 380--720 nm at 5 nm |
| graph annotations | 127 |
| maximum annotation-to-ink distance | 0 px |
| maximum axis residual | 1.4 px |
| full repository tests | 780 passed |

Input lineage is exact: config SHA-256
`0342eb4d4213f82e8afb6c9c58f7f15831cc83ac869cecd6dad2872946c208c1`,
curve-data SHA-256
`bf1cca83e7f81521ff623c3921412e835c3dd09366b1997d69da31262ef637c4`,
software commit `3d91dcdeca456fbc86b58ae41fa1d7adbb81d948`.

## Frozen gates

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| base D65 reconstruction Delta E76, median / p95 / max | `4.54e-12 / 8.27e-11 / 0.0293` | `<=0.25 / 0.75 / 2.0` | pass |
| metamer D65 reconstruction, p95 / max | `8.27e-11 / 0.0293` | `<=0.75 / 2.0` | pass |
| metamer spectral RMS, median | `0.5182` | `>=0.01` | pass; alternatives are distinct |
| neutral Y minimum step | `4.40e-6` | `>=-1e-6` | pass |
| neutral output chroma, max | `6.2247` | `<=4.0` | **fail** |
| metamer output Delta E76, median / p95 | `82.6983 / 144.3486` | `<=2.0 / 5.0` | **fail** |
| metamer/effect ratio, median / p95 | `4.1596 / 14.1769` | `<=0.25 / 0.50` | **fail** |

All spectra and transmittances are finite and bounded. The observer-null
residual of the stress direction is `2.18e-17`, so the enormous downstream
difference is not caused by a colour-matching bug: the alternative spectra
are effectively identical to the input under D65 CIE XYZ and sharply distinct
to the digitized film sensitivities.

## What the result means

The base smooth canonicalizer produces a large witness effect: median Delta
E76 `20.9131`, p95 `45.7921`, maximum `65.6708` against input on the non-neutral
grid. This is important positive mechanism evidence. Datasheet spectral
operators need not collapse into a bland saturation/contrast adjustment.

It is not identified evidence, however. The maximum film-witness disagreement
among D65-colour-matched spectra has median Delta E76 `82.6983`, roughly four
times the base effect at the median. The missing spectrum dominates the
stock-specific transformation. Selecting the smooth base reconstruction would
therefore select one aesthetic convention, not recover how the photographed
scene would expose Velvia.

The fixed Status-A-density approximation also fails its own neutral-axis gate.
This is consistent with the missing analytical dye-amount/base-density
inversion disclosed in H0. It cannot be repaired in this version by adding a
post-hoc white balance.

Raw linear sRGB spans `[-0.1745, 1.0859]` with `22.53%` of channel values out
of gamut. That diagnostic was not the formal closing gate, but it confirms
that a future product candidate would require a separately justified output
mapping rather than silent clipping.

## Branch decision

The exact H0A witness is `numerically_invalid`, and independently
`canonicalizer_sensitive_unidentified`. Consequently:

- H0B fixed visual evaluation does not open;
- no A0 photograph or film pixel may be rendered for candidate selection;
- do not tune exposure, white balance, graph points, metamer scale, viewing
  illuminant, strength or thresholds;
- do not call this a Velvia response, calibrated profile or recovered
  digital-to-film operator;
- do not add a learned model to hide the missing-spectrum result.

A distinct next question is allowed: whether publicly obtainable **measured
natural reflectances**, rather than adversarial feasible metamers, occupy a
much narrower conditional distribution at fixed RGB. That is a new data and
identifiability leaf, not a reopening of H0A. It must preserve the theoretical
non-identifiability finding and remain `film-inspired/datasheet-prior` even if
an empirical natural-material prior is useful.

## Claim ceiling

H0A proves only that a clean-room datasheet witness can be strongly stylized
and deterministic while remaining fundamentally unidentified from
display-sRGB. It establishes neither Velvia authenticity nor product value.
