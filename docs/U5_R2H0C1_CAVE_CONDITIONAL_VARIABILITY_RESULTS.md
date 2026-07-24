# U5.R2H0C1 CAVE conditional-variability results

Date: 2026-07-24
Decision: **bounded empirical-prior candidate; open H0C2 only**
Parent: `U5.R2H0C`

## Outcome

Measured CAVE reflectances that are close under D65 colour vary far less
through the frozen overlap-only datasheet witness than H0A's adversarial
feasible metamers. The preregistered primary and bootstrap gates pass.

This is positive evidence for researching a deterministic empirical spectral
prior. It is not evidence that RGB uniquely determines spectrum, that the
witness is an identified Velvia response, or that the resulting operator has
product value.

The formal report is
`outputs/u5_r2h0c1_cave_conditional_variability/formal/report.json`, SHA-256
`e59269d40b2d2cb903f47c6b2e190a4755cb7173886d1b2ee6f2ed41169280c0`.

## Source and integrity

| Check | Result |
|---|---:|
| official ZIP entries | 1,120 |
| official spectral PNGs | 992 |
| local path/size/CRC matches | 992/992 |
| missing / extra / CRC mismatch | 0 / 0 / 0 |
| image dimensions | 992 at 512x512 |
| declared 16-bit grayscale | 961 |
| excluded format anomaly | 31 RGBA bands in `watercolors_ms` |
| retained scenes / bands | 31 / 961 |
| transfer snapshot commit | `c7a6862f271b3a958c69a5c020f57064a04cc074` |

The mirror supplied transfer bytes only. Official Columbia central-directory
path, uncompressed size and CRC32 are the byte-level authority. The one
format exclusion was recorded before any witness output was evaluated.

## Population and support

The fixed 16x16 per-scene cell medians produce 5,951 valid 31-band
representatives. The output-independent radius-1 policy retains 263 unique
cross-scene pairs:

| Support gate | Result | Gate | Status |
|---|---:|---:|---|
| pairs | 263 | >=256 | pass |
| scenes | 31 | >=16 | pass |
| scene pairs | 157 | >=32 | pass |
| maximum scene incidence share | 9.13% | <=15% | pass |
| maximum scene-pair share | 3.04% | <=6.25% | pass |

Input Delta E76 is 0.670 median / 0.957 p95 / 0.999 maximum. Spectral RMS is
0.0131 median / 0.0418 p95, so the population contains real conditional
spectral differences but is much less adversarial than H0A's feasible set.

## Frozen result

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| output Delta E76 median | 1.3757 | <=5.0 | pass |
| output Delta E76 p95 | 5.4742 | <=12.0 | pass |
| scene-bootstrap median 95% UCB | 1.9088 | <=7.5 | pass |
| scene-bootstrap p95 95% UCB | 9.7605 | <=18.0 | pass |
| median / H0A adversarial median | 1.663% | <=25% | pass |
| p95 / H0A adversarial p95 | 3.792% | <=25% | pass |
| same-spectrum replay maximum | 0 | 0 | pass |
| exact repeats | 2/2 | 2/2 | pass |

Unconditioned cross-scene controls are appropriately broad: input/output
Delta E76 medians are 38.91/38.25. The conditional result therefore does not
come from a globally weak witness.

## Non-binding tail diagnostic

After the frozen decision, the high-spectral-distance tail was inspected only
to design the next experiment. For spectral RMS >=0.02, 61 pairs have output
Delta E76 2.27 median / 9.76 p95; for RMS >=0.03, 37 pairs have 2.41 / 11.49.
At RMS >=0.04 only 16 pairs remain and reach 5.02 / 13.54, with a 16.70 worst
case. These small post-hoc subsets do not change the pass. They show why H0C2
must report tails and OOD fallback rather than treating the empirical prior as
a universal inverse.

The overlap-only witness itself has raw linear-sRGB range
`[-0.0268, 0.8948]` and 4.41% channel values below/above gamut. This is not an
H0C1 gate, but no future visual/product path may silently clip it.

## Interpretation

H0A and H0C1 are compatible:

- H0A proves that arbitrary bounded D65 metamers can produce enormous and
  unidentified witness differences;
- H0C1 shows that one database of approximate measured real-material
  reflectances occupies a much narrower conditional region;
- therefore an empirical convention may be useful, while physical recovery
  remains impossible from display RGB alone.

This is the first evidence in this branch that a non-bland spectral mechanism
might be made operational without a generative RGB model. It does not yet say
which spectrum should be chosen for an unseen photograph.

## Branch decision

Open only `U5.R2H0C2`: compare the smooth H0A canonicalizer with the simplest
cross-scene hard measured-spectrum retrieval under group-held-out evaluation.
The target is the known witness output of CAVE reflectances, not film pixels.
The method must remain explicit, deterministic, sparse and OOD-fallback-safe.

H0C2 must not:

- fit or evaluate on real-film pixels or owner anchors;
- call CAVE targets film truth;
- use same-scene neighbours;
- add a neural network before simple retrieval is adjudicated;
- integrate into production or render project photographs;
- weaken the H0A non-identifiability or H0C1 tail caveat.

External measured-spectrum replication remains required before any broader
empirical-prior claim.

## Reproducibility

- config SHA-256: `b878aa9e2de93b5f633df084078a12d0ea204d78c6c8ecdc2b67cb03cccfb9d1`;
- official-tail SHA-256:
  `464817d87acef9e4db91a5420e98a34f4c48df697769756e9fc0b57fcce43015`;
- H0A config SHA-256:
  `0342eb4d4213f82e8afb6c9c58f7f15831cc83ac869cecd6dad2872946c208c1`;
- curve-data SHA-256:
  `bf1cca83e7f81521ff623c3921412e835c3dd09366b1997d69da31262ef637c4`;
- software commit: `4e3255983e349a9f80d0c003a6c9e10b96876449`;
- focused H0A/H0C1 tests: 8 passed;
- full CPU suite: 783 passed in 65.54 seconds.

## Claim ceiling

One source-limited empirical conditional-variability pass under an
overlap-only datasheet-prior Look Approximation. No identified spectrum,
digital-to-film operator, Velvia response, calibration, preference, visual
safety or product value.
