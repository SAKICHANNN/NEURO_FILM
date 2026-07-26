# U5.R2S2 Content-Palette Nuisance Development Contract

Date frozen: 2026-07-26

Parent: `U5.R2S1C`

Status: ready

## Why this node is necessary

S1 answers a deliberately easier question: recover a score flow when samples
come directly from the desired latent palette density. Real film photographs
do not expose that density. Their colour histograms are dominated by what was
photographed as well as by illumination, exposure, process and scan.

S2 therefore makes content an independent nuisance variable before any image
frontier can open.

## Synthetic group design

Each synthetic style is one fixed analytic S0 flow. Six independently
generated content scenes are transformed by that shared flow and aggregated
into a styled histogram.

A neutral control histogram uses six **different**, independently generated
content scenes from the same content meta-distribution. Styled and neutral
scenes are never paired or reused. This is a clean synthetic analogue of a
distribution-matched neutral pool, not a claim that the project currently has
such a real pool.

Training has 384 style groups. Development has 96 unseen styles, each observed
twice with independent content groups. The repeated development observations
measure content sensitivity directly. Seed `28105` remains untouched for a
possible later confirmation.

## Fixed candidates

- raw styled-histogram KDE `.12`, the intentionally confounded S1 control;
- styled-minus-neutral KDE score differences at scales `.25`, `.50`, `1.0`;
- deterministic multi-output ridge with alpha `.1`, `1`, `10`;
- global-mean velocity.

Ridge sees only concatenated square-root styled/neutral histograms and their
difference. Standardization and coefficients fit on training groups only. Its
192 outputs are projected per node to vector norm at most `2`; final RGB
always comes from the explicit diffeomorphic renderer.

No neural network and no direct RGB predictor are allowed in this leaf.

## Development decision

An eligible method must be exactly permutation invariant, structurally safe
and more stable across two content replicates of the same style than it is
collapsed across different styles.

Eligible methods rank by:

1. median oracle-output error;
2. same-style replicate error;
3. p90 oracle-output error;
4. reference separation;
5. simplicity.

An ML winner would prove only that bounded parameter recovery is possible
when exact synthetic style targets are supplied. Current unpaired film data
does not supply those targets.

## Stop rules

- If raw KDE wins but varies mainly with content, S1 does not survive the
  nuisance test and closes before images.
- If global averaging is competitive, no conditional value is identified.
- If no residual method passes, close without adding a neural network.
- A winner may open one separately frozen confirmation; it cannot open film
  pixels, stock fitting, LSM or image rendering.

## Claim ceiling

This is synthetic development about content-nuisance separation under an
idealized, distribution-matched neutral-control assumption. It cannot identify
a real unpaired film operator.
