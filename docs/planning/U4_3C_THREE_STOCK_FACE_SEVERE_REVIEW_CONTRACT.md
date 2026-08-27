# U4.3C three-stock face severe review contract

## Question

Do the unchanged Velvia 50, Portra 400 and Ektar 100 K=1 Look Approximation
baselines avoid a confirmed severe face artifact on the one lineage-bound
FilmSet face array already recovered by U5.R2CB56?

## Frozen scope

- Reuse the two byte-identical 512x512 decoded-sRGB arrays bound by CB56. The
  unavailable original file is not reopened and file-ingress replay is not
  claimed.
- Execute the current public three-stock renderer at `look_amount=1`, seed
  1729 and 128-row tiles. Do not change any profile, statistic, guardrail or
  renderer byte.
- Require exact forward/reverse reports, exact output replay, finite bounded
  RGB, unchanged geometry and zero exact new output boundary samples.
- Review the source and all three outputs at original pixels for dirty or
  blotchy skin colour, isolated impulses, banding/posterization, objectionable
  halos, or loss of eyes/lips/hair/facial-contour structure.

## Decision boundary

Any confirmed severe artifact vetoes that baseline on this one content-stress
input. A pass is one-input autonomous visual evidence only. It is not stock
truth, calibration, target-film closeness, preference, population safety,
multi-stock completion or product promotion. AO6 is excluded and its prior
RF3.D0S severe veto remains unchanged.
