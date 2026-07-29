# U5.R2AQ2 held-group explicit proxy baselines

## Decision

AQ2 closes the current ColorReference recorder-to-slide operator branch. No
candidate satisfies the frozen joint held-target-set/held-source-slide
accuracy and raw-XYZ bounds, so there is no champion and no photo or higher
capacity stage opens.

This does not contradict AQ1: the six target sets repeat very consistently,
but repeatability of a fixed source patch is not the same as generalization to
an unseen recorder RGB grid.

## Exact evidence

- Config SHA:
  `596ba1379aa9689dd0c02f4b359fd2a9513a59e19af503939d7bb33ab4da34f2`
- Software commit:
  `75839dee1f4d7b4217db51199c65186b7c76cbb0`
- Two exact formal reports:
  `be080bbb1aed922efe60854dae34029b2aa4d9b28ddd135d8c945f1847f56d48`
- 30 primary folds each fit on 5,760 rows with both the held target set and
  held source slide excluded, then score 288 untouched rows.
- Six held-set-only and five held-slide-only folds separate the two nuisance
  axes.

The provided XYZ and Lab measurements agree (median/p95
`0.0265/0.1576` Delta E76), so target-space parsing is not the failure.

## Primary joint holdout

| Model | mean Delta E76 | worst-fold mean | raw negative XYZ | Decision |
|---|---:|---:|---:|---|
| identity | 106.91 | 173.26 | 0% | control |
| target mean | 56.86 | 66.20 | 0% | inaccurate |
| diagonal affine | 107.04 | 172.47 | 13.91% | worse than identity |
| full affine | 42.61 | 57.83 | 15.14% | unbounded/inaccurate |
| nonnegative affine | 22.79 | 30.07 | 0% | bounded but above worst-fold gate |
| quadratic full | 13.72 | 17.71 | 3.29% | accurate direction, but unbounded and above worst-fold gate |

The quadratic candidate improves 67.81% over full affine, but it is ineligible:
3.29% of raw XYZ components are negative versus the frozen 0.5% ceiling and
its worst fold exceeds 15 Delta E76. Clipping was explicitly forbidden.

## Failure localization

Held-target-set-only quadratic evaluation reaches 11.41 Delta E76. The
held-source-slide result is 13.72, nearly identical to the joint result, so
unseen recorder-grid generalization dominates. Slide 4 is consistently worst
across all six target sets: mean 17.52 Delta E76 and 7.87% negative XYZ.

The correct conclusion is not “add a larger LUT.” On the unchanged five-grid
pool, higher capacity cannot distinguish real transferable structure from
grid-specific interpolation/extrapolation. AQ3 is closed. A read-only
source-support diagnostic may characterize the data gap, and a new controlled
source could reopen a separately frozen branch.

## Claim boundary

This is negative identifiability evidence for an internal manufactured-target
Velvia 100F recorder proxy. It is not a negative result about Velvia itself,
real camera-to-film mapping, the Emulating Emulsion paired-capture method or
the existing AO6 display-look approximation.
