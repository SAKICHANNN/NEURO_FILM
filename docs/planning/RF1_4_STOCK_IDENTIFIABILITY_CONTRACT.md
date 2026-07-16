# RF1.4 stock identifiability contract

Date: 2026-07-16

Node: `ULT > RF1.4 > RF1.4A`

Status: frozen before feature extraction

## Falsifiable question

Do the bounded BlueNeg post-negation 8-bit previews contain a stock-label signal
that survives whole-physical-roll holdout and cannot be explained equally well
by date, location/content, border/crop, exposure or simple global colour
statistics?

This is not a test of physical negative density and not an operator-fitting
experiment. Only Gold has a separate display-proxy lane; that lane is reserved
for a later RF1.4 child after this shortcut audit.

## Evidence and sample unit

- Inputs are pinned by the v2 integrity decision/report and the autonomous
  8-sheet vision decision.
- The independent vote and permutation unit is the physical roll, never the
  frame.
- Official-test rolls remain sealed.
- Exact and dHash<=4 cross-frame leakage must remain zero.
- Each held-out roll is predicted from stock centroids formed by first averaging
  frames within training rolls and then averaging rolls, so large rolls do not
  dominate.

## Stage-zero structural gate

A stock can enter the primary feature comparison only with at least three
rolls, at least two frames in every roll, and at least two content cells having
two frames each. A comparable stock pair must share at least two such cells.

Stocks that fail remain in the descriptive support table but are closed before
feature interpretation. Capacity, GPU training and random frame splits are
forbidden fallbacks.

## Frozen descriptor basket

The primary descriptor is deterministic RGB distribution quantiles from the
center 80 percent crop. Required controls are:

1. full-frame and center-60-percent versions for border/date/crop sensitivity;
2. luminance-only quantiles;
3. per-channel-standardized RGB quantiles, which remove mean/scale colour cues;
4. RGB mean/standard deviation only, representing the simple global cast,
   contrast and saturation family that previously produced bland pseudo-style;
5. metadata/shortcut features from date, dimensions/aspect, partition,
   coarse content cell and measured border darkness.

All features are standardized from the training folds only. Classification is
nearest roll-balanced stock centroid. Score is roll-balanced accuracy plus
per-stock held-out-roll recall and the roll-level confusion matrix.

## Null, gates and interpretation

The primary score is compared with 999 deterministic count-preserving stock
label permutations at the roll level. A candidate requires all of:

- permutation p-value <= 0.05;
- primary accuracy at least 0.10 above RGB mean/standard-deviation only;
- primary accuracy at least 0.10 above the shortcut baseline;
- full versus center-80 absolute accuracy delta <= 0.10;
- center-60 versus center-80 absolute delta <= 0.15;
- every included stock has held-out-roll recall >= 0.50.

Passing only means archive/scanner/content-distribution-specific preview-label
evidence. It does not establish causal stock response and cannot promote a
colour expert. Failing any shortcut/null gate closes the preview mechanism
instead of authorizing more model capacity.

## Decision DAG

```text
frozen evidence
  -> structural support fails: close affected stock
  -> structural support passes
       -> permutation fails: unidentified, close
       -> shortcut ties/wins: nuisance-confounded, close
       -> crop unstable: border/date/crop shortcut, close
       -> all gates pass: retain preview-label evidence only
            -> later Gold-only display-proxy child may test an explicit operator
```

Machine-readable contract:
`configs/real_film_stock_identifiability_v1.json`.
