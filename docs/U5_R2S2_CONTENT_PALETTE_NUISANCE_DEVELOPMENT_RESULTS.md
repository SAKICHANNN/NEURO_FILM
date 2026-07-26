# U5.R2S2 Content-Palette Nuisance Development Results

Date: 2026-07-26

Decision: **no residual method beats the content-confounded control; do not
open confirmation**

## Reproducibility

- software commit:
  `2a6fe625a907459c9ea2eb036d720dab3d60feb0`;
- config SHA-256:
  `d1a3eb804d038fb9665bf6acbf55fc34b5e2bce943ac46c7d376887c1d815adf`;
- two complete reports are byte-identical at
  `fe61d784ae7b1b87af220837d7b2b24ed47e409ec63611f6fc76d9e746483798`;
- reserved confirmation seed `28105` was not accessed;
- ten focused S1/S2 tests pass.

## Results

| Method | Median oracle error | Same-style replicate error | Identity-style negative | Max Jacobian norm |
|---|---:|---:|---:|---:|
| raw styled KDE `.12` | .06393 | .06081 | .10494 | 8.228 |
| ridge `10` | .07255 | .07789 | .09131 | 10.702 |
| ridge `1` | .07413 | .08088 | .09243 | 10.872 |
| ridge `.1` | .07432 | .08124 | .09251 | 10.890 |
| density ratio `.25` | .10534 | .09414 | 0 | 6.343 |
| density ratio `.50` | .10697 | .10999 | 0 | 8.595 |
| density ratio `1.0` | .10915 | .12032 | 0 | 10.599 |
| global mean | .11002 | 0 | .07526 | 1.577 |

The automatic development ranking puts raw styled KDE first because it has
the lowest oracle error. That is not a promotable result: this method was
preregistered as the content-confounded control.

## Why raw KDE cannot advance

When styled and neutral inputs are made identical to represent an identity
style, raw KDE still applies a strong `.10494` output RMSE. It is responding
to the content palette itself. Its sampled maximum Jacobian norm is `8.228`,
also above the inherited project cap of `8`.

This is the precise shortcut S2 was designed to reveal. A low synthetic oracle
error does not convert a content-sensitive observation into a style
identifier.

## Why the ML candidate cannot advance

The best bounded ridge is alpha `10`, but:

- its median error is `13.48%` worse than raw KDE;
- its same-style error across independent content groups is `.07789`;
- its identity-style negative remains strong at `.09131`;
- its maximum sampled norm is `10.702`.

Increasing capacity on the same observation design would attempt to learn
around an identifiability failure. This leaf therefore closes without a
neural-network rescue.

## What density ratio solves and does not solve

All density-ratio candidates return exact identity when styled and neutral
histograms are identical. This is a useful semantic property. However, the
best scale `.25` reaches median operator error `.10534`, only `4.25%` better
than the global mean and `64.77%` worse than raw KDE. The unpaired aggregate
score difference does not recover the true transport direction.

## Next distinct question

The next legal algorithm is not a bigger regressor. It is a direct
distribution-matching fit of the already-safe explicit flow:

```text
independent neutral samples
  -> bounded explicit flow
  -> match independent styled distribution
  -> separately score distribution fit and oracle-operator recovery
```

MMD or fixed-projection sliced Wasserstein can provide the optimization loss.
The experiment must expose the central identifiability risk: a method may
match the target distribution while recovering the wrong operator. It remains
synthetic and cannot open project images or film pixels.
