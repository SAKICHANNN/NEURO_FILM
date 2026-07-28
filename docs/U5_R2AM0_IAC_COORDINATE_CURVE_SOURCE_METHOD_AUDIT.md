# U5.R2AM0 IAC Coordinate-Curve Source and Method Audit

Date: 2026-07-28  
Node: `ULT > U5 > U5.R2 > U5.R2AM0`  
Decision: **retain a stricter clean-room coordinate-curve representation;
do not execute or reproduce IAC**

## Question

After AL1 showed useful colour-selective capacity but failed its iterative
inverse, can an image-adaptive-coordinate paper supply a genuinely distinct
explicit representation prior with an analytic inverse?

This audit does not ask whether IAC learns film. It does not: its reported
tasks are paired digital photo retouching, exposure correction and
white-balance editing.

## Primary-source freeze

| Source | Exact evidence |
|---|---|
| [Official BMVC paper](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_307/paper.pdf) | 4,558,846 bytes; SHA-256 `06ae7bf3babb89e9ca9ed95ddc165983ed309c23831d15300dc243bfa0796bbe` |
| [Official supplement](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_307/supplementary307.pdf) | 1,474,370 bytes; SHA-256 `117092eb12547508cd22bb46a86c4497ae28001fb1d6100bb381a34d11f23fdc` |
| [BMVC proceedings record](https://bmvc2024.org/proceedings/307/) | paper and supplement links; no code link |
| [arXiv record](https://arxiv.org/abs/2501.06448) | v1 dated 2025-01-11; paper source under CC BY 4.0 |

The official proceedings, arXiv record and authors' publication page expose
the paper/supplement but no author-linked code or checkpoint. Exact GitHub
repository searches for the title and arXiv identifier returned zero hits on
the audit date. This means **no reproducible package was located**, not that
code cannot exist elsewhere. No repository, checkpoint or dataset was
downloaded.

## What the paper actually defines

IAC predicts, from each input image:

1. three linearly independent RGB projection vectors forming a `3x3` matrix;
2. three 200-sample curves;
3. a forward operation that projects RGB through the matrix, normalizes each
   projected coordinate to `[0,1]`, applies its curve, denormalizes, then
   multiplies by the matrix inverse.

Its approximately 39.7K-parameter ConvNeXt-style predictor is trained on:

- 4,500 MIT-Adobe FiveK input/expert-C pairs with 500 evaluation images and a
  smooth-L1 plus VGG objective;
- the paired multi-exposure correction dataset;
- the Rendered WB digital white-balance dataset.

The paper reports useful efficiency and paired-task quality. These are valid
photographic-enhancement results, but they neither identify a reusable
reference operator nor provide film-stock evidence.

## Why direct IAC is outside the renderer contract

The paper does not report:

- monotonicity or positive-increment constraints on its three curves;
- determinant, condition-number or orientation bounds for its matrix;
- an analytic inverse of the **whole** curve operator;
- cube, Jacobian, clipping, severe-artifact or strength-path gates.

Its notation applies the inverse matrix after the curves; this is not the
inverse of the complete transform unless the curves themselves are invertible
and explicitly inverted. When matrix rank falls below three, the method adds
small random numbers. That repair is non-deterministic and supplies no
condition-number guarantee. The paper says projected coordinates are
normalized to `[0,1]` but does not specify a complete, replayable
image-independent extrema contract. If extrema are image-derived, content and
exposure statistics control the operator; AM1 therefore forbids that ambiguity
and derives them from the complete RGB cube.

Therefore the paper network, random repair, learned parameters and published
task claims do not enter the project.

## Surviving clean-room hypothesis

One narrower representation is both distinct from AL1 and compatible with the
explicit-renderer boundary:

```text
x in RGB
  -> t = x Q                         Q in SO(3)
  -> u = (t - cube_projection_min) / cube_projection_range
  -> v_i = monotone_curve_i(u_i)
  -> t' = cube_projection_min + range * v
  -> y = t' Q^T
```

The projection minima and ranges are derived from all eight RGB-cube corners,
not from the current image. `Q` is a bounded deterministic rotation with
determinant `+1` and condition number exactly one. Each curve has strictly
positive increments and an explicit inverse curve.

This yields a closed-form whole-operator inverse:

```text
y -> t' = y Q -> v -> inverse_curve(v) -> t -> x = t Q^T
```

The conjugated Jacobian has positive determinant when all curve derivatives
are positive. It does **not** automatically keep every transformed point
inside the RGB cube, so clipping is forbidden and an exact cube audit remains
mandatory.

This operator can later be conditioned only by predicting bounded rotation and
curve parameters. The first test must be a fixed synthetic representation
pilot; no predictor or photograph is needed.

## AM1 gate

Open one clean-room synthetic pilot with:

- one bounded `SO(3)` coordinate frame and three endpoint-fixed monotone
  curves;
- identity, RGB global-curve, bounded positive-matrix-plus-curve and
  stationary-K3 controls;
- frozen development/confirmation points and two independently reconstructed
  reports;
- absolute/relative fidelity, exact replay, cube, positive/bounded Jacobian,
  analytic inverse, neutral, hue-boundary, red-ramp and strength-path gates.

Any failed gate closes the fixed family without angle, curve, degree, target,
optimizer, range, threshold, clamp or photograph rescue. A pass is synthetic
representation evidence only. Later image-dependent parameter prediction
would still require separate data, shortcut, OOD and severe-artifact gates.

## Claim ceiling

`primary-source method audit and clean-room synthetic representation
hypothesis only`.

There is no reproduced IAC result, photograph, film-pixel fitting, unpaired
operator identification, stock, calibration, preference, severe-safety or
production claim.
