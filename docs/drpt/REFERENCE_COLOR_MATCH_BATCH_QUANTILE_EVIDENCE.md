# Reference Color Match Batch-Quantile Evidence

## Decision

`P21 = ROUTE CLOSED`.

Using the complete N-source batch is a new information regime, but adding
monotone marginal quantiles does not improve the already tested Gaussian
source-batch candidate. No real-photo review or integration opens.

## Contract

- The reference and all N sources may be inspected during fit.
- Exactly one immutable candidate LUT is produced for the complete batch.
- Source ordering cannot alter candidate identity.
- The result cannot claim a reference-only recipe reusable on a future album.
- No external images, training, neural weights or direct RGB generation.
- Confirmation uses previously unused stress operators 8--15 before any
  result from those rows was inspected.

## Results

| Candidate | Improved | Median captured style | Worst | New boundary |
|---|---:|---:|---:|---:|
| Source-batch Gaussian | 40/48 | +9.32% | -16.51% | 0% |
| Source-batch quantile | 40/48 | +9.13% | -15.00% | 0% |

The quantile extension:

- passes absolute improvement rate, constraints, fallback, boundary and
  worst-loss gates;
- has median gain `-0.0019153471082759554` instead of the required `+0.03`;
- therefore closes without real-photo review.

Two runs are byte-identical:

- report ID
  `e5a2d6707df68d61d3078a1f482a8ae5745adcbbfee2b688d70eab523e36c14a`;
- SHA-256
  `c9d261ad73ea0fb4ea7d0603b69b4fbaf12df79b559bda1c2a61b9d4cc5105b1`.

## Propagation

The Gaussian source-batch control is stronger on this synthetic slice, but it
is not promoted. P13 already found only 16/30 real-matrix improvements,
`+0.571%` median and `-11.524%` worst. P21 cannot overwrite that independent
negative evidence or the reference-only recipe contract. It only proves that
more marginal flexibility is not the missing ingredient.
