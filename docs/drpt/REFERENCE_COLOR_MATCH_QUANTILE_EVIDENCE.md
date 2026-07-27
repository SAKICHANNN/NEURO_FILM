# Reference Color Match Quantile Evidence

## Decision

`P20 = ROUTE CLOSED`.

The bounded affine plus monotone-quantile candidate is deterministic,
replayable, batch-invariant and structurally safe. It provides a small
stress-split improvement over Gaussian CFSM, but not the preregistered minimum
needed to justify real-photo review or product integration.

## Distinct hypothesis

P19 tested whether independent real-photo first/second moments fix the
canonical prior. P20 instead keeps the data-free uniform prior and expands the
operator objective beyond moments:

```text
uniform 17^3 canonical samples + uploaded reference
  -> symmetric affine transport
  -> eight fixed alternating marginal-quantile spline updates
  -> sample explicit operator on 17^3 cube
  -> boundary-pinned safe projection
  -> one fixed tetrahedral LUT for N sources
```

No external image, training, neural network, per-source fitting or direct RGB
generation is used.

## Frozen evaluation

- Development: first eight fit operators per family, two cross-content
  observations each.
- Confirmation: first eight previously unopened `stress` operators per
  family, using red-, blue- and green-omitted narrow palettes.
- Confirmation gates: at least +3 points median gain over uniform, at least
  75% improvements, no more than 2 points worst loss, no more than 5% new
  boundary pixels, all constraints and zero fallbacks.
- Development cannot modify any candidate parameter, split or gate.

## Results

| Split/candidate | Improved | Median captured style | Worst | New boundary |
|---|---:|---:|---:|---:|
| Development uniform | 43/48 | +9.70% | -6.35% | 0% |
| Development quantile | 44/48 | +9.49% | -6.56% | 0% |
| Stress confirmation uniform | 37/48 | +11.70% | -34.87% | 0% |
| Stress confirmation quantile | 38/48 | +12.96% | -36.02% | 0% |

Confirmation:

- improvement rate `0.7916666667` — pass;
- median gain `0.01257842644` — fail versus `0.03`;
- worst loss `0.01146880197` — pass versus `0.02`;
- constraints, identity fallback and boundary gates — pass;
- status `quantile-route-closed`.

Two confirmation reports are byte-identical:

- report ID
  `785d0d14531072bc8ecab192aa58bd3d943784189b506490a2f0efc7123ea478`;
- SHA-256
  `cf39f27cd18e72f4820141296998a7323f2563744dc1c04e3fbc0ff0383c1c99`.

## Propagation

1. A1 remains closed and default product delivery remains identity.
2. The small stress improvement is retained as mechanism evidence only.
3. Post-hoc iteration, knot, strength or threshold tuning is forbidden on
   these opened splits.
4. More expressive unpaired marginal fitting is not an eligible rescue under
   the same information regime.
5. The next viable branch must add user information or preregister a learned
   perceptual Look Approximation objective with separately cleared assets and
   a fresh evidence split.
