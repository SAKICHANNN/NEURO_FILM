# U6.2A Boolean/Poisson Grain Representation Results

## Decision

**Pass the synthetic representation gate; open only a separately frozen
existing-image crop visual frontier.**

The clean-room grain-wise renderer preserves homogeneous means, produces the
expected midtone variance peak, exposes stronger local correlation for larger
grains, and is bit-exact across repeat, serialization and arbitrary row
partitions.

This is not measured film-grain calibration. U6.2 remains open and still
requires rights-cleared real NPS/autocorrelation/repeat-scan evidence before
any stock, process or scanner claim.

## Reproducibility

- implementation commit:
  `efd0a62f78cf122864de8e17c45b3046441044a1`;
- config SHA-256:
  `9b13547fa48a01f7c992401400850fea708df9bb91fd64b73a76531ac9417da6`;
- two formal report SHA-256 values:
  `50d3ba3ec682c561a59496052e8a8cff291838f867d1b4dbb370e5bd454bb46d`;
- both formal runs pass and are byte-identical;
- 26 focused/adjacent tests pass;
- complete CPU suite: `878 passed`.

The ignored formal reports remain under
`outputs/u6_2a_boolean_grain_representation_v1/`.

## Frozen results

| Radius | Level | Grain count | Interior mean | Interior variance | lag-1 correlation |
|---|---:|---:|---:|---:|---:|
| small .22 | .1 | 425 | .11204 | .03295 | .65166 |
| small .22 | .5 | 2,647 | .50444 | .07077 | .63000 |
| small .22 | .9 | 8,754 | .90113 | .01667 | .57267 |
| large .38 | .1 | 142 | .11328 | .05560 | .79194 |
| large .38 | .5 | 897 | .49477 | .12487 | .77162 |
| large .38 | .9 | 2,960 | .89883 | .03263 | .71714 |

- maximum flat mean error: `.01328` versus `.04`;
- small/large midtone variance ratios: `2.148/2.246` versus `1.4`;
- large-minus-small midtone lag-1 correlation: `.14162` versus `.03`;
- all outputs stay in `[0,1]`;
- repeat, partition and serialization maximum errors: exactly `0`.

## Interpretation

The Poisson intensity equation successfully preserves mean coverage without a
post-hoc mean normalization. Variance peaks near midtone rather than behaving
like constant-amplitude additive noise. Larger disks create visibly stronger
local spatial correlation while requiring fewer grains for the same expected
coverage. Those are the intended structural signatures of the Boolean model.

The absolute variance and correlation values are properties of these synthetic
settings, Monte Carlo count, zoom and filter offsets. They are not measurements
of any film stock. The implementation is original clean-room code based only
on the published equations; the GPL reference implementation was not used.

## Binding branch

- retain the isolated primitive;
- freeze one existing-image crop visual/style/severe frontier using the exact
  `.22/.38` policies without fitting;
- do not integrate production renderer, profile, recipe, CLI or defaults;
- do not call the result calibrated, stock-specific, NPS-matched or universally
  realistic;
- any severe visual artifact closes the candidate regardless of its physical
  motivation.
