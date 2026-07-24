# U2.2A monotone sensitometry primitive results

Date: 2026-07-24  
Decision: **numerical primitive pass**  
Parent: `U2.2`

## Outcome

The clean-room linear-exposure-to-layer-density primitive passes every frozen
gate. It provides an explicit log-exposure encoder, three neutral-anchored
strictly monotone characteristic curves, analytic inverse/Jacobian and exact
replay without modifying the production renderer or claiming a named stock.

Formal report:
`outputs/u2_2a_sensitometry_primitive/formal/report.json`, SHA-256
`d15021f60e415d8acfd5ed8a925efb30ec9cd7b0c45abf354eb9a944ff2f08e9`.

## Numerical evidence

- 16,384 generated probes plus exact zero, 0.18 neutral and 16.0 boundaries;
- linear RGB range `[0,16]` maps to relative log exposure
  `[-3.2555, 1.9486]`;
- forward/inverse maximum RGB error: `3.73e-14` against `1e-11`;
- exposure encoder roundtrip maximum error: `7.11e-15` against `1e-13`;
- neutral input `0.18` maps exactly to density `[1,1,1]`;
- minimum Jacobian determinant: `4.90e-7`, strictly positive;
- exact canonical JSON replay;
- negative linear input and inverse below the zero boundary both reject.

## Descriptive curve shape

| Layer | toe slope | mid slope | shoulder slope |
|---|---:|---:|---:|
| red | 0.04243 | 0.51064 | 0.27273 |
| green | 0.06334 | 0.49275 | 0.27600 |
| blue | 0.05818 | 0.42012 | 0.27945 |

All three layers have a lower toe and shoulder slope than midscale. The
minimum maximum pairwise density separation away from the shared anchor is
`0.07047`, above the frozen `0.05` diversity floor. Thus the witness is not
three copies of one curve.

## Meaning and limits

This closes the missing numerical representation of exposure-dependent
toe/midtone/shoulder behaviour. Unlike a generic RGB curve, it records:

- nonnegative linear input and its exact reference exposure;
- the finite zero-exposure mapping through a declared black offset;
- the `layer_density` output semantic;
- one shared neutral gauge;
- analytic invertibility and positive local orientation.

The curves are descriptive clean-room witnesses. They are not fitted Velvia,
Ektar, Portra or any other stock, and layer density is not display RGB. Dye
mixing, negative/slide interpretation, print response, gamut handling and
profile integration remain separate.

## Branch decision

Retain U2.2A as isolated research infrastructure. Open U2.2B only to audit a
non-duplicative explicit composition with an existing density-to-display or
negative-to-print interpretation. U2.2 is not complete and U2.3/product
integration does not open from this primitive alone.

## Reproducibility

- config SHA-256:
  `8f43edcbc7e168af830081dce16ecb5f268a294d2f0cd777b942aec4b95c8af2`;
- software commit: `c67f8e63d44539051cb269824610cb527d2aa05d`;
- exact runs: 2/2;
- full CPU suite: 798 passed in 66.00 seconds.

## Claim ceiling

Deterministic numerical validation of an uncalibrated monotone
linear-exposure-to-layer-density sensitometry primitive.

