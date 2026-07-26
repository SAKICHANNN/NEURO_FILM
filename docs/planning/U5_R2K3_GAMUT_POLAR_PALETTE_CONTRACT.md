# U5.R2K3 — Gamut-Polar Palette Operator Contract

**Status:** frozen before implementation or result inspection
**DRPT level:** L2
**Parent:** U5.R2K2 structural pass / confirmation and norm failure
**Primary writer:** current Codex Goal session

## Question

Can a film-colour-specific factorization express strong tone, exposure-layered
hue trajectories and hue-specific chroma while making the RGB gamut boundary,
neutral axis, monotonicity and inverse structural instead of relying on clamp
or a generic high-capacity field?

K3 is a synthetic representation/diversity audit. It uses no image fitting,
film pixels, learned latent or neural network. A pass opens only a separately
frozen real-image frontier.

## Research basis and clean-room boundary

The official ACES 2.0 technical design for invertible gamut compression
separates lightness, colorfulness and hue, reasons within a constant-hue gamut
slice, and uses an invertible monotonic compression rather than output
clipping. K3 adopts those broad design principles because they directly address
the project's high-chroma artifact failures.

K3 is not an ACES implementation. It does not use Hellwig JMh, ACES matrices,
the ACES boundary approximation, powerP constants, code or parameters. Its
coordinate system and maps below are original, simpler clean-room constructions
on the bounded linear-sRGB cube. No ACES conformance or perceptual-uniformity
claim is allowed.

## Exact coordinate factorization

For positive Rec.709 luma weights `w`, every RGB point is decomposed as:

```text
Y = dot(w, RGB)
d = RGB - Y * [1,1,1]
```

`d` lies in the two-dimensional plane orthogonal to `w`. A deterministic
orthonormal basis maps `d` to polar hue `h` and radius `r`. For every `(Y,h)`,
the exact ray intersection with the RGB cube gives `r_max(Y,h)`, and normalized
chroma is:

```text
s = r / r_max(Y,h),  0 <= s <= 1
```

The operator maps `(Y,h,s)` with three finite explicit components:

1. `Y' = Y + t*Y*(1-Y)`, with `|t|<1`;
2. a unit-circle Möbius hue diffeomorphism whose complex parameter and rotation
   interpolate between frozen shadow/highlight values;
3. `s' = s + k(Y,h)*s*(1-s)`, with `|k|<1`.

Tone and chroma maps have stable quadratic inverses. The Möbius map has a
closed-form complex inverse. Reconstruction uses the exact destination
`r_max(Y',h')`, so the full cube maps onto itself without clamp. Endpoints and
neutral axis stay exact; finite parameters preserve positive scalar
derivatives.

## Frozen witness bank

Five witnesses are fixed in config:

- identity;
- cyan-shadow/warm-highlight-like;
- warm-dense-like;
- cool-soft-like;
- cross-palette-like.

The non-identity witnesses vary tone bend, shadow/highlight hue maps and a
six-term bounded chroma conditioner. Parameters are descriptive synthetic
choices, not stock labels or measured responses.

## DoR

- K0/K1/K2 results and gates remain immutable.
- K3 changes the coordinate factorization rather than increasing a failed
  candidate's capacity.
- The ACES principle/source and clean-room exclusion are recorded.
- Config and contract are committed/pushed before code or output inspection.

## Frozen audit and gates

Use a `25^3` full cube grid and separate `9^3` interior Jacobian grid.

- identity maximum error at most `1e-12`;
- neutral-axis chroma and black/white endpoint errors at most `1e-12`;
- every output in `[0,1]` with no clamp;
- analytic inverse roundtrip maximum error at most `1e-10`;
- every sampled Jacobian determinant strictly positive;
- maximum sampled Jacobian spectral norm at most `8`;
- minimum pairwise witness RGB RMSE `.015`;
- every non-identity witness at least `.03` RGB RMSE from identity;
- every non-identity witness leaves at least `.008` RMSE after its best affine
  RGB fit;
- exact serialization replay and partition parity;
- two reports byte-identical.

## Branches

- **All gates pass:** retain K3 as a numerical palette representation and
  freeze one bounded-strength real-image frontier with inherited
  style/non-basic/clipping/severe gates.
- **Diversity fails only:** close the fixed witness bank; do not amplify it on
  the same grid.
- **Range/inverse/orientation/norm fails:** close the formulation; do not add
  clipping, projection or post-hoc smoothing.
- **Later real-image style fails:** close without parameter tuning.
- **Later severe artifact appears:** veto regardless of style.

## DoD

- formula, boundary, inverse, property, replay and determinism tests;
- two byte-identical formal reports;
- result and binding decision records;
- full CPU suite;
- tracker/log propagation;
- scoped commits and pushes.

## Claim ceiling

Clean-room, data-independent numerical, diversity and regularity evidence for
an analytic gamut-polar palette Look Approximation that factorizes tone,
luma-conditioned hue and hue-conditioned chroma. No ACES conformance,
perceptual uniformity, measured film response, named stock, calibration,
authenticity, preference or production claim.
