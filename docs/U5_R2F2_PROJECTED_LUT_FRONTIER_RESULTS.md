# U5.R2F2 projected-LUT safety frontier results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2F2`  
Decision: **closed — no automatic survivor**

## Scope and integrity

This preregistered development experiment reused only the immutable U5.R2F1
CanonCGT-predicted LUTs. It did not run model inference, fit an image, train a
model, select references, or change any frontier gate.

- config SHA-256:
  `7204d5e9998b2d3bfd671c0681c97e87c862d63d1a350d2338b4692a8adc775f`
- software commit:
  `94ad7a0215394633c381c266141d3cc6ead0f62d`
- two-pass manifest SHA-256:
  `fbee9cbae18763751bb60ae24de1e35763b98b80f402f136cb20d4282b0da776`
- automatic report SHA-256:
  `d20e9592ad7e84450b5acea393013e7354d0ea42016eb5fcefd3123188bd3a7d`
- each pass: 27 policies, 243 PNG renders and 486 projected LUT arrays
- both stderr logs are empty

## Frozen automatic result

| Policy | Structure | Style | Non-basic | Clipping | Raw range | Survivors |
|---|---:|---:|---:|---:|---:|---:|
| node clip only | 0/9 | 6/9 | 6/9 | 1/9 | 1/9 | 0/9 |
| safe contraction, cap 0.75 | 9/9 | 0/9 | 0/9 | 9/9 | 9/9 | 0/9 |
| safe contraction, cap 1.00 | 9/9 | 0/9 | 0/9 | 9/9 | 9/9 | 0/9 |

Node clipping retained large transforms but did not make any of the nine
reference-conditioned policies structurally safe. Its worst new hard clipping
was 7.6581%, and interpolation still produced a maximum raw excursion of
0.0190%.

Both identity-contraction policies made every predicted LUT structurally safe
and reduced raw output excursions to zero. However, neither retained an
automatic style survivor:

- cap 0.75: maximum median style Delta E76 6.5178 and maximum non-basic
  residual 4.1835;
- cap 1.00: maximum median style Delta E76 6.5257 and maximum non-basic
  residual 4.1815;
- the frozen floors were respectively 7.0 and 4.9;
- the median applied contraction coefficient was 0.6861, and at least one LUT
  in every-bank evaluation required a coefficient as low as 0.2682.

The shortlist is empty. Under the frozen protocol no blind contact sheet or
full-resolution visual promotion review is permitted.

## Interpretation

U5.R2F1 already established that reference-conditioned ML LUT prediction is
not inherently bland: six of nine raw policies exceeded both style floors.
U5.R2F2 shows that simple node clipping or uniform identity contraction cannot
turn those raw policies into a Style-safe product candidate. The negative
result is specific to these projection families; it is not evidence that all
ML parameter prediction is hopeless.

The next scientifically distinct question is whether a bounded explicit
operator can be **refit on a fixed synthetic colour grid** to approximate the
immutable predicted transform while redistributing its colour action, instead
of uniformly weakening it. Such a leaf remains a generic external-control
experiment. It cannot use current real-film pixels, claim a stock response, or
open production integration.

## Claim ceiling and branch decision

The raw and uniformly projected CanonCGT routes close as current product
challengers. No candidate, stock, calibration, latent-mode, preference, or
production claim opens. A separately frozen constrained-explicit-distillation
leaf may proceed; increasing CanonCGT capacity, training, tuning the reference
bank, or changing the completed F1/F2 gates is forbidden.

Ultimate remains active.
