# RF2.C0 spektrafilm external spectral-control results

Date: 2026-07-17

Node: `ULT > RF2.C > RF2.C0`

Decision: **retain one isolated external comparison control; do not integrate,
train from it or treat it as stock truth**

## Result

The experiment answers one narrow question positively: a deterministic,
physically informed external spectral simulator can produce a visibly
stylised, non-basic and artifact-clean colour response on the provisional
nine-image gold set. The one retained control is
`kodak_ektar_100__negative_to_print__fixed_e0`:

| Metric | Result | Frozen gate |
|---|---:|---:|
| gold median style Delta E76 | 8.0337 | >= 7.0 |
| gold median matched-basic residual Delta E76 | 7.3435 | >= 4.9 |
| worst new hard clipping | 0.0000% | <= 0.5% |
| blind rounds / samples per round | 3 / 9 | 3 / 9 |
| full-resolution external outputs reviewed | 27 | all selected candidates |
| confirmed severe artifacts | 0 | 0 |

This does **not** show that the response is a measured Ektar response. It is a
Look Approximation produced by external spectral profiles derived from data
sheets/papers plus heuristic, partly hand-modelled coupler parameters.

## Runtime and reproducibility

The external repository stayed in an ignored temporary checkout at revision
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`. No external code, profile or LUT
was copied into tracked project files. The isolated headless runtime used
official CPython 3.13.14 and spektrafilm 0.3.4 without GPU or paid resources.
GUI-only dependencies were intentionally absent; core import and rendering
passed.

Six profiles, two exposure policies and nine inputs produced 108 PNGs totaling
91,621,829 bytes. Two formal runs produced identical output hashes, metrics
and branch decisions. After removing wall-clock fields, both reports are
byte-identical at SHA-256
`a11901b127ae19276df28cd3bf9d81dbc58b24e300d775a831af8583c9cb34c6`.
The raw report hashes differ only because elapsed times are recorded.

The ignored evidence includes exact input/output hashes, environment and
licence hashes, the pre-reveal blind review, mappings and the full-resolution
adjudication. The tracked decision is
`configs/real_film_spektrafilm_external_control_decision_v1.json`.

## Automatic screen

Nine of twelve profile/policy candidates passed the frozen automatic screen.
Velvia100 fixed and auto were strongly stylised but produced 7.11% and 7.30%
worst new hard clipping, so both were rejected before visual review. Provia100F
fixed was below the style floor. The other nine were finite, bounded and above
the style/non-basic gates.

This high survivor count did not establish nine useful film experts. It made
the frozen exposure-policy control essential.

## Blind and full-resolution visual audit

Before opening the mapping, three independently shuffled blind rounds compared
input, safe-rich, all five owner anchors and three external candidates across
all nine gold samples. The anonymous notes consistently found two strong-dark
directions and one warmer medium-strength direction, with no confirmed
geometry, face, text, banding, colour-block, red-speckle or posterization
failure. The mapping then showed:

- the darkest direction was Pro400H/auto;
- the next was UltraMax400/auto;
- the medium warm/dense direction was Ektar100/fixed.

All 27 selected external outputs were then inspected at their 1024-pixel Phase
A resolution. Ektar/fixed preserved faces, fine texture and geometry and did
not reproduce the known ID11 red-speckle/posterization regression. It has a
clear warm/dense response distinct from the bright 53/55/56 family.

## Why the strongest automatic scores do not win

The two selected auto-exposure candidates are visibly strong and have no
confirmed severe artifact, but their appearance is not stable enough to count
as stock distinction evidence:

| Candidate | Per-scene mean-luma ratio range | Interpretation |
|---|---:|---|
| Pro400H/auto | 0.205--4.208 | auto exposure dominates |
| UltraMax400/auto | 0.244--4.171 | auto exposure dominates |
| Ektar100/fixed | 0.833--1.020 | bounded, stable comparison control |

Pro400H/auto and UltraMax400/auto render most inputs dark but push ID11 into an
extremely bright peach/red state. Portra400/auto and UltraMax400/auto are also
only 1.1784 median Delta E76 apart in the diagnostic profile comparison. These
results are retained as negative evidence that an image-aware exposure policy
can manufacture apparent style strength while averaging away the meaningful
profile distinction the user actually wants.

## Binding boundary

RF2.C0 closes with exactly one promoted **external future comparison control**.
It does not reopen RF2.S0, repair community-data identifiability, authorize
operator fitting or provide training targets. No output may be used as a
teacher. No stock, mode, calibrated-response, authenticity, product integration
or root-licence decision follows from this result.

The useful architectural evidence is narrower: explicit spectral priors can
create non-basic, stylised colour without spatial corruption, but adaptive
exposure must be isolated and constrained, and the external profile source is
not evidence enough for a named-stock truth claim.
