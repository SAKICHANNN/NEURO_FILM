# U1.4C2 Rec.2020 Real-Image Visual/OOD Results

**Status:** closed at automatic gate; visual stage not opened

**Parent:** `ULT > U1.4 > U1.4C > U1.4C1`

**Config SHA-256:** `9747f1ba93fdefe1b409a3a2197ffe77a0fe6e04469e4c7b8f1c967bb726b57b`

**Report SHA-256:** `3b52032ea01a023189c4e03e6071cec3ae2e7c18f9dad63c6bb3ae08a8b8a118`

**Harness commit:** `9699d7198f825a146c1e6cd59f6c78051b6ba5cf`

## Outcome

The audit executed the complete frozen automatic matrix twice: eight exact
CC-BY FILM-R expert restorations, six colour styles and source/chroma gamut
modes, for 96 full-resolution RGB16 Rec.2020 masters per pass. Source hashes,
bytes and decode passed. Both normalized manifests are identical.

The automatic gate nevertheless fails:

| Measure | Result | Frozen gate |
|---|---:|---:|
| maximum new Rec.2020 boundary fraction | `6.7808155%` | `<=0.5%` |
| renders above boundary gate | `66/96` | `0/96` |
| source mode above gate | `33/48` | no policy exception |
| chroma mode above gate | `33/48` | no policy exception |
| maximum diagnostic sRGB clip fraction | `30.8934%` | reported only |
| maximum diagnostic linear-sRGB excursion | `0.312228` | reported only |
| median of per-render median style Delta E76 | `2.48699` | reported only |
| repeat normalized manifest identity | pass | required |

The full formal execution took 1,463.1 seconds. All 96 first-pass masters and
reports remain under the ignored output root for numerical reproducibility.

## Failure distribution

The failure is neither a single style nor a single gamut-policy accident.
Counts below combine both policies, 16 renders per style:

| Style | Over gate |
|---|---:|
| `ektar_100` | 12/16 |
| `portra_400` | 6/16 |
| `portra_800` | 16/16 |
| `velvia_50` | 10/16 |
| `vision3_250d` | 16/16 |
| `vision3_500t` | 6/16 |

All eight source groups have at least four failures. `provia100_135_1` peaks at
`6.7808%`; `superia400_135_1` peaks at `4.6530%`. The best source maximum,
`velvia50_half_2`, is still `0.5014%`, narrowly above the frozen gate.

Numerical channel attribution shows two different boundary mechanisms:

- `provia100_135_1/portra_800` sends about `6.78%` of pixels to the Rec.2020 red
  high boundary;
- `superia400_135_1/vision3_250d` sends about `4.40%` to the blue low boundary,
  plus smaller green-low support.

Source and chroma modes have the same worst fraction and the same `33/48` fail
count. Their encoded outputs are not identical, so this is not an accidental
duplicate-policy execution; both policies independently hit the gate.

## Visual epistemic boundary

Per the preregistered protocol, automatic failure prohibited visual candidate
generation and review. No anonymous sheet or private mapping was created. The
project therefore has **no C2 confirmed visual-severe verdict**. The correct
claim is an automatic boundary/clipping-risk failure, not observed banding or
posterization.

The user-visible sRGB preview statistics are diagnostic only and did not cause
the rejection. A valid wide-gamut image may lie outside sRGB; the rejecting
measure is new boundary occupancy inside the claimed Rec.2020 output gamut.

## Decision

Close U1.4C2. Do not integrate the C1C adapter into production renderer,
profiles, schemas or defaults, do not widen the 0.5% gate, and do not add model
capacity to rescue this branch. Retain:

- C1A conversion primitives;
- C1B pure Lab kernel and exact legacy sRGB parity;
- C1C as synthetic mathematical research evidence only;
- the C2 harness and negative, repeatable OOD evidence.

Ultimate remains active and returns to another evidence-authorized product or
research leaf. A future materially different gamut hypothesis would need a new
frozen contract and cannot rewrite C2.

## Claim ceiling

Repeatable automatic negative OOD evidence on eight rights-clear FILM-R
restoration scans under an explicit unprofiled-sRGB assumption. No confirmed
visual artifact, ordinary digital-photo safety, preference, film-stock
authenticity, calibration, HDR, ACES/OCIO or user-facing wide-gamut claim.
