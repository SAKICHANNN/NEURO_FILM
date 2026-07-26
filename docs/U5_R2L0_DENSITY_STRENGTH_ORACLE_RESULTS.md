# U5.R2L0 density-strength Oracle results

Status: **complete — material Oracle gap, severe veto passes; only simplest
inference-policy research opens**

Node: `ULT > U5 > U5.R2 > U5.R2L0`

## Result

The hash-pinned evaluator reused only the immutable U5.R2E1
cyan-shadow/warm-highlight `s0.50` and `s0.65` renders. For each of the exact
41 A0 development images, the Oracle selected `s0.65` when its already-measured
new-hard-clipping fraction was at most `0.005`; otherwise it hard-fell back to
`s0.50`.

| Evidence | Result |
|---|---:|
| selected `s0.65` | 36 / 41 (87.80%) |
| fallback `s0.50` | 5 / 41 |
| mean style gain versus fixed `s0.50` | 3.0676 Delta E76 |
| median style gain | 3.5642 Delta E76 |
| gold mean style gain | 2.6733 Delta E76 |
| stress mean style gain | 3.1784 Delta E76 |
| worst selected new hard clipping | 0.4925% |

All preregistered automatic gates pass. Two independent formal invocations are
byte-identical:

- report SHA-256:
  `d5093264e3b647f0b13e9b99b66715a85a69be4659763ba602dc28d7aace90cc`;
- selected-manifest SHA-256:
  `66d5fbb8cabc100049909c606b25d39e92f0fe6a5487e15cd1a6ede32df7ea72`.

## Visual veto

The review covered every selected output through six source/fixed/selected
overview pages, five unscaled corner/centre crops per image, and 41
same-coordinate source/fixed/selected 1:1 triplet pages. No confirmed severe
artifact appears in 41/41 outputs.

- ID11 at `s0.65` does not reproduce the historical red-speckle or
  posterization failure.
- `FS_FACE_01` retains coherent eyes, skin, lips, hair, hand and geometry.
- Samples 14/15 contain strong chroma noise already present in the source and
  continuously amplified at both strengths. This remains a stress limitation,
  not a new regular speckle or repeated-texture failure.
- Sample 37 contains subtle inherited smooth-gradient/JPEG layering, mildly
  amplified at both strengths without a new discontinuity or severe band.

Visual evidence SHA-256 is
`740ad30fe6fbbf350bddac12e37158b2175f4a8b058d42b7000d957615db0b94`;
the full-resolution adjudication SHA-256 is
`30ae9b47e4ba0fc80ff317ea114ba00af17394e6a3c9574f435dd0a487385140`.

## Interpretation and branch

The fixed `s0.50` challenger is not the useful strength ceiling on this set.
There is a material per-image strength Oracle gap, so U5.R2L1 may test the
simplest inference-time hard policy.

This is not yet a deployable router. The Oracle looks at the challenger output
after rendering, and clipping is not a universal severe-artifact detector.
R2L1 must begin with deterministic explicit state and preserve hard fallback;
ML is allowed only later if simple policy reproduction fails and a
leakage-safe development/confirmation split can be frozen.

No E1 result is rewritten. No training, fitting, stock learning, calibration,
production integration or general-safety claim opens.

## Verification

- eight focused Oracle/review tests pass;
- full local CPU suite: **861 passed** in 60.56 seconds;
- project structure is preserved under `src/eval`, `scripts`, `tests`,
  `configs`, `docs/planning` and ignored `outputs`.

## Claim ceiling

B0 autonomous development evidence that the immutable density-cyan
film-inspired Look Approximation has a material clipping-limited strength
Oracle gap and no confirmed severe artifact on the frozen 41-image review. No
deployable routing, general safety, preference, digital-to-film
identification, named-stock response, calibration, authenticity or production
claim.
