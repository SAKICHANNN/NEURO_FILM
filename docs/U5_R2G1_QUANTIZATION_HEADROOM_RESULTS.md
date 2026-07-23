# U5.R2G1 fixed quantization-headroom results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2G1`  
Decision: **closed — no automatic survivor; CanonCGT distillation branch closed**

## Integrity

- config SHA-256:
  `58c28b2b35aa979dc3e86cd2dd9ea51573dd94b7c10fbf145b352790d7319699`
- software commit:
  `e175a9541e34c7baf4fbcde1de9e908105c9737d`
- two-pass manifest SHA-256:
  `6bd7131d0ba75a0f787f4a3e9c33225f3852071d12f7044474cffdcd2878ca98`
- automatic report SHA-256:
  `9de823ec7b2c7274ef1cc89163465baa2204d331733475c1737ee4cb18c0e581`
- each pass contains 81 transformed operator records and 81 direct source
  replays; manifests and outputs are byte-identical; stderr is empty

## Result

The fixed map succeeds at continuous structure and style retention:

- structure and raw-range gates: 9/9;
- style: 9/9;
- non-basic residual: 6/9;
- reference sensitivity: `9.9725`, pass;
- median style loss from G0: `0.1484`, pass;
- median non-basic loss from G0: `0.0564`, pass.

It fails the unchanged empirical clipping gate for every reference. The
policy deliberately maps endpoints to RGB8 codes 1 and 254, but the frozen
metric defines values within `1/255` of either boundary as hard clipping.
Candidate worst-case new-clipping fractions range from approximately zero to
`7.2818%`; the near-zero ref05 policy still fails non-basic residual and
therefore cannot survive.

There are no automatic survivors. The shortlist is empty and visual promotion
is forbidden.

## Branch decision

G1's contract prohibited trying another headroom after results. Therefore the
raw, uniform-projection, constrained-distillation and fixed-headroom CanonCGT
branches are now closed as current product challengers. The positive evidence
remains important: reference-conditioned ML predicted strongly distinct
explicit transforms, and low-dimensional bounded refitting retained their
style. The failure is the complete severe-first output contract, not
inevitable blandness.

Future learned-parameter architectures should reserve headroom strictly larger
than the evaluator epsilon by construction and preregister it before fitting.
That general engineering lesson does not authorize a G1 rescue.

## Claim ceiling

`generic external-ML reference condition distilled into a deterministic
bounded explicit RGB operator with fixed RGB8 serialization headroom`.

No stock, calibration, preference, latent-mode, production, or severe-safety
claim opens. Ultimate remains active and moves to an independent
film-specific algorithm/data leaf.
