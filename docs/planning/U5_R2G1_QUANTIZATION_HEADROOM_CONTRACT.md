# U5.R2G1 fixed quantization-headroom contract

Date frozen: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2G1`  
Status: **frozen before any G1 render**

## Question

Does one data-independent RGB8 quantization-headroom map remove the exact
boundary failure of immutable G0 operators while retaining their strong style?

G0 is closed and may not be refitted. G1 isolates serialization safety; it is
not an added learned model or a new colour-fit family.

## Immutable input

Use all 81 serialized operators from the first byte-verified G0 pass. Verify
the G0 config, manifest, report, operator, source-image and lineage hashes
before rendering. No G0 PNG is used as a processing input.

## Sole policy

For the raw continuous G0 result `y`, apply:

`y_safe = h + (1 - 2h) y`, where `h = 1/255`.

The value is derived from the target RGB8 quantization lattice: exact
continuous endpoints map to integer codes 1 and 254, never 0 or 255. There is
no strength grid, alternative headroom, fitted value, image statistic,
conditional branch, or per-channel variant.

The map must be incorporated into the serialized explicit operator record and
replayed from source RGB. Processing an already quantized G0 PNG is forbidden.

## Integrity and structural gates

- exact 9 references x 9 gold inputs;
- two complete passes;
- byte-identical manifest, transformed operators and PNGs;
- finite raw outputs entirely in `[1/255, 254/255]`;
- RGB8 output contains no code 0 or 255 introduced by the operator;
- positive corresponding-channel steps and tetrahedral Jacobians;
- transformed-operator replay error at most `1e-12`;
- no source, geometry or output resampling;
- no effect layer or post-output adjustment.

## Automatic frontier

Retain the G0 absolute gates:

- median style Delta E76 at least `7.0`;
- median non-basic residual at least `4.9`;
- worst new hard clipping at most `0.5%`;
- raw out-of-range exactly zero;
- all nine operators structurally safe;
- reference sensitivity median pairwise Delta E76 at least `2.0`.

Additionally:

- at least one complete survivor;
- median style loss relative to matched G0 policies no worse than `0.5`
  Delta E76;
- median non-basic loss relative to matched G0 policies no worse than `0.5`
  Delta E76.

## Visual protocol

Only automatic survivors enter review. Select at most one per provenance
bucket and three total by non-basic residual then style. Run three
deterministic blind rounds against frozen anchor56 and the R2E1
cyan-shadow/warm-highlight challenger. Inspect every shortlisted candidate on
all nine full-resolution inputs, with ID11 explicitly checked for red speckle,
posterization, banding and colour blocks. Any severe failure rejects the
candidate.

## Branches

- no survivor or retention failure: close the CanonCGT distillation branch;
- automatic survivor but visual failure: reject and close;
- automatic and visual pass: retain as a B0 generic reference-conditioned
  explicit-ML challenger for later fixed-bank comparison.

No branch opens production integration. A retained challenger must later beat
the deterministic density and anchor56 comparators under the complete
FilmStyleSafe policy.

## Forbidden and claim ceiling

No refit, alternate headroom, capacity increase, model inference, photograph
fitting, real-film fitting, stock claim, calibration claim, LSM, population
preference claim, or production integration is allowed.

Maximum claim:

`generic external-ML reference condition distilled into a deterministic
bounded explicit RGB operator with fixed RGB8 serialization headroom`.

Ultimate remains active under every result.
