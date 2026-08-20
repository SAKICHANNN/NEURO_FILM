# U5.R2REPID6 — CanonCGT fixed-atlas shared-LUT contract

Date: 2026-08-20  
Status: frozen before model inference

## Question

The closed U5.R2F1 branch let CanonCGT inspect every application source while
predicting both its canonicalization LUT and its reference-conditioned grading
LUT. That is useful as an external end-to-end control, but it does not satisfy
the single-reference product boundary: `BuildReferenceStyle` must finish before
an application source is opened and must emit one shared explicit operator.

This leaf asks a narrower and genuinely different question. The unchanged
official E2E checkpoint extracts a grade vector from one reference. Its
Restyler is evaluated only on three deterministic, model-owned RGB lattice
atlases. The three resulting 17-cube LUTs are aggregated in sorted atlas-ID
order using a float64 mean and one float32 cast. That single LUT is frozen and
then replayed unchanged over all nine consumed A0 application sources.

## Execution order

1. **Build:** verify the parent contract, official source/checkpoint and the
   nine reference identities. Do not stat, hash, decode or otherwise read an
   application source. Generate the three atlases procedurally, build one LUT
   per reference and atomically write a build manifest.
2. **Apply:** re-verify the frozen build manifest and LUT identities, then and
   only then verify/decode the nine application sources. Replay each reference
   LUT over every source with no parameter change.
3. **Evaluate:** verify every artifact and compute the frozen atlas-stability,
   reference-sensitivity, style, non-basic residual, clipping and raw-range
   gates. A pass opens blind review; it is not itself preference evidence.

The runner must receive an explicit output root. During the current P-volume
repair gate, formal ignored artifacts belong under the project-owned D fallback
directory and tracked files must not encode a physical drive path.

## Controls and gates

- RGB/GBR/BRG atlas enumeration is reversed in the second run, but aggregation
  always sorts by atlas ID; every frozen LUT byte must remain exact.
- References and application sources are also enumerated in reverse in the
  second run; the normalized scientific payload must be exact across two fresh
  processes.
- Atlas-conditioned LUT pairwise RMSE p95 must be at most `.08`.
- Reference-bank median pairwise output Delta E76 must be at least `2.0`.
- At least one reference must pass median style `>=7.0`, median matched-basic
  residual `>=4.9`, worst new clipping `<=.5%`, and worst raw final range
  escape `<=.5%`.
- All values must be finite and every source must reuse the same per-reference
  LUT SHA.

Any failure closes this exact atlas/checkpoint/aggregation/cohort without
projection, strength, atlas, threshold or same-cohort rescue.

## Claim boundary

The inputs and references are already-consumed A0 material. A pass can support
only a private blind comparison of a reference-only explicit operator. It
cannot establish film-stock identity, calibration, population preference,
CanonCGT SSL reproduction, public schema/capability or product admission.
