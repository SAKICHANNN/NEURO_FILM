# P271 — R1EN DNG ProfileToneCurve callable intake

Status: **frozen before fixture deserialization or callable import**  
Date: 2026-08-27  
Parent: mature RAW/DNG explicit mechanisms; P261 refusal guard

## Question

Can this repository independently verify and execute the producer's exact,
versioned R1EN `ProfileToneCurve` callable from Git objects without copying the
implementation into consumer `src` or changing the default DNG loader?

## Frozen artifacts and domain

`configs/p271_r1en_dng_profile_tone_curve_callable_intake_v1.json` binds the
producer commits, Git blobs, byte counts and available SHA-256 identities.  The
callable accepts finite linear ProPhoto/ROMM values and a strict v1 payload,
then returns an owned, contiguous, writable float64 array.  Its frozen DNG stage
position is after `ProfileHueSatMap`, `ProfileGainTableMap`/Map2, exposure and
`ProfileLookTable`.  Samples outside the curve x domain use endpoint clamping;
coordinates outside `[0,1]` reject; nonmonotone y remains valid.

## Gates

1. Every producer Git object and declared byte/hash identity is exact.
2. The payload validates under the bound Draft 2020-12 schema.
3. Fixture input, payload and output identities are exact.
4. The isolated callable matches its frozen arithmetic core for scalar, array,
   in-domain and endpoint-clamped values.
5. Caller data is immutable and output ownership/dtype/layout are exact.
6. Wrong schema/fields/types, nonfinite input, coordinate range/count and
   nonincreasing-x controls reject without output.
7. Nonmonotone-y remains accepted.
8. Two fresh forward/reverse reports are byte-exact with zero network, new RAW
   or DNG pixel reads, producer-worktree imports and temporary residue.

## Stop rule and claim ceiling

Any identity, schema, arithmetic, ownership, failure, replay or cleanup mismatch
closes P271 without copying producer code, tolerance changes, loader integration
or cohort reruns.  A pass proves only private mechanical consumability of the
already-frozen R1EN arithmetic.  It is not a real-file stage composition, full
DNG renderer, image-quality result, package/schema/capability/product mapping,
or candidate-3 admission.
