# U1.4C2 Rec.2020 Real-Image Visual/OOD Audit Contract

**Status:** frozen before implementation or current-output inspection

**Parent:** `ULT > U1.4 > U1.4C > U1.4C1`

**Config:** `configs/u1_4c2_rec2020_visual_ood_v1.json`

**Risk:** R1 local A0 evidence generation; no product integration

**Primary writer:** `dev-research-reliability`

## Question

Does the isolated six-style Rec.2020 safe-Lab adapter remain deterministic,
within gamut and free of confirmed severe colour artifacts on a bounded set of
real photographic images that is materially outside its synthetic C1C witness?

This audit cannot answer whether the looks are attractive, authentic to a named
stock, calibrated, or safe on ordinary digital photographs.

## Evidence and rights boundary

Use only the eight exact FILM-R v2 expert-restoration payloads frozen in the
config. FILM-R v2 is CC BY 4.0; attribution remains Daniela Ivanova and original
scan contributor d@analog.cafe. Its filename families are weak hints used only
to prevent one-family selection collapse. They are not stock labels.

Selection is metadata/hash deterministic and frozen before viewing any current
adapter output: minimum restoration SHA within each family, then eight families
ordered by SHA-256 of the family string. Earlier dataset contact sheets may have
been viewed, so every result is A0 autonomous development evidence, never a
sealed confirmatory result.

The scans have no ICC profile. Treating their encoded samples as display sRGB
is an explicit OOD assumption for stress testing, not a colourimetric fact.

## Render matrix

- 8 sources x 6 colour styles x 2 gamut modes = 96 masters;
- safe-rich colour parameters, but grain/dither/output margin fixed to zero;
- master output is deterministic RGB16 Rec.2020 SDR PNG with CICP;
- any sRGB image is a separately labeled clipped visual preview only;
- execute twice and require byte-identical masters and normalized manifests;
- retain outputs under the ignored audit root; commit only code/config/reports.

## Automatic gate

Before visual review, require exact source hash/bytes/decode, 96/96 finite
shape-identical Rec.2020 masters, two-pass byte identity, and no more than 0.5%
new Rec.2020 boundary pixels on any output. Re-run frozen legacy hashes and the
complete CPU suite. Failure closes or narrows the policy before visual review;
thresholds cannot be widened after results.

sRGB diagnostic excursion/clipping is reported but is not an artifact gate,
because a valid wide-gamut result may intentionally lie outside sRGB.

## Visual gate

Generate two deterministic anonymous contact layouts containing all 96 outputs
and eight sources. Candidate identities remain in an ignored private mapping
until both layouts are reviewed. Review only severe artifact categories:
posterization, banding, colour blocks, unstable highlights, large unintended
clipping, detail corruption and seams. Do not score style or preference.

Rank 12 masters for original-resolution adjudication using only preregistered
automatic risks: new Rec.2020 boundary fraction, clipped-preview fraction and
the high-percentile spatial gradient of the colour residual. Any severe or
uncertain contact-sheet observation also enters adjudication. A confirmed
severe artifact rejects that policy; uncertainty cannot open integration.

## Branches and claim ceiling

- automatic failure: stop visual promotion and preserve the negative result;
- policy-specific severe: reject that policy without adding capacity or
  weakening the gate;
- uncertain: remain A0 research-only;
- pass: retain bounded A0 OOD evidence and require another frozen integration-
  readiness leaf before touching renderer/profile/schema/default paths.

Maximum claim: A0 autonomous visual/OOD evidence for an isolated colour-only
Rec.2020 research adapter on eight rights-clear, previously available film-scan
images. No digital-photo safety, preference, stock authenticity, calibration,
HDR, ACES/OCIO, FilmFX, production renderer or user-facing wide-gamut claim.
