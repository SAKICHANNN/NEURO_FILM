# U1.4F DNG to ACES 2 photographic smoke contract

## Question

Does the retained U1.4E adapter execute deterministically on the exact four
rights-cleared DNGs whose capture-time metadata receipts passed U1.3C?

## Frozen rows and execution

- Reuse all four mandatory U1.3C byte identities without replacement:
  DJI FC4382, Google Pixel 7 Pro, Apple iPhone 12 Pro and Huawei EML-L29.
- Verify every source byte count and SHA-256 before pixel decode.
- Decode through the unchanged generic LibRaw/rawpy scene-linear sRGB path with
  camera white balance and no automatic brightening.
- Apply the unchanged U1.4E adapter to both frozen targets, SDR Rec.709 and HDR
  Rec.2020-PQ. Process one row at a time; retain only hashes and bounded scalar
  diagnostics, not decoded/output image payloads.
- Run all rows in canonical and reverse order in two fresh processes.

## Gates

All four source hashes and decoded shapes must agree across runs; outputs must
be finite float32 with the decoded shape; inputs must remain byte-identical in
memory; SDR and HDR output hashes must differ for every row; canonical/reverse
and both fresh-process normalized reports must be byte exact. Any decode or
gate failure closes this exact four-device smoke without row replacement,
decoder tuning, exposure rescue, clipping or output encoding.

## Claim ceiling

A pass establishes only private Windows CPU photographic runtime mechanics for
the exact four DNGs under the existing generic LibRaw decode and official ACES
2 numerical outputs. It is not vendor-render parity, a calibrated camera IDT,
photographic-quality evidence, HDR file/metadata support, arbitrary-DNG
support, default-renderer integration, public capability, film/stock fidelity
or product admission.
