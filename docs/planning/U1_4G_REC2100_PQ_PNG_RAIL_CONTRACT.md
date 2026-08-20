# U1.4G deterministic Rec.2100 PQ RGB16 PNG rail contract

## Standard boundary

PNG Third Edition defines `cICP` and gives full-range BT.2100 PQ RGB as the
four bytes `09 10 00 01` (hex): BT.2020 primaries, PQ transfer, identity RGB
matrix and full range. The chunk must precede IDAT. U1.4G implements that exact
container signal for already encoded PQ samples.

## Frozen implementation

- Add the exact constant `REC2100_PQ_CICP = bytes((9, 16, 0, 1))`.
- Extend the existing deterministic streaming RGB PNG rail with a private
  16-bit Rec.2100-PQ writer and strict sample verifier.
- Accept only contiguous non-empty HxWx3 `uint16` RGB samples, preserve samples
  exactly, emit only IHDR/cICP/IDAT/IEND, and publish atomically.
- Keep the existing Rec.2020 SDR rail byte-compatible.
- Do not invent `mDCV`, `cLLI`, ICC, mastering-display or content-light facts.

## Frozen gates

For a fixed RGB16 boundary/random fixture: exact native sample hash on strict
readback; exact `09 10 00 01` cICP placement and CRC; canonical/reversed row
publication equality; two fresh process reports byte exact; existing SDR PNG
goldens unchanged; malformed dtype/shape/order/incomplete writes fail
atomically without a published output.

## Claim ceiling

A pass establishes only a private deterministic PNG Third Edition container
rail for already PQ-encoded full-range BT.2100 RGB16 samples. It does not prove
the samples are correctly tone-mapped, constitute HDR10, carry mastering or
content-light metadata, render correctly on a display, interoperate across
external decoders/platforms, integrate the default renderer, or establish a
public capability or product.
