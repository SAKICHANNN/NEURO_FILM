# P264 — Real-DNG AP0/PQ interchange contract

Status: frozen before implementation or output comparison.

## Question

Does the already accepted private chain remain exact at full real-file scale when
one frozen P98 real DNG is rendered directly to the official ACES 2 canonical
Rec.2100-PQ RGB16 PNG and independently routed through the exact P259
ACES2065-1/AP0 OpenEXR master plus the strict P251/P252 intake?

## Frozen role

- Input: the existing CC0 Blackmagic Pocket Cinema Camera 4K DNG already bound
  by P98/P259, SHA-256
  `1491a5e5d580a9be151747bddce89faca4d0b9d352503270f516bbf8e8cdf0d9`.
- Shape: `2176x4128x3`; the P259 ACEScg row contains both strong negative and
  strong-highlight samples.
- No new source, target, fit, photographic score, display score or network read.
- Direct control: exact P98 WorkingImage to the retained official ACES 2
  canonical Rec.2100-PQ RGB16 PNG publisher.
- Candidate: the same WorkingImage to the exact P259 producer-bound AP0/D60
  float32 ZIP OpenEXR writer, then strict P251 intake and the unchanged P252
  official ACES 2 canonical PQ publisher.

## Frozen gates

1. Parent files, source bytes and producer writer object are exact.
2. Direct and AP0-interchange decoded RGB16 samples are byte-exact.
3. Direct and AP0-interchange canonical PNG files are byte-exact.
4. Both routes are invariant to forward/reverse row partition schedules.
5. Direct and read-back working inputs are finite, source-owned and unchanged;
   negative/highlight support is retained by the AP0 master.
6. Outputs are strict Rec.2100-PQ RGB16 PNG and finite in `[0,1]` before
   quantization.
7. Source DNG is immutable; destinations are create-only; all P-backed
   temporary EXR/PNG files are removed.
8. Two fresh committed-head reports are byte-exact and network reads are zero.

Any decoded-sample, file-byte, identity, source, range, publication, replay or
cleanup failure closes this exact interchange route. No clipping, exposure,
matrix, rounding, tolerance, partition, source, output, metadata or cohort
rescue is allowed.

## Claim ceiling

Private one-file full-resolution mechanical interchange evidence only. It is
not arbitrary DNG/EXR/ACES/HDR support, calibrated camera rendering, display or
photographic quality, a public API/package/schema/capability, product admission,
film-stock evidence, single-reference matching, or candidate 3.
