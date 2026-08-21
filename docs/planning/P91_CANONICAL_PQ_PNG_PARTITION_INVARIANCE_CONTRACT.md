# P91 canonical PQ PNG partition-invariance contract

## Defect and question

P90 remains formally failed: three of four large PNG byte streams changed when
identical ordered RGB16 scanlines were supplied under a reversed sequence of
row-chunk sizes. Can a separately versioned writer feed zlib in fixed
64-KiB uncompressed blocks independent of caller writes, while preserving all
native samples, metadata, bounded buffering and existing writer bytes?

P91 is frozen before implementation. It is a deterministic runtime repair,
not new HDR quality evidence and not a reinterpretation of P90.

## Frozen mechanism and gates

- Keep the existing U1.4G/P90 writer unchanged and add one private canonical
  PQ writer version.
- Buffer at most one fixed 65,536-byte uncompressed filtered-scanline block
  plus the next caller contribution, feed complete blocks to the unchanged
  zlib level-0 compressor, and feed the final remainder only at finish.
- Preserve IHDR/cICP/IDAT/IEND, CRC, RGB16 samples, create-only publication and
  failure atomicity.
- Synthetic unequal partitions and all four exact P90 DNG outputs must have
  identical complete PNG bytes, sample hashes and strict readbacks.
- Two fresh processes and forward/reversed fixture and partition order must be
  byte exact. Existing U1.4G/P90 writer regression bytes must remain unchanged.

Any failure closes P91 without changing block size, zlib level, IDAT size,
filters, row order, samples or report normalization.

## Claim ceiling

Private deterministic PQ PNG compression-feed mechanics only. P90 remains a
formal failure; no HDR quality, HDR10 metadata, arbitrary input, default
writer/renderer/loader, public capability, product or delivery admission.
