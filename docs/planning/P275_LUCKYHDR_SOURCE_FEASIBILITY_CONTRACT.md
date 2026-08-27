# P275 LuckyHDR source-feasibility contract

## Scope

P275 is a zero-pixel, zero-model-execution source audit for the official 2026
LuckyHDR release.  It asks whether one exact bundled real iPhone ProRAW bracket,
the released checkpoint, and the exposure-metadata inference path are bounded
and independently addressable enough to justify a later private runtime D0.

P275 does not download any DNG or checkpoint body.  It does not decode pixels,
load weights, run inference, inspect an HDR target, or score quality.  It cannot
open candidate 3 or a product capability.

## Frozen source

- Repository: `princeton-computational-imaging/lucky-hdr`.
- Commit: `f5aec06fb70cd092973be5d15ec43899f8209293`.
- Tree: `37c05025490bd8a67494d953f23a7cebb521bc9a`.
- Root licence: MIT Git blob `3973fb0a663d2506f2e11e1fc769063af30b0ee8`.
- The exact three DNGs, checkpoint, README, dataset note, third-party notice,
  inference source and model source are bound in the committed P275 config.

The three DNGs and checkpoint must remain ordinary Git blobs with exact sizes;
Git LFS pointers, redirects, release assets or inferred filenames are not
substitutes.

## Frozen questions and gates

P275 passes private bounded source feasibility only if both request orders
establish all of the following from exact Git objects:

1. the repository commit and recursive tree are exact;
2. the root MIT licence and authored-code boundary are explicit;
3. one checkpoint is an exact regular Git blob smaller than 2 MiB;
4. exactly three full-resolution DNG blobs are named short/mid/long, their
   aggregate size is below 32 MiB, and the demo README gives ordered shutter,
   ISO and effective-exposure roles;
5. the inference source explicitly derives exposure from shutter times ISO;
6. the model source and checkpoint are fixed without loading either;
7. no DNG/checkpoint body, pixels, weights, target, training, inference or score
   are read during the audit;
8. forward and reverse scientific payloads are byte-identical.

Product rights must remain false unless the release itself fixes the capture
asset and upstream training-data rights.  The root MIT licence alone is not
silently generalized over ambiguous photographic assets or upstream SI-HDR
training data.  `DATASETS.md` says real capture data are not redistributed,
while later repository commits explicitly add and ship one demo bracket; this
conflict is recorded, not resolved by consumer interpretation.

## Stop rule and claim ceiling

Any source, object, size, role, licence, dependency, no-body-read or replay
mismatch closes P275 before acquisition.  A pass opens at most a separately
preregistered private source-only runtime D0 using only the exact three demo
DNGs and checkpoint.  It does not establish HDR ground truth, natural-image
quality, arbitrary brackets, dataset redistribution, commercial rights,
package/schema/capability/product admission, or candidate 3.
