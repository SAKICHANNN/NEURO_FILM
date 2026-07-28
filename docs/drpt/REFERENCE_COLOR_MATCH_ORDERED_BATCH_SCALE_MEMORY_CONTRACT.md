# P160 — 24 MP ordered-batch scale memory contract

P160 tests the product-facing `reference + N sources` file path after the
single-source P159 target-scale pass. It measures whether three ordered 24 MP
sources remain sequential in memory, deterministic in output order and clean
at the transaction boundary.

## Frozen subject

- Candidate commit: `4741507edcc03520147cdef11a13b1a78e4f8409`
- One deterministic 6000-by-4000 RGB8 PNG reference.
- Three distinct deterministic 6000-by-4000 RGB8 PNG sources, supplied in a
  fixed order and written to three matching RGB16 PNG outputs.
- Default product guard policy, which must return identity fallback for every
  synthetic source.
- Two fresh worker processes against detached worktrees of the exact candidate
  commit.

## Frozen gates

- Every worker exits successfully within 150 seconds with no orphan process.
- Peak process-tree RSS is at most 2.25 GiB and the larger/smaller repeat ratio
  is at most 1.15.
- The ordered list of source hashes and ordered list of output hashes are exact
  across both runs.
- Recipe bytes and the path-normalized report are exact across both runs.
- The report preserves source/output order and every source has
  `identity-fallback`.
- No reference-match staging temporary remains and every detached worktree is
  removed.

## Claim ceiling

A pass is only local Windows/Python evidence for a three-source sequential
24 MP batch. It is not a device-runtime, RAW, HDR, codec, visual-quality,
algorithm-promotion or four-platform readiness claim.
