# P159 24 MP Target-Scale File-Path Contract

## Question

P156-P158 make the colour kernels exact and bounded at 6 MP. Before changing
the remaining loader or encoder, P159 asks whether that implementation already
runs a realistic 24 MP reference/source transaction within a conservative
local memory and time envelope.

The fixed functional candidate is
`4741507edcc03520147cdef11a13b1a78e4f8409`.

## Frozen workload

- deterministic RGB8 PNG reference and source, each 6000 by 4000;
- one source, default identity fallback, 16-bit PNG output;
- two fresh worker-process repetitions from one detached candidate worktree;
- 20 ms complete process-tree RSS sampling and the existing 5 ms phase
  attribution;
- create-only output root and exact cleanup checks.

The input seeds, commit, geometry, bit depth and repeat count are fixed before
execution.

## Frozen gates

Both runs must finish within 120 seconds, use no more than 3 GiB peak
process-tree RSS, leave zero orphan workers and zero staging temporaries, and
reproduce exact output, recipe, normalized semantic report and
`identity-fallback`. The ratio of larger to smaller peak RSS must be no more
than `1.15`.

These are scale/readiness gates, not comparative optimization gates. A failure
does not authorize threshold changes or an automatic loader/encoder rewrite;
phase evidence must first identify the responsible boundary.

## Claim ceiling

A pass proves only that this fixed local Windows/Python SDR path executes this
24 MP workload within the frozen envelope. It is not a 100 MP, multi-source,
RAW/HDR/video, native/device, main-renderer, A1/A4/A5, visual-quality or
product-readiness claim.
