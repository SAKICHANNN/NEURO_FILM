# P159 24 MP Target-Scale File-Path Evidence

## Result

The fixed P158 colour path executes two fresh 6000-by-4000 reference/source
transactions within every frozen P159 gate. Peak process-tree RSS is
`1,887,232,000 B` and `1,876,660,224 B`; the repeat ratio is
`1.0056332925`. Worker wall time is `33.0761 s` and `30.9289 s`.

The 3 GiB, 120-second and 1.15 repeat-stability gates all pass. This is a
target-scale pass, not another optimization comparison.

## Integrity

Both runs reproduce:

- output SHA-256 `21e590ca...346c8`;
- recipe SHA-256 `a8506e9c...d76eb`;
- normalized report SHA-256 `637cc101...e61e`;
- default identity fallback;
- zero staging temporaries, orphan workers or detached-worktree residue.

The frozen config, runner and ignored report SHA-256 identities are
`81f6d951...1086a`, `239d6180...b755` and `8a8d2b34...f9861`.

## Phase evidence and decision

The 5 ms worker sampler observes reference/source loading around
1.876--1.884 GB, encoding around 1.523--1.598 GB and row-bounded gamut around
1.353--1.357 GB. Loader memory is now the dominant local phase, but the whole
path remains comfortably below the preregistered 3 GiB ceiling. P159 therefore
does not authorize a speculative consumer-side loader rewrite.

The next useful scale evidence is ordered multi-source execution or a real
target-platform/media rail, not another 6 MP micro-optimization.

## Boundary

This proves only one reference plus one source, SDR PNG input, 16-bit PNG
output, local Windows/Python and 24 MP. It is not a 100 MP, N-source constant-
memory, RAW/HDR/video, main/native/device, A1/A4/A5, visual-quality or product
readiness result.
