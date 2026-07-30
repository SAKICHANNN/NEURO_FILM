# P160 — 24 MP ordered three-source batch evidence

## Result

Two fresh three-source batches pass every frozen gate. Peak process-tree RSS is
`2,173,702,144 B` and `2,172,657,664 B`, with repeat ratio `1.0004807384`.
Worker wall time is `90.8728 s` and `89.5944 s`.

All three ordered output hashes, recipe bytes and normalized report reproduce
exactly. Every source takes `identity-fallback`; no staging temporary, orphan
worker or detached worktree remains. The raw ignored report SHA-256 is
`06a91a0e...d274c`.

## Lifecycle finding

The transaction is within the 2.25 GiB gate, but later source loading peaks at
`2.165--2.172 GB`, about 286 MB above the P159 one-source peak. Code inspection
shows `_execute_file_render` keeps the prior `rendered.image` alive until the
next `rendered = ...` assignment even though durable output metadata has
already been extracted and the image encoded.

This is not a P160 failure. It is a narrowly evidenced P161 opportunity:
release `rendered` immediately after appending its scalar/diagnostic metadata,
prove collection before the next source load, and replay the same frozen P160
workload without changing bytes, ordering, transaction or guard semantics.

## Boundary

This is local Windows/Python, SDR PNG input and 16-bit PNG output evidence. It
does not establish constant memory for arbitrary N, RAW/HDR/video, target
devices, visual quality, producer compatibility or product readiness.
