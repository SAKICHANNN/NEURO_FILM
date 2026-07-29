# P165 — BT.2020 SDR PNG/CICP file matrix evidence

## Result

Both frozen 2048-by-1536 cases pass twice:

| Case | Ordered output rail/profile | Peak process-tree RSS | Worker time | Repeat ratio |
|---|---|---:|---:|---:|
| BT.2020 only | `linear_rec2020` / CICP | 493,764,608 / 494,096,384 B | 5.5407 / 5.5733 s | 1.000672 |
| mixed SDR | `linear_srgb` / ICC, `linear_rec2020` / CICP | 508,538,880 / 514,576,384 B | 9.9937 / 9.7575 s | 1.011872 |

Every output is 16-bit PNG. Ordered output bytes, recipe and normalized report
reproduce exactly within each case. All three source decisions remain
`identity-fallback`, and the isolated candidate worktree is removed.

The frozen config, runner and raw report SHA-256 identities are
`a89d3b1c...bd725`, `06adf04a...9bc27` and `9a4b4231...bc629`.

## Execution note

The first parent invocation stopped before input generation because the new
config omitted the shared preflight block. Commit `bc0aa7a` added the same
8 GiB memory/competitor and 4 GiB disk preflight used by P163; no gate was
lowered and the failed invocation executed zero pixel tasks.

## Interpretation

The consumer file transaction preserves its advertised relative
display-linear BT.2020 SDR PNG16/CICP rail both alone and in an ordered mixed
sRGB/BT.2020 batch on this Windows/Python host.

## Boundary

This is not absolute HDR or PQ/HLG evidence. It does not prove arbitrary
ICC/CICP conversion, scene-linear RAW, OCIO/ACES, target-device runtime,
non-identity quality or product readiness.
