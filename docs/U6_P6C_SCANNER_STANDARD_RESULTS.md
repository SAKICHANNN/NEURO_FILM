# U6.P6C Scanner Standard Results

U6.P6C passes its frozen development gates. The new Standard path preserves
all five scanner stages in explicit float32 arithmetic, uses a float64 global
mean accumulated once and cast to float32, and does not hard clip or quantize.

Two independent 24MP runs produced the same stable evidence ID
`250aeec564a16c8b5e9cf61d73302a2058f9cad286e80b4e372534568463414c`
and output SHA-256
`33bd1784d48b5bb0a612ad9a6e9f0b696148a0df8647723cf7f4e2bfb94dd67e`.
The 257-row and 1024-row partitions are byte-exact. Maximum error against the
float64 reference is `1.33575e-7`, below the frozen `5e-6` gate.

| Tile rows | Apply time, run A/B | Peak process-tree RSS, run A/B |
|---:|---:|---:|
| 257 | 19.65 / 20.15 s | 0.920 / 0.925 GiB |
| 1024 | 14.95 / 14.71 s | 1.438 / 1.436 GiB |

Both policies pass the frozen 45-second and 1.5-GiB gates. The 257-row policy
is the safer current default because it has substantially more memory margin.
The reports are retained as ignored outputs under
`outputs/u6_p6c_scanner_standard_v1_run_{a,b}/`.

This is generic scanner-runtime development evidence only. It does not
identify or calibrate any scanner, stock, process, spectral response, MTF,
noise model, or final product renderer. The next leaf is the preregistered
joint colour-plus-physics ablation; simpler colour-only remains the fallback.
