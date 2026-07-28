# P162 — maximum 64-source batch scale contract

P162 executes the product's exact `MAX_REFERENCE_MATCH_BATCH_SOURCES=64`
ceiling through the real file transaction after P161 removed the preceding
render lifetime overlap.

## Frozen workload

- Exact candidate commit `6ae40df1ee87e7e994b0571da6d81c5e24e26ff6`.
- One deterministic 1000-by-1000 RGB8 PNG reference.
- 64 distinct deterministic 1000-by-1000 RGB8 PNG sources in ascending index
  order, producing 64 matching RGB16 PNG outputs.
- Default identity-fallback guard and two fresh worker/worktree runs.

## Frozen gates

- Both workers finish within 240 seconds, peak below 1.5 GiB and reproduce RSS
  within ratio 1.15.
- Ordered source and output hash lists each contain exactly 64 items and are
  exact across runs.
- Recipe and normalized report are exact; report source paths preserve every
  index from 0 through 63.
- Every source takes identity fallback.
- No staging temporary, orphan worker or detached worktree remains.

## Claim ceiling

A pass proves maximum-count transaction/order/resource behavior only at 1 MP
per image on local Windows/Python SDR PNG. It does not prove 24MP-by-64,
arbitrary image sizes, RAW/HDR/video, native/device or product readiness.
