# P276 — LuckyHDR private official-bracket runtime D0

Status: `FROZEN_BEFORE_BINARY_ACQUISITION_OR_PIXEL_DECODE`

## Question

Can the exact official LuckyHDR source/checkpoint execute twice on its exact
bundled real three-frame iPhone ProRAW bracket in the current local CUDA runtime,
with deterministic finite output and bounded resources?

This follows P275's zero-pixel source pass. It is a private runtime/mechanism D0,
not an HDR-quality test: the release has no aligned HDR target for the demo.

## Frozen roles and execution

- Acquire only the exact commit-bound inference/model sources, checkpoint and
  short/mid/long demo DNGs listed in the config. Verify byte count and Git blob
  SHA-1 before create-only publication under the repo-relative P-backed data root.
- Keep frame order short, mid, long and use the official README's explicit
  exposure ratios `[1.0, 3.7, 30.0]`; do not depend on host exiftool state.
- Execute the official inference module unchanged at `--max-size 2048` on CUDA.
  No target, metric, training set or external capture may be read.
- Run two fresh processes in forward/reverse manifest-verification order.
  Require exact output PNG bytes, finite decoded RGB8, nonempty dynamic range,
  checkpoint strict-load success, source immutability, wall time below 120 s,
  and zero formal temporary residue.

## Stop rule and claim ceiling

Any object, dependency, load, CUDA, output, replay, finite/range, resource or
cleanup mismatch closes P276 without CPU/backend/exposure/checkpoint/input/size
substitution or tolerance rescue.

Passing proves only private execution of one exact official checkpoint on one
exact bundled bracket. The root MIT software licence does not resolve demo
capture redistribution or upstream SI-HDR training-data product rights. No HDR
ground-truth quality, arbitrary bracket, package/schema/capability/product or
candidate3 admission may be inferred.
