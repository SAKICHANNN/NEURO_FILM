# U1.2C Multi-Frame Raster Fail-Closed Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.2 > U1.2C`

**Status:** frozen / implementation ready

**Config SHA-256:** `302366bcfaf5e5c1d7cec7d17f1099a0db341ec60c85316f0a0b9004ce473373`

## Defect and policy

Inspection already records Pillow's `n_frames`, but raster loading does not
enforce it and silently decodes only the first frame/page. Animated images and
multi-page scans can therefore lose content without warning.

U1.2C permits exactly one frame. Any inspected `frame_count != 1` must fail
before pixel decode with an explicit unsupported multi-frame message. Frozen
fixtures cover a two-frame GIF, two-page TIFF, ordinary single-frame regression
and integrated renderer rejection with no output.

This leaf does not implement animation, video, bursts, temporal effects,
multi-page export or frame selection.

## DoR / DoD

DoR: U1.2B passes; inspection exposes frame count; current head is
`7a1a9b435998415f5225a0983b23b006d2a5d347`; worktree is clean.

DoD: inspection/load tests prove rejection before decode, renderer creates no
output, ordinary single-frame inputs remain unchanged, complete CPU suite
passes, and results propagate through governance docs.

## Claim ceiling

Pass means only honest single-frame raster ingress. It does not establish any
animation, stack, video, temporal-consistency or multi-page capability.
