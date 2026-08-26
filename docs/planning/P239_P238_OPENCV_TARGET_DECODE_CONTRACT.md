# P239 — P238 OpenCV Target Decode Contract

Status: **FROZEN BEFORE P238 PIXEL DECODE**  
Parent: `ULT > U1.4 > P238`  
Date: 2026-08-26

## Question

Can installed OpenCV 4.13.0/libpng 1.6.53 independently decode the exact P238
canonical RGB16 PNG without sample loss when a strict predecode inspector first
validates the full-range Rec.2100-PQ cICP and deterministic PNG structure?

## Frozen roles and path

- Rebuild the exact 29x34 P238 file in a system-temporary create-only root.
- Before OpenCV decode, validate geometry, RGB16 layout, chunk/CRC/filter
  structure, cICP `09 10 00 01` and native sample hash with the frozen strict
  P238/P91 reader.
- Decode the file and the exact same bytes through OpenCV
  `IMREAD_UNCHANGED`; require uint16 HxWx3 BGR, convert to contiguous RGB, and
  require its sample hash to equal the strict reader.
- Run file-first and memory-first order in two fresh processes. No alternate
  decoder, flag, channel interpretation, tolerance or pixel conversion.

## Frozen gates

1. P238 evidence/file identities and OpenCV binary/build identities are exact.
2. Strict predecode cICP/structure/sample validation passes.
3. File and memory decodes are non-null uint16 `29x34x3`.
4. File and memory RGB sample hashes equal strict P238 samples exactly.
5. File and memory decoded arrays are byte-identical.
6. Source PNG is immutable; malformed cICP, CRC and truncated controls reject
   before usable output.
7. Temporary outputs are removed and two fresh-process reports are exact.

## Decision and boundary

PASS retains only private target-runtime sample decode compatibility for the
exact P238 file. OpenCV cICP visibility remains explicitly false: semantic
validation belongs to the strict wrapper. No HDR display, tone mapping,
arbitrary PNG, package/schema/capability or product claim opens. FAIL closes
this exact OpenCV/libpng path without backend/flag/tolerance rescue.
