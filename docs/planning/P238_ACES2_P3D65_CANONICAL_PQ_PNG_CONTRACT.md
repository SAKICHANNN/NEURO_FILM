# P238 — ACES 2 P3-D65 Canonical PQ PNG Contract

Status: **FROZEN BEFORE IMPLEMENTATION**  
Parent: `ULT > U1.4 > P226`  
Date: 2026-08-26

## Question

Can the exact private P226 official ACES 2 P3-D65 / 1000-nit
Rec.2100-PQ float target be published as one deterministic, create-only RGB16
PNG with full-range Rec.2100-PQ CICP, while preserving exact samples across
caller row partitions and leaving the existing Rec.2020-PQ target unchanged?

## Frozen inputs and information flow

- Use the exact P226 986-row accepted-domain ACEScg fixture and reshape it to
  `29x34x3`; no new pixels, target data, fit, optimization or network access.
- Apply only `hdr_p3d65_1000nit_rec2100_pq` through the pinned OCIO 2.5.2
  built-in ACES 2 config.
- Quantize with `floor(float64(code) * 65535 + 0.5)` to RGB16.
- Publish through the existing canonical Rec.2100-PQ writer. Caller row
  partitions are forward and reversed `7`-row partitions.
- Retain the untouched P226 wide fixture solely for the existing
  `hdr_rec2020_pq` regression hash.

## Frozen gates

1. P226 evidence, adapter and writer identities match the frozen contract.
2. The 986-row float output is contiguous float32, finite and within `[0,1]`.
3. Published samples equal the frozen round-nearest expression exactly.
4. Strict PNG sample readback equals the in-memory RGB16 sample hash.
5. PNG metadata is exactly full-range Rec.2100-PQ CICP `09 10 00 01`.
6. Forward and reversed partitions produce byte-identical complete PNGs.
7. The existing `hdr_rec2020_pq` output hash remains unchanged.
8. Input bytes are immutable; invalid shape/type/value/row count reject.
9. Existing outputs are create-only; injected writer failure leaves no final
   file or temporary residue.
10. Two fresh-process formal reports are byte-identical.

## Decisions

- **PASS:** retain one private numerical/media publication rail and open only a
  separately frozen target-runtime decode check.
- **FAIL:** close this exact publication route without changing quantization,
  CICP, OCIO target, writer, fixture, partitioning or gates.
- **INVALID:** missing frozen identities or runtime is infrastructure-invalid,
  not a scientific result.

## Claim ceiling

Private Windows/Python deterministic RGB16 PNG arithmetic and metadata only.
No HDR10 conformance, mastering metadata, display validation, photographic
quality, arbitrary-image support, public package/schema/capability, calibrated
stock or product admission.
