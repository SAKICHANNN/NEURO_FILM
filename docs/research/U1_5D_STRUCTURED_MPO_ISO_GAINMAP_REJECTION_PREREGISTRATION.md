# U1.5D structured MPO / ISO gain-map rejection preregistration

Date: 2026-08-21
Parent: `ULT > U1 > U1.5`
Contract: `configs/u1_5d_structured_mpo_iso_gainmap_rejection_v1.json`

## Question

Can the raster ingress identify recognized gain-map metadata inside JPEG/MPO
metadata before it reaches either an SDR frame-zero fallback or the generic
multi-frame rejection, without treating every MPF/MPO file as HDR?

ISO 21496-1:2025 defines gain-map metadata for dynamic-range conversion. The
current Android Ultra HDR guidance recommends support for both Ultra HDR v1
and ISO 21496-1 metadata, while libultrahdr is the reference gain-map codec.
This leaf only strengthens explicit rejection because this renderer has no
gain-map reconstruction path.

## Frozen implementation

1. Add the exact ISO namespace `urn:iso:std:iso:ts:21496:-1` to the bounded
   recognized metadata tokens.
2. Run the existing bounded JPEG APP metadata traversal for files classified
   by Pillow as either `JPEG` or `MPO`.
3. Check recognized dynamic-range signals before the generic multi-frame gate.
4. Do not infer HDR from MPF/MPO alone. An MPO without a recognized gain-map
   token retains the existing generic multi-frame rejection.

The two exact CC-BY-4.0 Apple gain-map fixtures pinned by U1.5C are mandatory.
Synthetic metadata fixtures cover the ISO namespace and the ordinary-MPO
negative control. No decoder, gain-map math, metadata-field interpretation or
output transform is added.

## Stop and claim boundary

Any silent SDR decode, ordinary JPEG regression, generic-MPO false HDR label,
output creation, malformed-metadata fail-open or focused regression failure
closes this exact implementation. Passing proves only bounded recognition and
fail-closed routing for the exact tokens and fixtures. It does not prove full
ISO 21496-1 validation, Ultra HDR decoding, HDR quality, arbitrary MPF
semantics, public schema/capability or product readiness.
