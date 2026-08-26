# P235 R1CZ structural black-anchor contract

## Purpose

P235 is a finite explanatory stop-rule leaf after the formal P234 UltraHDR
still-domain rejection. It asks whether the exact persisted R1CZ payload has
the structural black anchor required before spending another application
domain on safety testing.

This is deliberately not a third quality or transfer experiment. P235 reads
only the tracked P233 capsule and the already committed P233/P234 evidence.
Application images, paired-build pixels, media and network resources are all
forbidden.

## Frozen facts

- Payload SHA-256: `aa7fe040...28cb9`, 4,376 canonical JSON bytes.
- Bundle: `2759d219...c3220f`.
- Profile: absolute BT.2020/D65 float32 cd/m2 with reference white 203 nits.
- The D-PCT mapping and its existing `[0, 10000]` output clamp are unchanged.
- P234 is already observed and failed its chroma/boundary gates. Therefore
  P235 can explain and stop work, but cannot be counted as independent
  confirmation of that result.

## Frozen probes and gates

The audit evaluates the exact black vector and a fixed scalar-neutral shadow
sequence. It reports the unclamped channel mapping, the existing consumer's
clamped output, the first fitted source/reference knot in nits and the
per-channel input value whose extrapolated mapped value is zero.

All of these gates are mandatory:

1. P233 and P234 identities and statuses match exactly.
2. The local payload capsule and envelope match exactly.
3. Exact black maps to exact zero in every channel before the output clamp.
4. Exact black remains exact zero after the existing output clamp.
5. Every channel's extrapolated zero crossing is within `1e-6` nit of zero.
6. No fixed neutral-shadow probe has a mixture of exact-zero and positive
   output channels.
7. Fresh forward/reverse reports are byte-identical.
8. Application-pixel, paired-build, network and media reads remain zero.

The black and common-zero requirements are physical structural invariants for
an independently applied absolute-light transform; they are not fitted image
quality thresholds.

## Stop rule and claim ceiling

Any failed gate closes further arbitrary independent absolute-light
application-domain accumulation for this exact payload. P233 persistence and
the original paired R1CY result remain valid. There is no black-floor, offset,
knot, strength, clamp, tolerance, probe or additional-domain rescue.

Even a pass would retain only a private structural preflight. It would not
establish natural-HDR quality, after-only inference, a public payload format,
package/schema/capability or product admission.
