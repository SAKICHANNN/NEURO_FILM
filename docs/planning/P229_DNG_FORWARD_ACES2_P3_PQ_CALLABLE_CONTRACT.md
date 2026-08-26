# P229 — opt-in DNG ForwardMatrix to official ACES 2 P3-PQ callable

## Question

Can the retained exact-five-DNG P98 camera-linear ForwardMatrix raster and the
retained P226 official ACES 2 P3-D65 1000-nit Rec.2100-PQ target be composed
behind one narrow, source-bound callable without changing the generic RAW
loader or any default renderer behavior?

## Frozen scope

- Inputs are exactly the five source identities already frozen by P98.
- The callable must require caller-supplied source byte count and SHA-256 and
  must delegate to the unchanged P98 raster implementation.
- The only output target is the exact P226 private target
  `hdr_p3d65_1000nit_rec2100_pq` under the pinned built-in OCIO 2.5.2 config.
- The return value is an owned contiguous finite float32 HxWx3 normalized-PQ
  array. It is not an encoded file or a new `WorkingImage` colour state.
- The default `load_working_image`, generic RAW loader and `render_film` paths
  must remain byte-identical to their frozen pre-leaf source identities.

## Gates

All five rows must pass source identity/immutability, exact equality against
the direct retained P98-plus-P226 composition, float32 contiguity, finite unit
range, input ownership and normal/reverse report replay. Wrong source length,
wrong source hash, non-DNG input and dependency/runtime drift fail before a
usable output is returned.

## Stop rule and claim ceiling

Any failed gate closes this exact callable without clipping, exposure,
tone-map, config, target, source-row or tolerance rescue. A pass retains only
a private exact-five-DNG Windows Python callable. It does not establish
arbitrary DNG support, sensor/IDT calibration, vendor rendering, photographic
quality, default loader/renderer integration, integer/media output, public
package/schema/capability or product admission.
