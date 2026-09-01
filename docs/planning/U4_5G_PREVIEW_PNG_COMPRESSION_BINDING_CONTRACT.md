# U4.5G Preview PNG Compression Binding Contract

## Question

Does the existing direct three-stock preview path honor its already-public
`png_compression` argument, record that choice for replay, and preserve the
historical default level-6 image bytes?

This is a product correctness repair. It does not change the three Look
Approximation algorithms, preview pixels, source decode, cache authorization,
stock claims, or final-export behavior.

## Frozen defect witness

Before implementation, one deterministic 121x83 synthetic fixture was rendered
with compression levels 0 and 9. All three output SHA-256 identities and byte
lengths were identical. Code inspection localized the defect: the argument was
validated by `render_three_stock_previews_to_directory` and passed by the CLI,
but omitted from the `save_srgb8` call.

## Frozen roles

- Source: the existing rights-cleared U7.3G JPEG at
  `data/preference/repid/u5_r2repid3_shared_operator_v1/fit/winner/a4593-kme_0276.jpeg`.
- Render geometry: at most 4,000 pixels using the already-proven opt-in libjpeg
  scaled decode, followed by the unchanged linear-light area resize.
- Look order: `velvia_50`, `portra_400`, `ektar_100`.
- Candidate levels: 0, 6 and 9.
- Backward oracle: the three exact pre-repair default level-6 PNG hashes locked
  in `configs/u4_5g_preview_png_compression_binding_v1.json`.

## Required behavior

1. The existing RGB8 encoder accepts an optional PNG compression level in
   `[0, 9]`; its omitted default remains byte-identical to historical behavior.
2. The preview renderer passes the validated level to the encoder and records
   `png_compression` in `preview.json`.
3. Default and explicit level 6 produce identical PNG and manifest bytes.
4. Levels 0 and 9 decode to exactly the same RGB8 samples and ICC bytes as
   level 6, while their encoded PNG identities differ.
5. Level 9 must not exceed level 0 in byte length on every frozen look.
6. Invalid levels and non-PNG compression requests fail closed; preview invalid
   levels reject before source inspection or decode.
7. Source bytes, profile/statistics/guardrail inputs and pre-existing foreign
   destinations remain unchanged; owned temporary material is removed.
8. Forward and reverse execution reproduce the same scientific payload and all
   candidate output identities.

## Stop rules

- Do not change source, looks, pixels, resize/decode path, codec, ICC profile,
  thresholds or compression levels after observing results.
- Failure closes this exact parameter-binding repair; do not rescue through a
  different encoder, output format, fixture, tolerance or compression sweep.
- Do not extend this leaf into cache wrappers, RAW preview timing, batch/export
  tuning, calibrated-stock claims, public packaging or release work.

## Claim ceiling

Private deterministic PNG8 preview-encoding parameter correctness for three
existing film-inspired Look Approximations. No calibrated or physical film-stock
response, image-quality preference, final-export performance, public API,
package, installer or release claim.
