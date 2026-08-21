# P98 — opt-in DNG ForwardMatrix raster contract

Status: frozen before implementation and before the formal five-row raster run.

## Question

Can the exact P94 DNG `ForwardMatrix` metadata path and the exact P97
D50-PCS-to-linear-Rec.2020 primitive be composed with one deliberately narrow
camera-linear LibRaw demosaic into a deterministic `WorkingImage`, without
changing the generic RAW loader or silently clipping scene-linear values?

This is mature private RAW/DNG explicit-operator engineering. It is not an
after-only reference-matching candidate and does not establish photographic
quality, camera calibration, a scene-to-display rendering, or product support.

## Frozen parents and rows

- P94 evidence SHA-256
  `28dbe488dac052ca374d35b883d9eaec93d3f2adb63223bc04ab87290c4ba355`.
- P97 evidence SHA-256
  `63dfd262b06ba37faafa77fabf6f0cc34a1e039766eab959e06ac5914629ee5a`.
- The five exact P94 CC0 raw.pixls.us DNG byte identities are mandatory:
  Motorola moto g(7) play, LG-H850, Xiaomi M2010J19CG, Blackmagic Pocket
  Cinema Camera 4K, and Huawei EML-L29. There is no replacement or row rescue.

The P94 metadata matrix must be reconstructed from the bound DNG tags for each
row. A stored report matrix is not accepted as the runtime input.

## Frozen implementation

The opt-in primitive may:

1. use rawpy 0.26.1 / LibRaw 0.22.0 to demosaic to three-channel camera RGB;
2. use unit user white balance, linear gamma, no automatic brightening,
   16-bit output, LibRaw black/white normalization, and source orientation;
3. convert the uint16 camera RGB to float64 in `[0,1]`;
4. apply the exact P94 camera-to-D50-PCS matrix and P97 D50-PCS-to-linear-
   Rec.2020 primitive without clipping;
5. return a float32 `WorkingImage` in `linear_rec2020`, `scene_linear`, with
   explicit warnings that demosaic and calibration/appearance are unqualified.

It must reject a non-DNG path, missing or unsupported P94 metadata, decode
failure, wrong camera-raster shape/type, non-finite intermediate or output, and
source-byte drift. It must not call the generic `load_raw_working_image`, use a
LibRaw output colour matrix, auto white balance, auto brightness, tone map,
gamut map, output encoder, or project default loader. The primitive is not
exported from `src.preprocess` in this leaf.

## Frozen evaluation and gates

- exactly five rows and five camera makes; every source byte count/SHA matches;
- rawpy/LibRaw versions exactly `0.26.1` / `[0,22,0]`;
- every decoded camera raster is non-empty uint16 HxWx3 and every returned
  `WorkingImage` is finite float32 with the same HxW;
- camera-raster values lie in `[0,1]`; output values are not clipped, with
  absolute maximum at most `8` and outside-unit component fraction at most
  `0.25` on every row;
- staged camera→PCS→Rec.2020 and one directly composed float64 matrix agree to
  maximum absolute error `5e-15` on every decoded pixel;
- the primitive float32 pixels agree exactly with the independently staged
  float32 oracle bytes;
- source files remain byte exact after decode; input camera arrays remain
  unchanged after colour conversion;
- forward/reverse row enumeration and two fresh-process reports are byte exact;
- all failure cases are fail-before-output.

The existing generic rawpy direct-to-Rec.2020 result is recorded only as a
non-binding diagnostic because it uses LibRaw colour-matrix policy rather than
the P94 DNG `ForwardMatrix` construction. It is not a truth target or gate.

## Stop rule and claim ceiling

Any binding, decode, matrix, finite, range, mutation, error, or replay failure
closes the exact P98 path. No row, rawpy parameter, matrix, tolerance, clipping,
exposure, resize, demosaic, or report-normalization rescue is allowed.

PASS means only a private opt-in five-DNG camera-linear-demosaic plus explicit
DNG ForwardMatrix to D50 PCS to linear-Rec.2020 `WorkingImage` mechanism. It
does not establish arbitrary DNG support, sensor/IDT calibration, vendor or
Adobe parity, photographic/colorimetric quality, tone mapping, default-loader
integration, output files, public package/schema/capability, product, film,
stock, or preference admission.
