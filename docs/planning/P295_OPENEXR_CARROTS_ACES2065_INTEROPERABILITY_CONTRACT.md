# P295 — official Carrots ACES2065-1 interoperability contract

Status: prospectively frozen before the EXR body is requested or decoded.

## Question

Does the current strict P251 ACES2065-1/AP0 OpenEXR WorkingImage ingress accept
one independent official ASWF photographic reference that the OpenEXR project
describes as stored with ACES2065-1 (AP0) chromaticities?

P295 tests ecosystem interoperability, not a new renderer or a relaxation of
P251. It is the sole representative official-sample leaf; failure closes this
sample-interoperability family without trying other images.

## Exact source and rights

- repository `AcademySoftwareFoundation/openexr-images`;
- revision `e38ffb0790f62f05a6f083a6fa4cac150b3b7452`;
- BSD-style licence blob `268d234adc762429a5d7db6a456651d7d0933dd1`;
- declaration blob `ScanLines/README.rst`,
  `ed0909aaf5930b39179afae905e93e15cccb284a`;
- `ScanLines/Carrots.exr`, 914,825 bytes, Git blob
  `b4697022c7120065b1d6131b20da363a8ff88069`.

The JPEG preview is forbidden. No other OpenEXR sample may replace Carrots.

## Frozen protocol

1. acquire and verify the exact one-file body after this contract commit;
2. bind the pinned OpenEXR 3.4.15 wheel and unchanged P251 module bytes;
3. read only the header first;
4. require every existing P251 identity and storage fact before pixels:
   single non-deep scanline packed RGB float32, AP0/D60 chromaticities,
   D60 `adoptedNeutral`, `acesImageContainerFlag=1` and
   `colorInteropID=lin_ap0_scene`;
5. if any header/storage gate fails, stop with zero channel/pixel reads and zero
   WorkingImage output;
6. only if all header gates pass, call unchanged P251 and require finite owned
   writable contiguous ACEScg/AP1 pixels, source immutability and two exact
   opposite-order fresh-process reports.

## Stop and claim boundary

Do not insert missing metadata, reinterpret HALF as FLOAT, rewrite the file,
change P251, use the JPEG, choose another sample, normalize/clip, or compare a
display rendering after a failed header gate.

A pass would prove only one private independent official-file interoperability
case. A failure is a precise strict-container compatibility result, not an
OpenEXR, ACES or image-quality negative. No default loader, arbitrary EXR,
SMPTE certification, package/schema/capability/product, stock evidence or
candidate-3 change opens.
