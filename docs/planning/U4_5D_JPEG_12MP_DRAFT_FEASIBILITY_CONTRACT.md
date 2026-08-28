# U4.5D JPEG 12MP native draft feasibility contract

## Question

Can the exact 24.39MP U4.5B JPEG source satisfy the still-open 12MP product
tier through a native libjpeg draft that is both smaller than the source and
not smaller than the frozen 12MP target, before any pixel decode or renderer
work?

## Frozen source and runtime

- Reuse the exact U4.5B display-sRGB JPEG fixture, 4032x6048, without replacing
  or decoding it.
- Derive the target with the unchanged `preview_dimensions` function and a
  12,000,000-pixel ceiling: 2828x4242 / 11,996,376 pixels.
- Bind Pillow 12.1.1, its exact `JpegImagePlugin.py`, `_imaging` binary and
  reported libjpeg-turbo 8.0 runtime.
- Invoke only `Image.open` and `Image.draft("RGB", target)`. Do not call
  `load`, `convert`, `getdata`, NumPy conversion, preview rendering or output
  encoding.

## Frozen gates and stop rule

Both fresh forward/reverse processes must reproduce the exact source identity,
target geometry, draft geometry, mode, decoder-tile geometry and zero-pixel
boundary. A feasible native 12MP draft must satisfy all of:

1. draft width and height are at least the frozen target width and height;
2. draft width and height are no larger than the source;
3. at least one draft dimension is strictly smaller than the source;
4. the resulting draft pixel count is at most 12,000,000;
5. source bytes remain exact and no pixel or output artifact is read/written.

Failure closes only this exact native libjpeg-draft route. Do not undershoot and
upsample, change the target, select another source, decode full pixels, alter
Pillow/libjpeg, or introduce a custom decoder after observing the result.

## Claim ceiling

Private Windows/Pillow exact-source 12MP JPEG native-draft feasibility only.
No preview fidelity, render latency/memory, final export, generic ingress,
arbitrary JPEG, device performance, calibrated stock response, product release
or stock-evidence claim.
