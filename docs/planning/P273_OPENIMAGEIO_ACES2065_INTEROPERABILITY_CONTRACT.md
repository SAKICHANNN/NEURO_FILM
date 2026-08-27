# P273 OpenImageIO ACES2065-1 Interoperability Contract

Status: frozen before OpenImageIO reads the P249 container.

## Question

Can the exact private P249 ACES2065-1/AP0/D60 OpenEXR container be read by the
already-installed OpenImageIO runtime with exact float32 pixels and all required
ACES identity metadata, without changing P249/P251 or adding a public dependency?

## Frozen execution

- Rebuild only the exact P249 7x9 synthetic container from its locked producer
  writer and offline OpenEXR wheel.
- Read it in two fresh OpenImageIO 3.1.11.0 processes in forward/reverse control
  order.
- Require exact container SHA, float32 pixel SHA, shape, RGB channel order,
  compression, `acesImageContainerFlag`, `colorInteropID`, AP0 chromaticities and
  D60 adopted neutral.
- Require rejection of a truncated container, immutable source bytes, byte-exact
  reports and zero temporary residue.

## Stop and claim boundary

Any runtime, source, pixel, metadata, negative-control, replay or cleanup mismatch
closes this exact interoperability leaf without version substitution, metadata
rewrite or tolerance rescue. A pass proves only private compatibility of one
synthetic P249 container with one existing Windows OpenImageIO runtime. It does
not add a dependency or API, validate arbitrary ACES/EXR media, establish SMPTE
certification or image quality, change the renderer, map a product capability, or
change candidate 3.
