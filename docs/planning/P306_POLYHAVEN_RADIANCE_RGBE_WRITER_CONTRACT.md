# P306 Poly Haven Radiance RGBE writer contract

## Question

Can the exact P305 unlabelled linear RGB radiance array be quantized and
published as deterministic new-style Radiance RGBE such that both the official
Radiance C routines and the strict P305 decoder recover the same code stream,
without inventing a colour identity or widening generic HDR support?

## Frozen parents and oracle

- Reuse only the exact P305 CC0 `sunset_jhbcentral` 1K source and its strict
  decoded float32 array. No new image or source request is allowed.
- Bind official Radiance `src/common/color.c` revision `2.16`, downloaded from
  the project ViewVC endpoint, 6,492 bytes, SHA-256
  `2b60358ed05f69733c1856d1a72376168285cbaa408266da5e8194e7f1d7a4a2`.
- The independent oracle is a compiled C harness using the revision's exact
  `setcolr` quantization and `fwritecolrs` packet selection. It must not call
  the Python implementation.

## Frozen execution

1. Accept exactly finite, nonnegative, C-order `HxWx3` numeric arrays with
   `8 <= W <= 32767`; reject booleans, non-finite, negative, empty, extra-axis,
   undersized/oversized-width and exponent-overflow inputs before publication.
2. Quantize with official `setcolr`: select the maximum component, use
   `frexp(max) * 255.9999 / max`, truncate positive primaries to uint8, and
   store exponent bias 128; values at or below `1e-32` become exact black.
3. Encode each channel using the exact official new-style scanline RLE
   selection (`MINRUN=4`, literal blocks <=128, runs <=127), with a canonical
   minimal header and `-Y H +X W` order.
4. In fresh forward/reverse processes, require Python and compiled official-C
   outputs to be byte-exact, their decoded RGBE codes to reproduce the P305
   source codes exactly, and strict P305 float decode to reproduce the input
   exactly.
5. Test create-only atomic publication, source/input immutability, byte-exact
   replay and invalid-input/failure atomicity. Formal outputs are reports only;
   generated HDR media are temporary and must be removed.

## Gates and stop rule

All gates are exact: official source/harness/local bindings; Python-vs-C full
container bytes; source RGBE code bytes; decoded float32 bytes; forward/reverse
reports; input ownership/immutability; create-only atomic publication; invalid
controls; and zero formal media residue. Any failure closes this exact
writer/source pair without changing quantization, RLE, header, source, width,
precision or tolerance.

Passing opens only a private exact-source deterministic Radiance RGBE writer
and arithmetic/container roundtrip. It does not assign primaries, white point
or transfer characteristics, create a `WorkingImage`, establish arbitrary HDR
support or quality, alter default ingress/egress, or open a package, schema,
capability, product claim or candidate 3.
