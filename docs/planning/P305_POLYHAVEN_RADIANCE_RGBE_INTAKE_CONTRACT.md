# P305 Poly Haven Radiance RGBE intake contract

## Question

Can one exact, rights-clear, hash-bound Radiance RGBE file be decoded by a
strict project-owned parser with independent compiled-decoder parity, without
silently assigning RGB primaries, a white point or a `WorkingImage` color
state?

## Frozen source

- Official Poly Haven Public API repository commit
  `400522721de9f917dbd7340d58c7659312e7df6a`; its commit-pinned README states
  that downloaded assets are CC0.
- Official live API asset `sunset_jhbcentral`, bound by info/file response
  hashes and `files_hash=e2e14a204ee06104209396caa9e7b40322dbb3d4`.
- Exactly one variant may be acquired through its frozen direct URL:
  `hdri/1k/hdr`, 1,573,705 bytes, MD5
  `97335a81ba615beb6f6ae0da707ecd75`.
- The asset remains repo-relative under
  `data/external/p305_polyhaven_rgbe_v1/`; no broader API enumeration,
  thumbnail, EXR, backplate or alternate-resolution request is allowed.

## Frozen implementation and execution

1. Validate `#?RADIANCE`/`#?RGBE`, a supported `FORMAT=32-bit_rle_rgbe`, one
   canonical `-Y H +X W` resolution line, exact EOF and scanline dimensions.
2. Decode only standard Radiance new-style per-channel scanline RLE. Reject
   old-style pixels, orientation variants, malformed headers, truncated or
   overlong runs, width mismatches, non-finite results and trailing bytes.
3. Convert RGBE to owned, C-contiguous, writable float32 RGB radiance using
   the Radiance half-bin convention `(mantissa + 0.5) * 2^(E-136)` for
   nonzero exponents; zero exponent maps to exact black.
4. In two fresh processes and opposite check order, compare the full decoded
   array to installed OpenCV's independent compiled Radiance decoder after
   BGR-to-RGB reordering.
5. Freeze exact source/container/header/shape/range/hash/replay and invalid
   controls before interpretation.

## Gates and stop rule

All gates must pass: official CC0/source/API identity, exact payload
size+MD5+SHA-256, 1024x512 RGB float32 decode, finite/nonnegative values,
owned/contiguous/writable output, maximum absolute parity error <= `1e-7`,
source immutability, two reports exact, and malformed/truncated/orientation/
trailing-data controls all reject before output.

Failure closes this exact parser/source pair without tolerance, orientation,
header, decoder or source substitution. Passing opens only a private
source-locked Radiance RGBE arithmetic/container primitive. It does not assign
color primaries or white point, create a `WorkingImage`, establish arbitrary
`.hdr` support, HDR quality, package/schema/capability/product admission or
candidate 3.
