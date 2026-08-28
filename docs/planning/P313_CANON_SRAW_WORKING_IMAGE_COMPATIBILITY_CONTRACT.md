# P313 Canon sRAW/mRAW WorkingImage compatibility contract

## Question

Can the unchanged generic LibRaw/rawpy product ingress decode six exact,
rights-clear Canon sRAW/mRAW files from six camera models into deterministic,
well-formed `WorkingImage` values without changing source bytes or overstating
camera-accurate rendering?

## Frozen source and rights

- The source is the official `raw.pixls.us` repository API as observed on
  2026-08-28. Every selected row uniquely binds an exact SHA-256 and declares
  Creative Commons Zero / Public Domain.
- The cohort contains exactly EOS 7D, EOS 7D Mark II, EOS 5D Mark II,
  EOS 5D Mark III, EOS 5D Mark IV and EOS 50D sRAW/mRAW files. No replacement,
  download or outcome-selected camera is allowed.
- The canonical six-row subset is retained repo-relative at
  `data/external/p313_canon_sraw_working_image_v1/source_rows.json`; the input
  CR2 files remain at their existing repo-relative paths.
- Source hashing and metadata-only repository inspection occurred before this
  contract. No P313 pixel decode occurred before the prospective freeze.

## Frozen implementation and execution

1. Use the unchanged `src.preprocess.raw_decode.inspect_raw` and
   `load_raw_working_image` implementation with rawpy 0.26.1 / LibRaw 0.22.0,
   camera white balance, no auto bright, 16-bit intermediate, explicit linear
   sRGB output, gamma `(1, 1)` and LibRaw camera orientation.
2. Run two fresh committed-head processes in forward and reverse source order.
3. For every row, require inspection and decode success, a nonempty HxWx3
   float32 array, finite values in `[0,1]`, nonconstant output, owned
   C-contiguous writable storage, `linear_srgb` / `scene_linear`, applied
   orientation, absent alpha, exact generic-render and uncalibrated-display
   warnings, and source immutability.
4. Canonicalize rows by source ID. Require the complete reports and every
   decoded float32 pixel hash to be byte-identical across process/order.
5. Read no network resource and publish no image. Temporary and persistent
   output residue must remain zero.

## Gates and stop rule

All six rows and every frozen gate must pass. Any source, runtime, inspection,
decode, structure, ownership, warning, replay, source-immutability or cleanup
failure closes this exact cohort. Do not replace a file, change white balance,
enable auto bright, change output space/gamma/orientation, add a decoder,
relax a gate or use producer R1FX/R1FY private reconstruction as a rescue.

Passing proves only private Windows/Python generic product-ingress
compatibility on these six exact CC0 Canon sRAW/mRAW files. R1FX/R1FY may be
recorded as independent mechanics evidence for their own exact EOS 7D/EOS 40D
sources, but P313 neither copies nor validates the producer implementation.
P313 does not establish vendor-exact colour, general Canon/sRAW support,
camera calibration, scene-to-display tone mapping, image quality, a new
decoder/interface/package/capability, stock evidence, product promotion or
candidate 3.
