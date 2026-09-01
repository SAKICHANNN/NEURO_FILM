# U7.19A SRW/ARQ generic WorkingImage product-ingress contract

## Product question

Can the existing public K-MCFM input path recognize source-locked Samsung SRW
and Sony ARQ files, decode them through the unchanged generic LibRaw/rawpy
WorkingImage boundary, and carry one representative of each admitted extension
through the public Look Approximation CLI, strict recipe, and byte-exact replay?

The current dispatcher recognizes many LibRaw formats but omits `.srw` and
`.arq`. P318 and P321 prove private stored-code callables only; they explicitly
do not prove generic product ingress, demosaic, colour, or rendering. U7.19A
does not copy or execute those private callables.

## Frozen sources and predecode state

- The five exact source files remain in the producer repository behind its
  repo-relative P-backed `data/` junction. U7.19A reads them in place and does
  not copy them into this repository.
- SRW is one independent stratum containing all four exact CC0 Samsung NX500
  12/14-bit normal/lossless files bound by P318.
- ARQ is one independent stratum containing the exact CC0 Sony ILCE-7RM3
  four-shot ARQ bound by P321.
- Source size and SHA-256 were verified before this contract. No U7.19A
  `rawpy.imread`, RAW inspection, postprocess, pixel decode, product render, or
  output occurred before this freeze.
- P318/P321 remain immutable private-mechanics evidence. A U7.19A result must
  not rewrite their claim ceilings or use their callables as an oracle/rescue.

## Frozen independent-stratum admission

SRW and ARQ are evaluated as two independent, prospectively declared strata.
For each stratum, all exact members must pass every applicable gate. A passing
stratum may be admitted even if the other stratum fails; a failing stratum must
remain absent from `RAW_SUFFIXES`. This per-extension policy is frozen before
any U7.19A pixel decode and cannot be changed after observing results.

For every exact member of a stratum:

1. Verify source byte length and SHA-256 before and after execution.
2. Use rawpy 0.26.1 / LibRaw 0.22.0 with the unchanged generic path: camera
   white balance, no auto bright, 16-bit intermediate, explicit linear sRGB,
   gamma `(1, 1)`, and LibRaw orientation.
3. Require clean inspection and a nonempty, nonconstant HxWx3 float32
   `WorkingImage`; all values must be finite in `[0,1]`, storage must be owned,
   C-contiguous and writable, and the boundary must remain
   `linear_srgb` / `scene_linear`, applied orientation, absent alpha, 16-bit
   input, and the existing `generic_raw_render` plus
   `generic_raw_display_mapping` warnings.
4. Run fresh forward and reverse committed-head processes, canonicalize by
   source ID, and require exact decoded float32 hashes and exact scientific
   records across order.

Only after an extension stratum passes the preflight may that lower-case suffix
be added to `src.preprocess.raw_decode.RAW_SUFFIXES`. The public dispatcher must
then route both lower- and upper-case filenames through the generic RAW path.

## Frozen product-chain gate

For each preflight-passing extension, run the frozen representative named in
the config through `scripts/render_film.py` with explicit Ektar 100
`film-inspired / Look Approximation`, look amount `1.0`,
`safe-rich-product-v1`, PNG8, and strict recipe publication. Require:

- successful public CLI execution from the repo-relative entrypoint;
- RGB PNG with embedded sRGB ICC;
- recipe input/output/profile/software identities and explicit look fields;
- `Style-safe`, `film-inspired`, `look-approximation`, and
  `calibrated_reference_allowed=false` claim fields;
- strict replay output byte-identical to the original product output;
- no source mutation, network request, foreign-file overwrite, or owned
  scratch residue.

If the product-chain gate fails for a preflight-passing extension, that
extension is not admitted. The implementation must be narrowed back before the
formal result is committed.

## Stop rule and claim ceiling

Do not replace a source, split an exact stratum, change white balance,
auto-bright, output space, gamma, orientation, style, amount, profile, output
depth, warning boundary, decoder, runtime, or gate. Do not use the P318/P321
private callable, producer runner, or stored-code output as rescue. A failure
closes only its prospectively independent extension stratum; it does not weaken
the passing requirements for the other stratum.

A pass proves only that the public product dispatcher and existing generic
LibRaw Look Approximation chain work on the exact admitted source-locked files
under the frozen Windows/Python runtime. It does not prove general SRW/ARQ or
Samsung/Sony support, component-ARW handling, pixel-shift fusion, vendor-exact
colour, camera calibration, scene-to-display tone mapping, image quality,
stock response, calibrated film reproduction, or a new decoder. Every named
film output remains `film-inspired / Look Approximation`.
