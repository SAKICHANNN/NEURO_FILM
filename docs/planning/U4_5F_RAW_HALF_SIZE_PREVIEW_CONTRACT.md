# U4.5F — RAW half-size three-stock preview contract

## Question

Can the existing private 1 MP three-stock preview path reduce RAW decode memory
and latency by requesting LibRaw's native `half_size` demosaic, while retaining
the unchanged full RAW decode as the reference and remaining inside the frozen
U7.3G preview-fidelity and boundary gates?

This is a product-preview performance question. It is not a new RAW renderer,
camera-colour result, stock-calibration result, or adjacent RAW/HDR wrapper.

## Current facts and pre-protocol feasibility

- U4.5A localized the preview memory floor to full-source decode. U4.5B solved
  the 1 MP JPEG case with native decoder scaling; U4.5D closed the 12 MP JPEG
  draft route. U4.5C/E cover admitted warm-cache lookup, not fresh RAW render.
- The unchanged RAW ingress expands a full-resolution uint16 LibRaw image and
  then a full-resolution float32 `WorkingImage` before the preview resize.
- Before this contract was written, a non-formal, input-only feasibility smoke
  used all five already-retained P98 DNGs. `half_size=True` reduced LibRaw
  postprocess time by 2.63x--5.48x. After area resize to the exact 1 MP preview
  geometry, input-level RGB RMSE was .00246--.00687, p95 absolute error was
  .00455--.01484, and new-boundary fraction was at most 7.67e-6. These rows are
  therefore development/product-mechanics rows, not hidden confirmation data.
- Formal output-level metrics, process-tree RSS, wall time, replay and negative
  controls have not been observed before this freeze.

## Frozen scope

- Reuse the exact five P98 rights-cleared DNG files and rawpy 0.26.1 / LibRaw
  0.22.0. No new download or source substitution is allowed.
- Add one private `load_raw_preview_working_image` entry point that differs from
  the existing RAW loader only by `half_size=True`.
- Add one opt-in `raw_half_size_decode` path to the existing direct three-stock
  preview and CLI. The default remains full decode.
- The opt-in is RAW-only, 1 MP-only, and valid only when the half-size decoded
  geometry is at least the requested preview geometry. It must never upsample.
- `raw_half_size_decode` and `jpeg_scaled_decode` are mutually exclusive.
- Preserve the existing Look Approximation profile, statistics, guardrails,
  look amount, seed, tile geometry, PNG encoding and three stock identities.
- Compare each half-size candidate against the same source rendered through the
  unchanged full RAW decode and exact final preview geometry.

## Frozen gates

1. All five source hashes, dimensions and P98 provenance bindings match.
2. Each candidate decodes before float expansion at dimensions smaller than
   source and no smaller than the requested preview dimensions.
3. Every source emits three distinct stock previews at the exact deterministic
   preview geometry.
4. Across all 15 candidate/reference output pairs: RGB RMSE <= .03, p95 absolute
   error <= .08, and new-boundary fraction <= .001 (unchanged U7.3G gates).
5. Candidate peak process-tree RSS is <= 512 MiB and <= 75% of the paired full
   decode peak for every source.
6. Candidate wall time is <= 15 seconds per source, median paired wall ratio is
   <= .95, and the worst paired wall ratio is <= 1.05.
7. Two from-zero committed-head reports, with reversed source enumeration,
   reproduce the timing-excluded scientific payload and every candidate output
   hash exactly.
8. Full/default RAW decode, JPEG scaled decode, and invalid/mutually-exclusive/
   undersized half-size requests retain their existing or fail-closed behavior.
9. Sources are immutable and owned scratch/output residue is zero.

Any gate failure closes this exact mechanism without source, decoder, resize,
threshold, stock, profile, output-format, or measurement rescue.

## Claim ceiling

Private Windows/Python 1 MP preview mechanics on five exact P98 DNG files for
three existing deterministic Look Approximations only. No arbitrary RAW,
vendor/Adobe parity, calibrated camera colour, scene-to-display quality, final
export, stock response, public API/package, device, release, or product-admission
claim is permitted.
