# P95 — Adobe DNG Validate bounded runtime contract

Status: frozen before formal `dng_validate` image execution.

## Question

Does an exact source-bound build of Adobe DNG SDK 1.7.1 build 2652 parse and
render the five P94 DNGs repeatably through its stage-2, stage-3, and final
sRGB16 paths on this Windows host?

This is a mature RAW/DNG runtime-conformance leaf.  It is not an image-quality
comparison, a learned model, an after-only reference matcher, or a product
admission experiment.

## Frozen binary and build provenance

- SDK archive SHA-256:
  `73499b47f4683e12120a234bd0946f02e52ab2ff9834bcbd0e9f8ab4f923360e`.
- `dng_validate.exe` SHA-256:
  `3ac9a60e8989088721c272dfefa5e7086ddcf17bb5ee381624fd84d6337a7d13`.
- The exact official Visual Studio solution was built as `Validate Release|x64`
  with the command-line-only property override `PlatformToolset=v145`.  No SDK
  source or project file was edited.
- MSBuild SHA-256:
  `81263dd8ec9f60137708bfe6fcb301773b881eb21a6831870a2ba025e02140b7`.
- MSVC `cl.exe` SHA-256:
  `ed1a6206c8179ed6826187b63306cb9cc421de6507e14166864556851bc2820d`,
  reported compiler version `19.50.35729` for x64.

The first two build attempts failed before producing a binary because the
published solution requested unavailable ClangCL/v100 toolsets.  The successful
retarget is infrastructure provenance, not scientific evidence.

## Frozen rows and invocation

All five exact P94 rows are mandatory.  Each invocation uses only:

```text
dng_validate.exe -size 1024 -16 -cs1
  -2 <row-stage2> -3 <row-stage3> -tif <row-final> <exact-input.dng>
```

Run A enumerates rows in canonical source-id order.  A fresh process Run B
enumerates the same rows in exact reverse order.  Output names are canonical
and independent of enumeration order.

## Frozen gates

- exact input size/SHA and executable SHA before every run;
- five successful process exits per outer run;
- no SDK validation error line and no uncaught exception;
- every row emits stage2, stage3, and final TIFF;
- every TIFF decodes through `tifffile`, is finite, non-empty, and has the
  expected component count: stage2 one component, stage3/final three;
- final output is unsigned 16-bit RGB and its long side is at most 1024;
- each row's Run A and Run B stage2/stage3/final file SHA-256 values are exact;
- canonical scientific reports from two fresh runner processes are byte exact;
- source DNG hashes remain unchanged after execution.

Any tool, source, validation, output, or replay failure is fail-closed.  There
is no SDK-source edit, render-option, row, resolution, warning, or tolerance
rescue.

## Claim ceiling

PASS means only that the exact retargeted official Adobe tool has a repeatable,
bounded Windows x64 runtime on these five already-consumed CC0 DNGs.  It does
not establish Adobe-binary identity, arbitrary DNG support, photographic or
colorimetric quality, equality with P94, sensor/IDT calibration, project-loader
integration, ACES/HDR output, public package/schema/capability, product, film,
stock, or preference admission.

This product includes DNG technology under license by Adobe.

