# Reference-match portable conformance v1

Date: 2026-07-27

Status: frozen reference implementation vectors pass; independent platform
ports are not yet tested.

## Purpose

Python repeat-exactness is necessary but cannot prove that Android, iOS,
macOS and Windows implementations execute the same colour operator. This
contract supplies a language-neutral executable boundary for ports of
`safe-lab-reference-statistics.v1`.

It validates algorithm mechanics only. The baseline remains rejected as the
final photographic matcher and defaults to identity delivery until A1, A4 and
A5 pass.

## Frozen artifacts

- bundle:
  `configs/reference_match_portable_conformance_v1.json`
- bundle schema:
  `configs/schemas/reference_match_portable_conformance_v1.schema.json`
- result schema:
  `configs/schemas/reference_match_portable_conformance_result_v1.schema.json`
- verifier:
  `scripts/verify_reference_match_portable_conformance.py`
- reference implementation:
  `src/color_match/conformance.py`

Bundle identity:
`ed064c761e729a29bc7d5dda4180de016d2ed8aa68f66a781172e74012efa1ed`.

## Wire and execution contract

1. Decode reference, source and expected output arrays from
   `ieee754-binary32-big-endian-hex` in row-major HxWx3 order.
2. Recompute `fixture_id` from the typed canonical v1 payload after removing
   only `fixture_id`; any drift fails before execution.
3. Fit one recipe from exactly the case reference and frozen policy.
4. Require exact `recipe_id` and `reference_pixel_sha256` agreement.
5. Apply that recipe to the source without hidden clipping or source-batch
   adaptation.
6. Compare the port output to the frozen output in the declared source
   working space.
7. Require all numeric gates:
   - linear-RGB maximum absolute error <= `2e-5`;
   - Delta E76 p95 <= `0.003`;
   - Delta E76 maximum <= `0.01`;
   - maximum absolute diagnostic error <= `2e-5`.

The exact identity gates prevent a port from silently changing recipe state.
The bounded pixel gates allow small differences between correctly implemented
math libraries without weakening perceptual or boundary agreement.

## Coverage

| Case | Working space | Gamut policy | Guarded path |
|---|---|---|---|
| `srgb-default-detail` | linear sRGB | source-to-target segment | default detail, neutral and skin guards; 1/6 pixels adjusted |
| `rec2020-chroma-compression` | linear Rec.2020 | fixed-L/hue chroma compression | tone rolloff and stronger detail/chroma guards; 12/12 pixels adjusted |

Both cases use non-uniform 3x4 images so full-image reductions and the
finite-support detail path execute. They are conformance vectors, not an
aesthetic benchmark or evidence of reference-look identifiability.

## Verification

Run:

```powershell
C:\Users\hhvrf\Documents\neuro_film\.venv\Scripts\python.exe scripts\verify_reference_match_portable_conformance.py
```

Current reference result:

- two of two cases pass;
- all exact identities match;
- reference implementation errors are zero against the frozen bits;
- dedicated tests: 9 passed;
- focused colour-match/preprocess tests: 147 passed;
- complete suite: 1019 passed, one skipped, 36 unchanged known failures.

The 36 failures are pre-existing ignored-output and checked-out-byte/CRLF hash
classes in this isolated worktree. No colour-match test fails.

## Port acceptance and change rules

- A platform port may claim conformance only after emitting a result matching
  the result schema with every case passed.
- Four-platform product conformance requires separately recorded Android,
  iOS, macOS and Windows reports; the current Python report is not a
  substitute.
- A promoted replacement algorithm receives a new `algorithm_id`, bundle
  schema/version and fixture identity. It must not overwrite v1.
- Relaxing a tolerance requires a preregistered numerical cause and
  full-resolution photographic boundary evidence; it is not a routine porting
  convenience.
- RAW/HDR decoding remains outside this bundle. A future MatchView adapter
  must first establish the explicit scene/display render boundary under A3.
