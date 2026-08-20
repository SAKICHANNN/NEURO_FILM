# U1.3C DNG capture-metadata receipt contract

Date frozen: 2026-08-21  
Node: `ULT > U1 > U1.3C`  
Status: frozen before implementation and formal extraction

## Question

Can the project preserve standardized, capture-time DNG facts as one
deterministic, source-bound private receipt without decoding image pixels or
changing the renderer?

This is a deliberately narrow RAW/DNG engineering leaf. It supplies factual
input observations for future paired/capture-time explicit operators. It does
not estimate a look, render RGB, identify film, or authorize a product path.

## Standards and implementation boundary

- DNG is treated as the TIFF-based public format described by Adobe's current
  DNG specification and SDK page:
  <https://helpx.adobe.com/camera-raw/digital-negative.html>.
- Parsing uses the project's existing `tifffile` dependency. The implementation
  may traverse TIFF IFD and SubIFD metadata, but must not call `asarray`,
  `imread`, LibRaw postprocessing, or otherwise decode raster samples.
- Only `.dng` inputs are accepted. Unsupported or ambiguous files fail closed.
- No public recipe, receipt, package, capability, renderer, `WorkingImage`, or
  consumer schema changes are permitted in this leaf.

## Fixed real-file cohort

All four rows are mandatory. Their bytes and CC0/Public Domain source facts
were frozen previously in the two exact raw.pixls.us source manifests.

| id | logical path | source manifest |
|---|---|---|
| `dji_fc4382` | `data/external/rawpixls_bh0_v1/raw/dji_fc4382.dng` | `configs/u5_r2bh0s_fixed_bank_oracle_source_preflight_v1.json` |
| `google_pixel_7_pro` | `data/external/rawpixls_bh0_v1/raw/google_pixel_7_pro.dng` | `configs/u5_r2bh0s_fixed_bank_oracle_source_preflight_v1.json` |
| `apple_iphone_12_pro` | `data/external/rawpixls_bh1_v1/raw/apple_iphone_12_pro.dng` | `configs/u5_r2bh1s_global_policy_source_preflight_v1.json` |
| `huawei_eml_l29` | `data/external/rawpixls_bh1_v1/raw/huawei_eml_l29.dng` | `configs/u5_r2bh1s_global_policy_source_preflight_v1.json` |

The cohort intentionally covers top-level and SubIFD raw planes, CFA and
LinearRaw photometric interpretations, multiple orientations, integer and
rational black/white/crop values, and float NoiseProfile values.

## Receipt v1

The canonical receipt is UTF-8 JSON with sorted keys and compact separators.
It contains:

1. schema/version and caller-supplied logical source id/path;
2. exact source byte count and SHA-256;
3. selected raw-IFD path and all traversed IFD structural summaries;
4. DNG version/backward version, unique camera model and orientation;
5. raw geometry, bits/sample and photometric interpretation;
6. when present, CFA repeat/pattern/plane/layout;
7. black-level repeat/value, white level, active area, default scale/crop;
8. ColorMatrix1/2 and AsShotNeutral;
9. NoiseProfile with an explicit presence flag; and
10. a SHA-256 over the canonical receipt body excluding the digest field.

TIFF integer values remain integers. BYTE/UNDEFINED arrays remain exact byte
arrays. RATIONAL/SRATIONAL values remain exact numerator/denominator pairs;
derived finite decimal values may be included only in addition to those exact
pairs. FLOAT/DOUBLE values must be finite and use their parsed numeric values.

Global DNG tags may reside on a root/preview IFD while raw-plane tags reside on
the selected raw IFD. The extractor uses raw-local tags first and otherwise
requires one unambiguous value across ancestor/root IFDs.

## Raw-IFD selection

An eligible raw IFD must:

- have PhotometricInterpretation `CFA` (32803) or `LinearRaw` (34892);
- not be marked reduced-resolution, transparency mask, or semantic mask; and
- expose positive image width and height.

Exactly one eligible raw IFD is required. Zero or multiple eligible IFDs is a
structured failure, not a heuristic choice.

## Formal gates

All must pass:

1. all four actual source hashes equal their frozen manifest hashes;
2. all four receipts build with exactly one eligible raw IFD;
3. required facts are present: DNGVersion, UniqueCameraModel, Orientation,
   raw geometry, BitsPerSample, PhotometricInterpretation, BlackLevel,
   WhiteLevel and AsShotNeutral;
4. CFA rows expose a complete CFA pattern; the LinearRaw row is explicitly
   classified and does not fabricate CFA fields;
5. all numeric values and derived rationals are finite and denominators are
   non-zero;
6. raster decode call count is exactly zero;
7. forward and reverse source enumeration produce the same per-source receipt
   bytes and the same normalized report bytes; and
8. focused malformed/ambiguous/non-DNG tests fail closed without a partial
   receipt.

## Stop rules and claim ceiling

- Any missing mandatory row, source-hash mismatch, ambiguous raw IFD, required
  field failure, non-finite value, pixel decode, or replay mismatch closes v1.
- Do not repair a failed real row with ExifTool, LibRaw, vendor SDK fallback,
  tag guessing, default values, or row replacement.
- A pass establishes only a private deterministic DNG capture-metadata receipt
  over this four-device cohort. It does not establish vendor-render parity,
  calibrated exposure/noise, arbitrary-DNG support, a public schema, a product
  capability, or any film/stock claim.
