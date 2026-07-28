# U5.R2AI1S Independent RAW Confirmation Source Preflight

## Purpose

U5.R2AI0 passed on a 41-image development population. U5.R2AI1S creates an
independent digital-photo population before the retained operator is applied.
It is a source and integrity leaf, not an algorithm result.

The local unused RAW cache is not sufficient: its 18 unique unused files are
all Canon. The frozen acquisition instead selects 18 exact raw.pixls.us CC0
rows across nine makes, two rows per make, with no raw hash or camera-model
overlap against the 20 RAW development inputs.

## Frozen acquisition

- exact source: raw.pixls.us repository/API as observed 2026-07-28;
- 18 exact URLs and SHA-256 values;
- reported total approximately 259.07 MB, hard cap 270 MiB;
- camera makes: Canon, Nikon, Sony, Fujifilm, Olympus, Panasonic, Pentax,
  Ricoh and Samsung;
- 5-18 MB rows only;
- explicit exclusion of target/chart, cinema, series and already cached or
  development-model rows.

No row may be replaced after its image is seen. Hash or decode failures remain
in the evidence.

## Decode and integrity contract

The preflight may:

1. download only the frozen 18 files;
2. verify every exact RAW SHA-256;
3. generate a bounded neutral camera-WB display preview using the existing
   raw.pixls path;
4. record dimensions, file hash, rendered hash, dHash and simple colour-state
   diagnostics;
5. compare exact and dHash<=4 duplicates within the pool and against all 41
   R2AI0 development inputs;
6. create a contact sheet for autonomous source-content and severe-input
   review.

Automatic pass requires at least 16 valid rows, eight camera makes, largest
make share no more than 25%, zero exact/dHash<=4 overlap within and across
pools, and no near-empty or effectively monochrome preview. The visual review
must cover at least six content/safety buckets and must preserve any source
corruption rather than silently replacing it.

## Boundaries

The R2AI0 operator is forbidden until this preflight has a frozen decision.
The source is digital-photo/OOD safety evidence only. It is not film, stock,
operator-identification, calibration, authenticity or preference evidence.
No fitting, training, routing, parameter change or production integration is
allowed.

Parent:
`configs/u5_r2ai0_dual_champion_global_composition_decision_v1.json`.

Authority:
`configs/u5_r2ai1s_rawpixls_confirmation_source_preflight_v1.json`.
