# Halation Real-Photo Validation Tracker

> Created: 2026-06-06
>
> Branch: `research/physical-halation-v2p2-calibration`
>
> Status: planned
>
> Goal: add a license-aware, unpaired real-photo validation and display-level
> statistical calibration path for the V2.3 halation renderer.

## Scope

This tracker is separate from:

- `docs/HALATION_SYSTEM_SPEC.md`: current system specification and GUI guide.
- `docs/HALATION_GUI_READINESS_TRACKER.md`: strict validation, presets, schema,
  GUI outputs, and family evaluator readiness.
- `docs/PHYSICAL_HALATION_V2_TRACKER.md`: V2/V2.1/V2.2/V2.3 implementation
  history.

This tracker focuses only on:

```text
real-photo, unpaired, display-level halation validation
```

The goal is to compare the simulator's visible halation statistics against
license-safe or citation-safe real film photographs, not to claim a complete
stock-calibrated physical model.

## What Can Be Done Autonomously

Autonomous work is possible for:

1. Searching and auditing candidate public real-film photo sources.
2. Building a source manifest with URL, author, license, stock claim, and usage
   restrictions.
3. Downloading only assets whose license permits local analysis, or storing only
   URL metadata when downloading/redistribution is not allowed.
4. Mining high-light halation patches from real photos.
5. Computing unpaired halo patch statistics.
6. Comparing real-photo statistics with simulator outputs.
7. Tuning uncalibrated heuristic ranges based on display-level statistics.
8. Writing reports and contact sheets.

## What Cannot Be Honestly Claimed

This work cannot by itself claim:

1. True film-physics calibration.
2. Paired digital-to-film calibration.
3. Accurate Kodak/CineStill stock constants.
4. Accurate lab/scan pipeline reproduction.
5. RAW/HDR scene exposure recovery from ordinary web JPEGs.
6. General authenticity across all photos and film stocks.

The correct claim target is:

```text
The V2.3 halation renderer has been constrained and adjusted against
display-level, unpaired real-photo halo patch statistics from audited sources.
```

## Evidence Levels

Use the evidence discipline from `docs/HALATION_SYSTEM_SPEC.md`.

For this tracker:

| Item | Evidence Level |
|------|----------------|
| Source URL and license metadata | code fact / manifest fact after audit |
| Downloaded image pixels | license-dependent analysis input |
| Patch metrics extracted from real photos | measured project result |
| Comparison between simulator and real patches | measured project result |
| Any parameter adjustment from patch statistics | uncalibrated heuristic unless validated on held-out sources |
| Claim that a photo is a given stock | external-source-supported only if the source page states it |
| Claim that parameters are true stock constants | forbidden |

## Candidate Source Types

Candidate sources to audit:

| Source Type | Potential Use | Risk |
|-------------|---------------|------|
| Wikimedia Commons film-stock categories | license-audited real-photo candidates | stock metadata may be incomplete or user-supplied |
| Official Kodak pages / sample pages | behavioral reference and citation | images may not be reusable/downloadable |
| CineStill sample galleries / support pages | no-remjet behavior reference and citation | images may be copyrighted; may require URL-only manifest |
| Dehancer profile/example pages | behavior constraints and citation | images likely not reusable as dataset |
| Public-domain / CC BY / CC BY-SA film photos | local patch analysis if license allows | stock and scan process may be uncertain |
| User-provided future samples | best practical target later | requires user action and license/consent |

Initial known web references to consider:

```text
https://commons.wikimedia.org/wiki/Category:Photographs_taken_on_Kodak_Vision3_500T_film
https://www.kodak.com/en/motion/product/camera-films/500t-5219-7219
https://www.dehancer.com/profiles/film/cinestill-800t
https://help.cinestillfilm.com/hc/en-us/articles/360028874012-What-is-Remjet
```

These are starting points, not automatically approved data sources.

## License Policy

Before downloading or storing any real image in the repo workspace:

1. Record source URL.
2. Record author/creator if available.
3. Record license string.
4. Record whether local analysis is allowed.
5. Record whether redistribution/commit is allowed.
6. Record whether derivative contact sheets can be committed.
7. Prefer storing generated outputs under ignored `outputs/`, not committed
   docs, unless license clearly permits redistribution.

Recommended manifest fields:

```csv
source_id,url,page_url,author,title,license,license_url,stock_claim,
source_type,download_allowed,redistribution_allowed,derivative_allowed,
local_path,notes,audit_date
```

Default policy:

- commit manifests and scripts;
- do not commit downloaded photos;
- do not commit real-photo contact sheets unless license and project policy
  explicitly allow it.

## Patch Mining Targets

The patch miner should search for:

- point lights on dark backgrounds;
- neon signs;
- white text/signage on black or dark backgrounds;
- car headlights and taillights;
- bright bulbs;
- strong specular highlights;
- warm colored light blocks;
- high-contrast highlight edges.

Avoid or down-rank:

- white walls;
- bright sky;
- faces and skin highlights;
- low-resolution compression artifacts;
- large bloom/glare that may be lens or sensor behavior rather than film
  halation;
- heavily edited social-media images with unknown processing.

## Proposed Metrics

Real-photo patch metrics:

| Metric | Meaning |
|--------|---------|
| `source_luma_peak` | Estimated display-space source brightness. |
| `background_luma` | Surrounding local background brightness. |
| `visible_radius_px` | Estimated halo extent above background threshold. |
| `radius_normalized` | Radius normalized by patch/source scale. |
| `radial_falloff_slope` | Decay curve around source. |
| `red_green_ratio_inner` | Inner halo/core red-green balance. |
| `red_green_ratio_outer` | Outer halo red-green balance. |
| `blue_leakage` | Blue contribution in detected halo region. |
| `orange_core_score` | Whether core is warmer/yellower than outer halo. |
| `dark_side_ratio` | Halo strength on dark side vs bright side. |
| `halo_confidence` | Heuristic confidence that this patch contains halation. |

Simulator comparison metrics:

| Metric | Meaning |
|--------|---------|
| `distribution_distance_radius` | Distance between real and simulated visible-radius distributions. |
| `distribution_distance_hue` | Distance between real and simulated hue-ratio distributions. |
| `blue_leakage_delta` | Simulator vs real blue leakage. |
| `dark_side_ratio_delta` | Simulator vs real dark-background behavior. |
| `patch_acceptance_rate` | Fraction of mined patches passing confidence threshold. |

## Proposed Output Layout

All generated data should live under ignored `outputs/` unless explicitly
approved for commit:

```text
outputs/eval/halation_real_photo_v1/
  sources_manifest.csv
  downloaded/
  patches/
    real/
    simulated/
  metrics/
    real_patch_metrics.json
    simulated_patch_metrics.json
    alignment_summary.json
  contact_sheets/
    real_patches_contact_sheet.png
    simulated_patches_contact_sheet.png
    alignment_contact_sheet.png
  report.md
```

If license does not allow downloaded image storage, use:

```text
outputs/eval/halation_real_photo_v1/url_only_manifest.csv
```

and skip local pixel analysis for those entries.

## Dependency Order

```text
Order 1: Source/license audit plan
Order 2: Source manifest builder
Order 3: Download/cache policy implementation
Order 4: Real-photo patch miner
Order 5: Real patch metrics
Order 6: Simulator patch generation for comparison
Order 7: Alignment report and contact sheets
Order 8: Parameter suggestion pass
Order 9: Optional parameter update experiment
Order 10: Documentation and evidence-level update
```

## Task Board

| Order | Task | Status | Depends On | Completion Test |
|:---:|------|:---:|------------|-----------------|
| 1 | Source/license audit plan | planned | none | candidate source list has license policy and allowed actions |
| 2 | Source manifest builder | planned | Order 1 | writes manifest with URL/license/stock/use fields |
| 3 | Download/cache policy implementation | planned | Order 2 | only allowed assets are downloaded; URL-only entries are skipped |
| 4 | Real-photo patch miner | planned | Order 3 | produces candidate patches and confidence scores |
| 5 | Real patch metrics | planned | Order 4 | writes real patch metrics JSON |
| 6 | Simulator patch generation | planned | Order 5 | renders comparable simulator outputs for patch/source classes |
| 7 | Alignment report/contact sheets | planned | Order 5, 6 | writes JSON summary, report, and contact sheets |
| 8 | Parameter suggestion pass | planned | Order 7 | proposes heuristic range changes without modifying defaults |
| 9 | Optional parameter update experiment | planned | Order 8 | branch/flagged experiment produces before/after contact sheets |
| 10 | Evidence/documentation update | planned | Order 7-9 | spec/tracker records what was measured vs inferred |

## Order 1: Source/license Audit Plan

### Goal

Create a written, source-by-source plan before downloading anything.

### Steps

1. Search for candidate sources.
2. Record URL and source type.
3. Check license and terms.
4. Classify allowed actions:
   - cite only,
   - URL-only manifest,
   - local analysis allowed,
   - redistribution allowed,
   - derivative contact sheet allowed.
5. Record stock claim confidence:
   - official,
   - page-stated,
   - user-tagged,
   - inferred,
   - unknown.

### Completion Test

Create or update:

```text
docs/HALATION_REAL_PHOTO_VALIDATION_TRACKER.md
```

with audited source candidates, or create:

```text
outputs/eval/halation_real_photo_v1/source_audit.json
```

if the list is generated.

## Order 2: Source Manifest Builder

### Goal

Automate manifest creation from audited source entries.

Possible script:

```powershell
scripts\build_halation_real_photo_manifest.py
```

Required output:

```text
outputs/eval/halation_real_photo_v1/sources_manifest.csv
```

### Completion Test

Manifest validates:

- all required columns exist;
- every row has source URL;
- every row has license status;
- rows without download permission are marked URL-only.

## Order 3: Download/Cache Policy Implementation

### Goal

Download only what is allowed.

Possible script behavior:

```powershell
scripts\build_halation_real_photo_manifest.py --download-allowed
```

or separate:

```powershell
scripts\download_halation_real_photo_sources.py
```

### Completion Test

- Allowed entries have local paths.
- URL-only entries are not downloaded.
- Local files are under ignored `outputs/`.
- Manifest records actual local path and checksum.

## Order 4: Real-Photo Patch Miner

### Goal

Detect likely halation patches automatically.

Possible script:

```powershell
scripts\mine_halation_patches.py `
  --manifest outputs\eval\halation_real_photo_v1\sources_manifest.csv `
  --output-root outputs\eval\halation_real_photo_v1\patches\real
```

### Candidate Algorithm

1. Load image.
2. Convert to float RGB and luminance.
3. Detect bright source candidates using percentile threshold.
4. Reject large smooth bright areas.
5. Favor high local contrast and compact highlights.
6. Compute local background.
7. Estimate halo annulus.
8. Compute confidence from:
   - dark background,
   - red/orange excess,
   - compact source,
   - radial falloff quality.
9. Save patch and metadata.

### Completion Test

- Writes patch PNGs under ignored `outputs/`.
- Writes patch metadata JSON/CSV.
- Produces a contact sheet for manual review.

## Order 5: Real Patch Metrics

### Goal

Compute statistics from mined patches.

Possible script:

```powershell
scripts\measure_halation_real_patches.py `
  --patch-root outputs\eval\halation_real_photo_v1\patches\real `
  --output outputs\eval\halation_real_photo_v1\metrics\real_patch_metrics.json
```

### Completion Test

Metrics include:

- visible radius;
- red/green ratios;
- blue leakage;
- background luminance;
- radial falloff;
- patch confidence.

## Order 6: Simulator Patch Generation

### Goal

Generate comparable simulator outputs for the same source/background classes.

Options:

1. Use synthetic diagnostic patches matched to real patch source/background
   statistics.
2. Apply current simulator to source images if license permits local processing.
3. Generate simulator patches from public-domain/allowed originals.

### Completion Test

- Simulated patch metrics are written.
- Simulated patch contact sheet is generated.
- Each simulator run records halation controls and evidence level.

## Order 7: Alignment Report And Contact Sheets

### Goal

Compare real and simulated statistics without pretending the data is paired.

Output:

```text
outputs/eval/halation_real_photo_v1/metrics/alignment_summary.json
outputs/eval/halation_real_photo_v1/contact_sheets/alignment_contact_sheet.png
outputs/eval/halation_real_photo_v1/report.md
```

### Completion Test

Report includes:

- source/license summary;
- number of audited sources;
- number of downloaded/analyzed sources;
- number of mined patches;
- metric distribution tables;
- simulator-vs-real comparison;
- limitations;
- no claim of true stock calibration.

## Order 8: Parameter Suggestion Pass

### Goal

Suggest changes to heuristic ranges without changing defaults automatically.

Examples:

```text
cinestill_no_remjet source_selectivity range may be too permissive
amber_core hue_green may be too strong/weak
classic_dense_base diffusion may be too broad
bw density tint may be too warm
```

### Completion Test

Write:

```text
outputs/eval/halation_real_photo_v1/parameter_suggestions.json
```

and summarize in report.

## Order 9: Optional Parameter Update Experiment

### Goal

If suggestions are strong, run an isolated experiment.

Rules:

- do not silently overwrite V2.3 defaults;
- use a branch, flag, or explicit experimental preset suffix;
- produce before/after contact sheets.

Completion output:

```text
outputs/eval/halation_real_photo_v1/parameter_update_experiment/
```

## Order 10: Evidence/Documentation Update

### Goal

Update docs only after measured results exist.

Potential docs:

- `docs/HALATION_SYSTEM_SPEC.md`
- `docs/PHYSICAL_HALATION_V2_TRACKER.md`
- this tracker
- new result doc if outputs are substantial:
  `docs/HALATION_REAL_PHOTO_VALIDATION_RESULTS.md`

## Risk Register

| Risk | Mitigation |
|------|------------|
| Copyrighted sample images accidentally committed | Keep downloads under ignored `outputs/`; commit only manifests/scripts unless license explicitly allows redistribution. |
| Stock label is unreliable | Track `stock_claim_confidence`; do not treat user tags as measured truth. |
| Scanner/lab/postprocessing dominates halation look | Label all results display-level and unpaired. |
| Patch miner detects bloom/lens flare instead of film halation | Use confidence scoring and manual contact sheets. |
| Calibration overfits small web sample | Use parameter suggestions first; require held-out validation before changing defaults. |
| JPEG compression affects hue/radius metrics | Track resolution/compression notes and confidence. |

## Current Manual Requirements

None for Orders 1-8. They can be pursued autonomously with license caution.

Manual/user input is needed for:

- approving final default changes;
- providing proprietary or personal film scans;
- deciding whether derivative real-photo contact sheets may be committed;
- accepting that results are display-level, not physical stock constants.
