# U1.6G4F Explicit New-Operator Audit Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4F`

**Decision:** retain as an explicit opt-in research candidate; integration remains closed

## Reproducibility

- frozen audit config SHA-256: `c36bc840c5acdc603800abc33a12ede59ebf82e107aa352073ec88cfa7773133`;
- audit implementation commit: `b90b0a1`;
- 1:1 visual fix commit: `1220d8d46568dcd5a99cc2fef3299fd772330a3d`;
- ignored reports: `outputs/u1_6g4f/formal_1220d8d_{a,b}.json`;
- report hashes: `6fd13907...` and `68be3bb4...` (runtime observations differ);
- deterministic payload SHA-256, identical twice: `5f49cd5f82ce23bc01a05aada0f7a6417d4a47acf2a032e8629619d942913d71`;
- contact-sheet SHA-256, byte-identical twice: `76304ade3b6f935a31586f2345b22714a34a9f850560f04b202ecac39e18ddbf`;
- autonomous visual-decision config SHA-256: `72be2db7b0b5ad9fa425eb4fde4b544e5b775b5ff60276a3b2575352e6066bc1`.

The first generated sheet incorrectly downscaled panels labelled 1:1. It was
not used for the final visual decision. Commit `1220d8d` fixes this and adds a
test that preserves one source pixel per output pixel. Both formal audits were
then rerun.

## Automatic result

All three fresh confirmatory cases pass every frozen automatic gate:

| Metric | Frozen gate | Observed range |
|---|---:|---:|
| staged-v2 alpha max error | `<=2e-6` | `<=1.02e-7` |
| staged-v2 composite max error | `<=2e-6` | `<=8.95e-8` |
| seam max error | `<=2e-6` | `<=5.97e-8` |
| changed uint8 channel fraction | `>=0.04` | `0.1006-0.1741` |
| composite absolute p99 | `>=0.0035` | `0.00612-0.01060` |
| maximum uint8 code delta | `>=3` | `10-13` |
| alpha-active fraction | `>=0.05` | `0.4509-1.0` |
| highlight selectivity | `>=3x` | `3.65x-18.41x` |
| new high/low clipping | each `<=1e-4` | all zero |
| median local seconds/MP | `<=2.5` | `1.14-1.33` across both runs |

Both tile sizes are rounded-sRGB8 identical to materialized v2. Deterministic
pixel/metric payloads match across complete audits. Legacy composite drift is
reported only as a diagnostic and remains at most `1.69e-4`; this does not
rewrite or rescue G4E.

## Autonomous visual result

The corrected sheet and isolated original-resolution crop comparisons show:

- `u41-07`: safe, but the unamplified effect is weak;
- `u41-12`: restrained localized response around the bright windmill blades;
- `u41-16`: restrained localized response around high-reflectance chart patches.

Thus exactly two of three cases meet the frozen unamplified localized-effect
count. All three have zero confirmed seam, band, posterization, colour block,
clipping expansion, objectionable broad veil or other severe artifact.

The visual pass means the effect is non-empty and plausible as a localized
optical layer. It is not evidence of a strong global film look or population
preference.

## Verification

- 21 focused implementation/audit tests pass after the 1:1 correction;
- 79 focused and adjacent tests passed before the formal audit;
- 532 complete CPU tests pass in 21.90 seconds;
- renderer, current effects entry point, profiles, recipes and CLI are unchanged.

## Branch decision

Retain `staged-density-halation-v1-defaults` as a separately versioned,
research-only candidate. Do not call it legacy-compatible and do not connect or
default it. The next legal leaf is a separately frozen 24MP measured
memory/runtime and orchestration-readiness audit. Only that evidence may decide
whether an integration leaf is worth opening.

## Claim ceiling

G4F supports a deterministic, bounded, opt-in density-halation research
candidate on three fixed real cases. It does not establish strong global film
style, default/product promotion, physical calibration, stock response,
colour-family execution, streaming decode, total 100MP memory or renderer
integration.
