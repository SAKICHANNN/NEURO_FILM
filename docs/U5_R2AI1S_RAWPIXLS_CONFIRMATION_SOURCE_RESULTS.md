# U5.R2AI1S independent RAW confirmation-source results

Status: **complete / pass with one frozen-rule exclusion and no replacement**

Node: `ULT > U5 > U5.R2 > U5.R2AI1S`

## Result

The exact bounded acquisition completed at 18/18 files and 271,651,338 bytes.
Every frozen RAW SHA-256 and decode passed. Two independent executions over
the cached RAW files produced byte-identical evidence:

| Artifact | SHA-256 |
|---|---|
| manifest | `8ac0fa47b8ad3058c7bd3f9a398dfebe657b681cbd4a154152f0dbb255c4ecf1` |
| automatic report | `4cd54cc3f853a43b3afd08734f8a3078f1e76e1d2fcee26b63240b915d26c6de` |
| source contact sheet | `1592ab58d18463ec26775c262a486220fbe344c859ccf9cad4df903580ec9545` |
| autonomous visual review | `ee493ada0dff0e92ab32fba371b738600a26ad3a215285653fd26f93a4e2f73e` |

The automatic population gates all pass:

- 18 decoded rows across nine camera makes;
- 11.11% largest make share;
- zero exact or dHash <= 4 duplicates within the pool;
- zero exact or dHash <= 4 overlap against all 41 R2AI0 development inputs;
- zero near-empty or effectively monochrome previews;
- zero download, hash or decode failures.

No colour operator was rendered during this leaf.

## Frozen selection-rule exclusion

Autonomous visual review discovered that `fujifilm_s2pro` is a visible Kodak
grey-scale and colour-control target. The source was not known from the
repository row before decoding, but the contract had already excluded
target/chart content.

The row is therefore retained in the acquisition and audit evidence but
excluded from the U5.R2AI1 confirmation population. It is not called a severe
artifact and is not replaced. The exact confirmation population is 17 rows
across all nine camera makes, with an 11.76% largest make share.

## Visual source audit

The 17 eligible rows cover eleven recorded content/safety buckets, including
people and skin, text and fine detail, architecture and geometry,
foliage/landscape/wildlife, sky and clouds, water/glass/specular detail,
mixed or warm indoor light, deep shadow and backlight, shallow-depth still
life, saturated hues and texture, and snow/near-neutral highlights.

All 18 decoded previews were reviewed, with five full-resolution spot checks.
There are zero confirmed severe source failures. Strong tungsten warmth,
shallow depth of field, window reflection and hard snow highlights are
preserved as legitimate source conditions rather than hidden as defects.

## Decision

U5.R2AI1S passes at 17/18 after the preregistered target/chart exclusion.
This opens only a separately frozen U5.R2AI1 confirmation of the one unchanged
R2AI0 retained composition over all 17 eligible rows.

The operator order, density strength, final margin, comparators, automatic
thresholds, blind-review procedure and severe-artifact rule must be frozen
before any operator output is rendered. Failed rows may not be replaced and
no parameter or gate may be retuned after confirmation results are visible.

## Verification and boundaries

- five focused implementation tests pass;
- `compileall` passes;
- the complete local CPU suite passes at **1046 tests**;
- both source executions are byte-identical;
- the decoder records rawpy 0.26.1, camera white balance, gamma, resize,
  encoding and exact source/decoded hashes;
- author, uploader, roll, process, lab and scanner remain `unknown`; no
  appearance inference backfills them.

This is bounded CC0 digital-photo source/OOD feasibility evidence only. It is
not an algorithm result, real-film or stock evidence, preference evidence,
operator identification, calibration, authenticity, production promotion or
universal-safety evidence.

Authority:
`configs/u5_r2ai1s_rawpixls_confirmation_source_preflight_decision_v1.json`.
