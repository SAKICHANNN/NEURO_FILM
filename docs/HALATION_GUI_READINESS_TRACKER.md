# Halation GUI Readiness Tracker

> Created: 2026-06-06
>
> Branch: `research/physical-halation-v2p2-calibration`
>
> Status: Orders 1-5 implemented on 2026-06-06
>
> Goal: turn the V2.3 halation research implementation into a safer,
> GUI-ready subsystem with strict validation, presets, schema export,
> richer layer outputs, and family-specific evaluation.

## Context

The current halation system has:

- a detailed architecture and GUI integration specification:
  `docs/HALATION_SYSTEM_SPEC.md`;
- a V2/V2.1/V2.2/V2.3 implementation tracker:
  `docs/PHYSICAL_HALATION_V2_TRACKER.md`;
- locked physical controls in `src/filmfx/halation_controls.py`;
- two rule families:
  - `color_negative_backscatter`,
  - `bw_density_halation`;
- integrated CLI entry via `scripts/render_film.py`;
- V2.3 contact sheets under:
  `outputs/eval/halation_v2p3_families/`.

The system is now useful, but not yet fully GUI-ready. The remaining autonomous
work is mostly safety, schema, presets, outputs, and evaluator coverage.

## Non-Goals

These are not part of this autonomous tracker unless explicitly restarted later:

1. Real-film patch calibration.
2. Full negative-density / dye-density / scan pipeline.
3. A production GUI app if the repository has no GUI surface to modify.
4. Video temporal consistency.
5. Cloud training or neural residual halation.

These may be future work, but they require additional product/data decisions.

## Evidence Discipline

Every task should preserve the evidence levels defined in
`docs/HALATION_SYSTEM_SPEC.md`:

- code fact,
- measured project result,
- external-source-supported claim,
- physically motivated model assumption,
- uncalibrated heuristic,
- future ideal.

Do not present heuristic preset values as measured stock constants.

## Dependency Order

Recommended execution order:

```text
Order 1: strict validation
Order 2: preset system
Order 3: GUI schema export
Order 4: renderer black/white layer outputs
Order 5: additional tests
Order 6: family-specific evaluator
Order 7: optional source_linear sidecar
Order 8: optional arbitrary contact-sheet helper
Order 9: richer metrics
Order 10: evidence metadata propagation
Order 11+: higher-risk visual/physics changes
```

The first five tasks are the highest-value GUI-readiness work.

## Task Board

| Order | Task | Status | Depends On | Completion Test |
|:---:|------|:---:|------------|-----------------|
| 1 | Strict validation layer | done | V2.3 controls | invalid GUI combinations fail in strict mode and existing compatible paths can still opt into fallback |
| 2 | Halation preset system | done | Order 1 | named presets resolve to `PhysicalHalationControls` and are covered by tests |
| 3 | GUI schema / contract JSON | done | Order 1, 2 | schema describes valid types, responses, slider ranges, defaults, visibility rules, evidence levels |
| 4 | Integrated renderer black/white layer outputs | done | Order 1 | `--write-layers` exports default layer, layer-on-black, and layer-on-white previews |
| 5 | More halation control tests | done | Order 1-4 | pytest covers strict validation, presets, schema, output metadata, and family invariants |
| 6 | Family-specific evaluator | planned | Order 2, 5 | new or extended evaluator checks family and color-response behavior across diagnostic scenes |
| 7 | Source-linear sidecar support | planned | Order 1 | integrated renderer can accept `--halation-source-linear-npy` for RAW/HDR/synthetic exposure input |
| 8 | Arbitrary halation contact-sheet helper | planned | Order 4 | script can build four-column sheets from existing render/layer outputs |
| 9 | Richer halation metrics | planned | Order 6 | metrics include red/green ratio, blue leakage, tint behavior, and dark/background visibility summaries |
| 10 | Evidence metadata propagation | planned | Order 2, 3, 9 | presets/schema/metrics report evidence level for values and outputs |
| 11 | Directional dark-side prototype | deferred | Order 6, visual approval | prototype outputs side-by-side sheets and can be reverted easily |
| 12 | Source-map improvement prototype | deferred | Order 6, visual approval | new source map improves diagnostics without increasing false positives on white walls/skin |
| 13 | Performance preview optimization | deferred | after GUI path exists | reduced-resolution preview or cache path improves latency without changing full-res output |

## Order 1: Strict Validation Layer

Status: done on 2026-06-06.

Implemented:

- `validate_physical_halation_controls(controls, strict=True)`.
- `resolve_physical_halation_controls(..., strict=True)`.
- `build_physical_halation_layer(..., strict=True)`.
- Strict mode rejects GUI-invalid type/color-response combinations.
- Compatibility mode keeps the B&W fallback tint behavior with `strict=False`.

### Problem

`docs/HALATION_SYSTEM_SPEC.md` states GUI-level valid combinations, but the
Python resolver currently tolerates some invalid B&W color-response combinations
with fallback tint behavior.

### Desired Behavior

Add a strict validation path:

```text
strict=True:
  invalid family/type/color-response combinations raise ValueError

strict=False:
  preserve current compatibility/fallback behavior where needed
```

Candidate API:

```python
validate_physical_halation_controls(controls, strict=True)
resolve_physical_halation_controls(controls, strict=True)
build_physical_halation_layer(base_rgb, controls, strict=True)
```

The exact API can be adjusted if a cleaner repo-local pattern appears.

### Strict Invalid Combinations

Normal GUI-invalid examples:

```text
type=bw_clear_base + color_response=amber_core
type=bw_clear_base + color_response=deep_red
type=bw_clear_base + color_response=red_orange_core
type=cinestill_no_remjet + color_response=neutral_density
type=vision3_ahu + color_response=warm_neutral_density
type=classic_dense_base + color_response=neutral_density
```

### Completion Tests

- Unit test: valid color-negative combinations pass.
- Unit test: valid B&W combinations pass.
- Unit test: invalid combinations raise in strict mode.
- Unit test: compatibility mode preserves documented fallback behavior if kept.
- Docs update: `HALATION_SYSTEM_SPEC.md` reflects actual strict API.

## Order 2: Halation Preset System

Status: done on 2026-06-06.

Implemented presets:

```text
vision3_clean
vision3_push
cinestill_balanced
cinestill_strong
cinestill_amber
classic_soft
bw_neutral
bw_warm
```

Public helpers:

```python
get_halation_preset(...)
list_halation_presets()
HALATION_PRESETS
```

CLI:

```powershell
--halation-preset cinestill_amber
```

### Problem

GUI should not have to hard-code all recommended defaults from the spec.

### Desired Behavior

Add named presets in code, likely in `src/filmfx/halation_controls.py` or a new
small module:

```text
vision3_clean
vision3_push
cinestill_balanced
cinestill_strong
cinestill_amber
classic_soft
bw_neutral
bw_warm
```

Each preset should contain:

- internal preset id,
- user-facing label,
- `PhysicalHalationControls`,
- short description,
- evidence level,
- recommended preview target if useful.

### CLI

Possible CLI:

```powershell
--halation-preset cinestill_amber
```

Preset should fill defaults but explicit CLI flags should be able to override
individual values.

### Completion Tests

- Every preset resolves in strict mode.
- Every preset builds a layer on a small synthetic image.
- CLI preset smoke test writes metrics with preset id.
- Presets are listed in GUI schema from Order 3.

## Order 3: GUI Schema / Contract JSON

Status: done on 2026-06-06.

Implemented:

- `halation_gui_schema()` in `src/filmfx/halation_controls.py`.
- `scripts/export_halation_gui_schema.py`.

Smoke output:

```text
outputs/schema/halation_gui_schema.json
```

### Problem

The GUI needs a machine-readable contract for:

- valid values,
- labels,
- tooltips,
- slider ranges,
- default values,
- visibility rules,
- valid combinations,
- presets,
- evidence levels.

### Desired Behavior

Add a schema generator, either as a function and/or script:

```powershell
.\.venv\Scripts\python.exe scripts\export_halation_gui_schema.py `
  --output outputs\schema\halation_gui_schema.json
```

Schema should include:

```json
{
  "version": "halation_v2p3",
  "modelFamilies": [],
  "types": [],
  "colorResponses": [],
  "sliders": {},
  "validCombinations": {},
  "presets": [],
  "previewModes": [],
  "evidenceLevels": {}
}
```

Generated `outputs/` schema should remain ignored unless a committed canonical
schema under `configs/` is more appropriate. Decide based on repo pattern.

### Completion Tests

- Schema JSON validates against its own minimal expected keys.
- GUI-invalid combinations are absent from schema.
- All presets appear exactly once.
- Docs include example schema usage.

## Order 4: Integrated Renderer Black/White Layer Outputs

Status: done on 2026-06-06.

Implemented for halation screen layers:

```text
<layer_name>.png
<layer_name>_on_black.png
<layer_name>_on_white.png
```

Smoke outputs:

```text
outputs/integration/render_film_halation_preset_schema_smoke_layers/physical_halation.png
outputs/integration/render_film_halation_preset_schema_smoke_layers/physical_halation_on_black.png
outputs/integration/render_film_halation_preset_schema_smoke_layers/physical_halation_on_white.png
outputs/integration/render_film_halation_bw_preset_schema_smoke_layers/density_halation.png
outputs/integration/render_film_halation_bw_preset_schema_smoke_layers/density_halation_on_black.png
outputs/integration/render_film_halation_bw_preset_schema_smoke_layers/density_halation_on_white.png
```

### Problem

The evaluator writes layer-on-black and layer-on-white views, but
`render_film.py --write-layers` currently writes only the default layer view.

### Desired Behavior

When a halation layer is written, add:

```text
<output_stem>_layers/<layer_name>.png
<output_stem>_layers/<layer_name>_on_black.png
<output_stem>_layers/<layer_name>_on_white.png
```

Only halation-like screen layers need the extra black/white exports.

### Completion Tests

- CLI smoke with color-negative halation writes all three previews.
- CLI smoke with B&W density halation writes all three previews.
- Metrics remain unchanged except optional artifact path listing.
- Existing non-halation layers are not broken.

## Order 5: More Halation Control Tests

Status: done on 2026-06-06.

Implemented test coverage in `tests/test_halation_controls.py`:

- strict validation rejects invalid family/color combinations;
- compatibility mode preserves B&W fallback;
- every preset resolves and builds a layer;
- schema has required keys and valid combinations;
- previous slider invariants remain covered.

Latest targeted result:

```text
tests/test_halation_controls.py: 11 passed
```

### Problem

Current tests cover important slider invariants but not the full GUI-readiness
surface.

### Add Tests

Minimum additional tests:

- strict validation accepts every valid type/color-response pair;
- strict validation rejects every invalid pair;
- every preset resolves and builds;
- schema has required keys;
- schema contains only valid combinations;
- `amount` changes `amplify` but not diffusion for every family where relevant;
- `source_selectivity` is monotonic for every type;
- black/white layer export path works in CLI smoke;
- metrics include `halation_metadata` and `halation_resolved`.

### Completion Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_halation_controls.py
```

and any new test file added for schema/CLI behavior.

## Order 6: Family-Specific Evaluator

### Problem

The existing physics suite mostly verifies color-negative/cinestill behavior.
V2.3 introduced multiple rule families and color-response laws that deserve
explicit diagnostics.

### Desired Behavior

Add or extend evaluator to check:

- color-negative family has near-zero blue leakage;
- `deep_red` has lower green/red ratio than `amber_core`;
- `amber_core` changes color response without geometry change;
- B&W neutral has near-neutral RGB tint;
- B&W warm has lower blue/green than neutral but no red-layer halo law;
- every family produces four-column sheets.

Possible script:

```powershell
scripts\evaluate_halation_rule_families.py
```

or an extension flag on:

```powershell
scripts\evaluate_physical_halation_v2.py --control-mode family
```

### Completion Tests

- Writes JSON summary.
- Writes contact sheets.
- Fails or warns if expected hue/tint ordering is violated.

## Order 7: Source-Linear Sidecar Support

### Problem

The low-level renderers accept `source_linear_rgb`, but integrated
`render_film.py` has no CLI path to pass it.

### Desired Behavior

Add optional:

```powershell
--halation-source-linear-npy path\to\source_linear.npy
```

The `.npy` should contain:

```text
H x W x 3 float array
```

If size mismatches the post-color-render base, either:

- raise a clear error, or
- resize only with a documented method.

Prefer a clear error for first implementation.

### Completion Tests

- Synthetic `.npy` sidecar accepted.
- Shape mismatch raises a helpful error.
- Metrics note that sidecar source was used.

## Order 8: Arbitrary Halation Contact-Sheet Helper

### Problem

Contact sheets are evaluator-specific. GUI/export tests may produce arbitrary
layer folders that still need visual comparison.

### Desired Behavior

Add:

```powershell
scripts\make_halation_contact_sheet.py
```

Inputs could be:

```text
--original-dir
--layer-black-dir
--layer-white-dir
--combined-dir
--output
```

or a manifest JSON.

### Completion Tests

- Builds a four-column contact sheet from an existing evaluator output.
- Output visually matches expected column ordering.
- Handles missing images with clear errors.

## Order 9: Richer Halation Metrics

### Problem

Current metrics focus mostly on alpha/bounds/visible percentage.

### Desired Metrics

Add family-aware metrics:

```text
alpha_max
alpha_mean
visible_affected_percent
red_green_ratio
blue_leakage
layer_luma_mean
warm_core_ratio
dark_background_visibility_ratio
white_background_tint_delta
```

Only compute metrics that are meaningful for the active family.

### Completion Tests

- Color-negative metrics include blue leakage and red/green ratio.
- B&W metrics include density tint neutrality/warmth.
- JSON remains serializable.
- Docs explain metrics and evidence level.

## Order 10: Evidence Metadata Propagation

### Problem

The spec has evidence levels, but code/schema/metrics do not expose them.

### Desired Behavior

Expose evidence metadata for presets and schema values:

```text
code_fact
uncalibrated_heuristic
external_source_supported
measured_project_result
future_ideal
```

Example:

```json
{
  "preset": "cinestill_amber",
  "evidenceLevel": "uncalibrated_heuristic"
}
```

### Completion Tests

- Every preset has evidence metadata.
- Schema exposes evidence metadata.
- Metrics can record preset evidence metadata.

## Deferred / Higher-Risk Work

### Order 11: Directional Dark-Side Prototype

Current implementation uses symmetric convolution followed by visibility
gating. A more realistic model may bias visible scatter toward the dark side of
bright/dark edges.

Risk:

- could change visual look substantially;
- may introduce artifacts near text, faces, or high-contrast geometry.

Rollback requirement:

- isolate behind a flag or branch;
- produce side-by-side contact sheets before replacing defaults.

### Order 12: Source-Map Improvement Prototype

Potential improvements:

- better point-light detection;
- neon/sign detection;
- white-wall/sky suppression;
- skin/specular protection.

Risk:

- false positives and false negatives can be worse than current stable behavior.

Completion should require diagnostics and contact sheets.

### Order 13: Performance Preview Optimization

Potential work:

- reduced-resolution preview path;
- kernel/cache reuse;
- output-equivalent full-res path preserved.

This should wait until a GUI or preview loop exists.

## Output / Commit Discipline

- Keep generated outputs under `outputs/`.
- Do not commit generated images/contact sheets unless explicitly requested.
- Commit code/docs/tests only.
- Keep `halationguide.md` untracked unless the user explicitly requests adding
  it.
- Each order should ideally be one small commit or a clearly grouped set of
  commits.

## Current Manual Requirements

None for Orders 1-10. These are autonomous engineering tasks.

Manual/user input will eventually be needed for:

- selecting final visual defaults;
- choosing real-film calibration sources;
- approving higher-risk directional/source-map changes;
- integrating into a concrete GUI surface if one is not present in the repo.
