# U2.5B profile-driven safe-Lab adapter contract

Date: 2026-07-18

Node: `ULT > U2 > U2.5 > U2.5B`

Status: **frozen before implementation**

## Parent evidence and defect

U2.1A proves that `safe_rich_v1.json` is an exact, hashed migration of the
legacy safe-rich YAML/assets. U2.6A exposes its evidence truthfully. However,
the current renderer still sources colour parameters directly from the legacy
YAML. `--render-profile` is consulted only after image encoding when a recipe
is requested, so it is not yet a render adapter and an invalid profile can be
reported after an output has already been written.

## DoR

- U2.1A exact profile migration/replay and U2.6A evidence inspection pass;
- the current default legacy render path and all frozen output hashes remain
  authoritative;
- the tracked profile supports all eight current safe-rich styles;
- no operator, evidence, schema or effect change is required.

## Allowed implementation

1. Add one explicit CLI opt-in, `--use-render-profile`.
2. When enabled, validate the selected profile and all assets before loading
   pixels or writing output, require the requested style, and source only the
   colour parameters from `profile.style_parameters[style]`.
3. Reuse the existing safe-Lab kernel and effect/export path; do not duplicate
   rendering math.
4. When `--write-recipe` is requested, validate the selected profile before
   rendering even without the new flag, so invalid provenance cannot leave a
   rendered image or sidecar.
5. Record whether the profile-driven adapter was used in metrics without
   changing the v1 recipe schema.

## Frozen gates

- all eight styles at sRGB8 are byte-identical between legacy and opt-in
  profile-driven paths on a deterministic nontrivial fixture;
- HP5, Velvia50 and Ektar100 at sRGB16 are sample-identical;
- opt-in output repeats byte-identically;
- tracked legacy/default smoke hash remains unchanged;
- missing style, invalid schema, evidence escalation or asset hash mismatch
  fails before image, metrics, recipe or layer output exists;
- a profile-driven recipe verifies against the selected profile and assets;
- default CLI behavior remains legacy-compatible and the flag is false in
  metrics;
- focused/full/compile/diff checks pass.

## Forbidden fallback

No default-path switch, v1 schema edit, profile/evidence upgrade, new colour or
effect operator, tolerance widening, calibrated/stock-authenticity claim,
profile editor, or broad renderer refactor. U2.5A B&W neutral-axis behavior
must remain exact.

## Claim ceiling

Passing proves an explicit profile-driven adapter reproduces the existing
legacy safe-Lab render for the frozen styles and fails closed on invalid
profiles before output. It does not prove new image quality, calibration,
stock authenticity, arbitrary future-profile compatibility or default-path
promotion.

