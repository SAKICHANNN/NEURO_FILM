# U1.6G4E Staged Density-Executor Results

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4E`

**Decision:** staged-v2 executor passes; legacy-compatible/default promotion closes

## Reproducibility

- implementation commit: `25cb3858c31390dd89f3dfa4fd20aa1b1bd58811`;
- final audit commit: `57ac77a317c7936c1fa067dd0eaf36fa527b5ced`;
- corrected config SHA-256: `90785681500d9d2c217a755202591c37edd72ef90de834867e631f1b5f3ae69b`;
- ignored reports: `outputs/u1_6g4e/formal_57ac77a_{a,b}.json`;
- report SHA-256: `4caa78fe89d4a05a0bbbaaa342268c7ce16080d1d5c6fed79cb545425d2d8d48`;
- contact sheet SHA-256: `a4876a85184accff186a7ac56799d7f9d0a8b142dc1b41cfd65238b1c514c274`;
- non-promoting visual-diagnostic config SHA-256: `4c943c33baae8a5eeca58aab70ca83fb628f373f7251265c53b5cb2f00fa7f6c`;
- both report and sheet are byte-identical across two full executions.

The pre-implementation coarse-chunk correction and two audit-report-only fixes
are recorded in the contract/log. None changed pixels, cases, thresholds or
gates.

## Staged executor result

All ten tile policies (two synthetic plus three real cases, two tile sizes each)
pass the staged-versus-materialized-v2 gates:

- synthetic alpha/composite/seam differences are exactly zero;
- real alpha maximum is at most `1.02e-7`;
- real composite maximum is at most `5.96e-8`;
- all seam maxima are zero;
- all rounded sRGB8 composites are byte-identical;
- every same-policy layer and metadata repeat is byte-identical;
- every source/context row request stays below full height;
- no persistent derived full-frame scalar or scratch disk is declared.

Thus the bounded executor implements its explicit v2 numerical target correctly.

## Frozen legacy compatibility failure

Nine policies pass every legacy gate. Both `u41-14` policies fail one gate:

| Metric | Frozen gate | Observed |
|---|---:|---:|
| legacy alpha max drift | `<= 5.0e-4` | `5.337968e-4` |
| legacy alpha mean drift | `<= 2.0e-5` | `5.969e-6` |
| legacy composite max drift | `<= 5.0e-4` | `1.591e-4` |
| legacy composite mean drift | `<= 2.0e-5` | `2.513e-6` |
| changed uint8 channel fraction | `<= 0.003` | `6.855e-4` |
| maximum uint8 code delta | `<= 1` | `1` |

The alpha maximum exceeds the gate by about 6.8%. The gate is not changed.
Automatic promotion therefore fails even though composite/uint8 gates pass.

## Non-promoting visual diagnosis

Autonomous full contact review finds no confirmed seam, band, block, clipping
expansion or other severe artifact. Staged and materialized-v2 panels are
visually indistinguishable; amplified differences are black at sheet scale.
Legacy and v2 also appear indistinguishable, but visual evidence cannot override
the preregistered automatic failure.

## Resource evidence

The first real/tile-127 policies use about `0.90-0.91MB` of global contexts,
`1.0MB` of actual percentile histograms, maximum source windows of
`81x1600x3`, and a declared `8.95MB` tile-workspace bound. Input and output are
reported separately; no scratch disk is used.

24MP arithmetic (not execution) reports:

- input: 288,000,000 bytes;
- required RGB+alpha output: 384,000,000 bytes;
- global context peak: 12,791,340 bytes;
- percentile histogram maximum: 1,572,864 bytes;
- finite-blur peak workspace: 4,091,472 bytes;
- declared tile workspace: 83,759,104 bytes.

This is not a total-memory measurement or 100MP claim.

## Verification

- 16 focused executor tests and 107 adjacent prerequisite tests pass;
- 527 complete CPU tests pass in 21.83 seconds;
- invalid input/geometry, nonfinite context and injected provider failure close;
- legacy effects, renderer, profiles, recipes and CLI remain unchanged.

## Branch decision

Retain `staged-density-halation-v1-defaults` and its materialized reference as
explicit, unconnected research evidence. Do not replace the legacy density
effect or call this legacy-compatible/default-promoted.

The next legal research leaf may preregister an **explicit new-operator** audit
on fresh confirmatory cases, where legacy drift is reported but not a drop-in
gate. That would be a separate behavior/version claim, not a retroactive rescue
of G4E. Renderer integration remains closed.

## Claim ceiling

This proves bounded execution of the explicit default-formula v2 target on
frozen cases and records one legacy-compatibility failure. It does not prove
drop-in compatibility, physical calibration, renderer integration, colour-family
execution, streaming decode, total 100MP memory, stock response or product
promotion.
