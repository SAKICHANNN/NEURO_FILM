# U1.6G4F Explicit New-Operator Value/Safety Audit Contract

**Date:** 2026-07-18

**Node:** `ULT > U1.6 > U1.6G4F`

**Status:** frozen / audit implementation ready

**Config SHA-256:** `c36bc840c5acdc603800abc33a12ede59ebf82e107aa352073ec88cfa7773133`

## Question and epistemic boundary

G4E proved that `staged-density-halation-v1-defaults` executes its explicit
materialized-v2 target, then failed one frozen legacy-compatibility gate. G4F
does not rescue that result. It asks a separate question:

> Is v2 useful and safe enough to retain as an explicitly versioned, opt-in
> density-halation candidate rather than a drop-in replacement?

This is an optical-effect audit, not a film-stock colour/style claim. A pass
does not connect the module to the renderer, make it default, identify a
physical film process or establish total 100MP memory.

## Evidence split

Development uses only `u41-03` and `u41-05`. Their hashes and shapes are frozen
in the config. Before this contract, both were used to define metrics and gates:

| Metric | `u41-03` | `u41-05` |
|---|---:|---:|
| changed rounded uint8 channel fraction | 0.0810 | 0.1197 |
| composite absolute p99 | 0.00577 | 0.00689 |
| maximum uint8 code delta | 8 | 7 |
| alpha active fraction (`>1e-4`) | 0.4287 | 0.8628 |
| top-luma-decile / other alpha mean | 25.86x | 7.38x |
| new high/low clipping | 0 / 0 | 0 / 0 |
| tile-256 median seconds/MP | 1.21 | 1.16 |

Development visuals show a selective warm response around bright flowers and
sunlit driftwood. It is subtle in a whole-image thumbnail, so the contract does
not call it a strong global style. It is visibly structured in localized
effect/difference views without seams or broad veiling.

Confirmatory uses only the previously unviewed `u41-07`, `u41-12` and
`u41-16`. Before freezing, only filename, bytes, hash and decoded shape were
read. No pixels, metrics or visual outputs from these three informed gates.

## Fixed audit

For each confirmatory case:

1. verify source SHA-256, shape and finite float32 decode;
2. execute staged v2 at tile sizes 127 and 256;
3. compare each result to the materialized-v2 reference;
4. compute identity-relative alpha, composite, rounded-code, highlight
   selectivity and clipping metrics from the tile-256 result;
5. report legacy output drift as a diagnostic only;
6. time three independent tile-256 executions and report median seconds/MP;
7. emit whole-image panels and a deterministic 1:1 crop centred on maximum
   smoothed alpha for autonomous visual review;
8. repeat the complete audit and require byte-identical JSON fields except
   runtime observations and output paths; pixel and metric hashes must match.

The maximum-alpha crop locator is fixed before confirmatory viewing. It is an
effect-location tool, not a content selector.

## Frozen automatic gates

Every confirmatory case must pass:

- staged-v2 alpha/composite/seam maximum error `<=2e-6` at both tile sizes;
- staged-v2 rounded sRGB8 composite byte identity at both tile sizes;
- changed rounded uint8 channel fraction `>=0.04` versus identity;
- composite absolute p99 `>=0.0035`;
- maximum rounded uint8 code delta `>=3`;
- alpha-active fraction (`alpha > 1e-4`) `>=0.05`;
- top input-luma decile mean alpha / remaining mean alpha `>=3.0`;
- new high and low clipping fractions each `<=1e-4`;
- finite bounded output and alpha within the explicit `[0,0.26]` cap;
- tile-256 three-run median `<=2.5 seconds/MP` on the current local CPU;
- deterministic pixel and metric hashes across two full audits.

Runtime is a local engineering screen, not a cross-platform benchmark.

## Frozen visual gate

- zero confirmed seam, band, posterization, colour block, clipping expansion,
  objectionable broad veil or other severe artifact on all three cases;
- at least two of three deterministic 1:1 crop comparisons show a plausible,
  localized highlight glow at the unamplified output scale;
- an intended warm highlight response is not itself an artifact;
- amplified maps may diagnose location but cannot satisfy the unamplified
  plausibility count by themselves.

This is autonomous effect evidence, not owner/population preference.

## DoR / DoD

DoR: G4E evidence is frozen; the failed `5e-4` legacy alpha gate is unchanged;
the two development and three confirmatory groups are disjoint; current clean
head is `5fff518c345627659945eb3853ceeb09a9331a75`; no renderer integration, download, training or GPU job is
opened.

DoD: committed audit script/tests; two reproducible executions; automatic and
full-resolution visual decisions; focused plus complete CPU tests; result
propagation and a scoped commit/push.

## Branches

- **All gates pass:** retain as a separately versioned research candidate and
  open a separate integration/memory leaf; integration remains closed here.
- **Non-empty metrics fail:** close for insufficient product value; do not
  increase strength after confirmatory viewing.
- **Selectivity fails:** close as broad/scene-insensitive behavior.
- **Severe or clipping fails:** reject regardless of salience.
- **Runtime/resource fails:** retain numerical evidence only; no integration.
- **Legacy diagnostic diverges:** report it; do not rewrite G4E or call v2 a
  replacement.

## Claim ceiling

A pass supports only a deterministic, bounded, opt-in research candidate for
localized density halation. It does not establish a strong global film look,
legacy compatibility, production/default status, physical calibration,
stock response, colour-family execution, streaming decode or 100MP total
memory.
