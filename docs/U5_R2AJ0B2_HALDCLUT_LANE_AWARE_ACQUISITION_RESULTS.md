# U5.R2AJ0B2 lane-aware HaldCLUT acquisition results

## Decision

`U5.R2AJ0B2` passes its frozen lane-aware integrity contract. Two new child
processes at commit `9bd4e683...b1b9c8` independently rehash the retained
archive, verify every ZIP member/CRC, decode all 295 image members and produce
byte-identical canonical manifests and reports.

This result repairs no v1 result. AJ0B v1 remains closed on its false
RGB/RGBA-only assumption.

## Exact evidence

| Field | Value |
|---|---|
| config SHA-256 | `6201d70c...ec376` |
| archive bytes | `421602289` |
| archive MD5 | `4742e362a70c1a1c0fb9042a17d285e1` |
| archive SHA-256 | `0ffca81f...9dc2` |
| run A/B manifest SHA-256 | `b6cbd445...a1bda` |
| run A/B report SHA-256 | `98287994...3d7f6` |
| repeat decision SHA-256 | `ce9249ec...cadd` |
| primary paths | `194`, SHA-256 `83b22255...6115f` |
| image records | `295`, SHA-256 `5e59e95e...03e2` |
| profile assignments SHA-256 | `d2d40521...07881` |

Every repeat-decision check is true, including canonical evidence bytes,
schema/key/type validation, config/software identity, report/manifest binding
and all single-run facts. Each single report says `single_run_pass=true` but
keeps `automatic_pass=false` and `structural_audit_ready=false`; only the
parent repeat decision opens readiness.

## Exact profiles

| Profile | Count |
|---|---:|
| Color PNG RGB8 level 12 | 226 |
| exact Ektar Color PNG RGB8 level 16 | 1 |
| B&W PNG L8 level 12 | 66 |
| Negative PNG RGB16 level 12 | 1 |
| identity TIFF RGB16 level 12 | 1 |

All 295 members have no embedded ICC. The B&W `L` entries are accepted only
in their exact auxiliary lane; no conversion occurs. The 194 primary
non-Creative Color paths remain unchanged and all are RGB8.

## Verification

- 20 dedicated acquisition/evidence tests pass;
- 32 acquisition-adjacent tests pass;
- all 1,071 CPU tests pass;
- independent adversarial review confirms fail-closed behavior for Boolean
  zero values, unknown root/nested fields, profile-count/type laundering and
  overlapping selectors;
- legal v1 manifest/report bytes and repeat-decision object remain unchanged.

## Branch and claim ceiling

B2 opens only contract design for `U5.R2AJ0C`, a synthetic-only,
per-candidate structural frontier. AJ0C execution is not ready until its
config, controls and thresholds are committed before primary metrics.

AJ0C must:

- use only the 194 RGB8 Color primary members;
- generate its encoded-sRGB identity lattice analytically;
- preserve each member's native N=144 or N=256 cube;
- avoid Pillow samples from the 16-bit root controls;
- screen candidates individually so one failed preset does not close the
  whole bank;
- render no photograph before a later gate explicitly opens it.

No current result establishes aesthetic value, severe-artifact safety on
photographs, film authenticity, stock response, calibration, a real
digital-to-film operator, training target, product integration or asset
distribution approval.
