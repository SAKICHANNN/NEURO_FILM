# Roll2Film BlueNeg metadata and acquisition freeze

Date: 2026-07-15

Node: `ULT > U5.CT2/U5.CT6`

## Decision

The metadata/licence/whole-roll gate passes for a bounded one-film-string
mechanism pilot. It does not pass for a broad 13-film-type Roll2Film claim.

The acquisition is reduced from the complete 290GB archive and the initially
considered 955,748,961-byte two-lane mirror to exactly 101 files / 118,929,719
bytes needed by the frozen operator-eligible rolls. No image payload was
downloaded or decoded while making this decision.

## Official source freeze

- Repository: `ttgroup/blueneg-release`
- Exact revision: `b038a1ae68f42067ff12b5e79ddbe62919b7af23`
- Metadata: 491 frames, 53 rolls and 13 film-type strings
- Alignment metadata: 428 restricted-unpickler-validated matrix/bbox records;
  five bboxes extend slightly beyond preview bounds and require intersection
  cropping rather than negative indexing
- Full public 8-bit lanes at that revision: 491 previews / 687,735,976 bytes
  and 247 pseudo-GT files / 268,012,985 bytes
- Licence snapshot SHA-256:
  `ce5b4aae6ae83158b950bac00107bab6ff20466e47a0f4771a2606bb7660aeda`
- Required image credit: `Copyrighted by Tien-Tsin Wong`

The repository card and ICCV paper define BlueNeg as negative-film restoration
and archival data. Printed photographs are used to estimate proxy ground truth.
This is not a captured digital-scene-to-film dataset.

## Failure shield discovered by the gate

`meta.json` contains pseudo-GT-looking paths for some blue-intact frames, but
those paths are not present in the public repository. The first evidence run
failed closed instead of fabricating targets. Availability now requires both a
metadata path and a matching entry in the exact-revision remote inventory; a
regression test preserves this rule.

## Whole-roll eligibility

Any roll containing even one official test frame is sealed in full. This seals
17 rolls and leaves 233 frames across 36 nonsealed rolls. Requiring at least
four actually published, non-test pseudo-GT frames leaves five operator rolls:

| Roll | Film string | Public paired frames | Pool | Same-film wrong-roll control |
|---|---|---:|---|---|
| 19960816G | Kodak Gold 100-5 | 13 | development | yes |
| 19960816H | Kodak Gold 100-5 | 6 | development | yes |
| 19960817H | Kodak Gold 100-5 | 12 | confirmatory | yes |
| 19970620C | Kodak Gold 100-5 | 12 | confirmatory | yes |
| 19941219C | Kodak Gold 100 | 7 | development/exploratory | no |

Thus the matched core contains four rolls of only one film string. The fifth
roll is exploratory and cannot support a correct-roll-versus-same-film-wrong-
roll promotion claim.

Within each eligible roll, paired frames are deterministically divided into
unpaired support sets and hidden aligned queries. The support source and target
sets may be shuffled and used for set-level inference; query alignment is
evaluator-only. Development and confirmatory pools have zero whole-roll
overlap.

## Frozen evidence

- Evidence report:
  `outputs/roll2film/blueneg_v1/evidence/report.json`
- Report SHA-256:
  `2703d07036f7bdc5ac296df7bfa9617e294562dacd2b75bb27623dc037b50232`
- Acquisition manifest SHA-256:
  `c221016837284c0a8110909852480a118db29ebd9a2e44a59c9dc6ac11798067`
- Evidence software commit: `d9965e9d4e2e5d3d0da8b269a8a0933a6b1f55be`
- Tests: 86 pass
- Frozen acquisition decision:
  `configs/roll2film_blueneg_acquisition_decision.json`

## CT6 experiment boundary

The confirmatory question is narrow:

> For two held-out `Kodak Gold 100-5` rolls, does an explicit operator inferred
> from the correct roll's unpaired support set improve hidden-query transfer
> over pooled, shuffled, random-roll and the two development same-film
> wrong-roll controls at fixed support budget?

Date/location/scene, deterioration, exposure, scanner and pseudo-GT construction
remain possible confounders and must be reported. Passing would establish only
archive/scanner-specific roll-group information in this small corpus. Failure
or ambiguity closes the current real-roll novelty claim; it does not damage the
independent deterministic product recipe bank.

## Storage, retention and release

- Local source bytes remain ignored under `data/raw/blueneg`.
- Download only the 101 paths and verify every byte count and LFS SHA-256.
- Retain source bytes for reproducible internal research; any deletion uses a
  separately reviewed cleanup.
- Do not redistribute images, examples, derived artifacts or weights without a
  separate rights/release review and the exact required credit.
- The full 290GB archive remains forbidden for this pilot.
