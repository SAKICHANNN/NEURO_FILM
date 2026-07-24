# U5.R2H0C3 external measured-spectrum replication contract

Date frozen: 2026-07-24  
Parent: `U5.R2H0C2`  
Source gate: `U5.R2H0C3` USGS source audit pass

## Confirmatory question

Does the H0C2 hard CAVE-spectrum canonicalizer, with its already selected D65
Lab radius of Delta E76 `1.0`, improve prediction of the same known nonlinear
witness for independent USGS measured reflectances versus the existing smooth
bounded reconstruction?

No witness target was rendered during the source/support audit or while this
contract was frozen.

## Immutable populations

- **Bank:** the H0C2 official-CRC-verified CAVE 31-scene representative
  population, unchanged.
- **Queries/targets:** eligible `AREF` rows from the hash-pinned USGS
  `ASCIIdata_splib07a.zip` under the frozen source audit.
- **USGS group:** filename-derived sample-record identity after removing only
  the `splib07a_` prefix and terminal instrument/measurement tokens.
- **Nuisance stratum:** official archive chapter.

USGS rows are forbidden from the retrieval bank. No query may retrieve itself
or any USGS row because the complete bank is CAVE.

## Immutable policies

1. smooth bounded reflectance reconstruction from query D65 XYZ;
2. CAVE hard Top-1 by D65 Lab when nearest distance is `<=1.0`, otherwise the
   identical smooth reconstruction.

The evaluator also reports raw hard Top-1 as a diagnostic, never as the
candidate policy. Feature, bank, interpolation, threshold, tie tolerance,
fallback and witness remain frozen.

## Gates

Support gates are derived only from the pre-target feasibility scan:

- at least 1,700 eligible queries across all seven chapters;
- at least 128 selected queries and 10% coverage;
- at least 128 selected sample-record groups;
- no sample-record group above 2% of selected rows;
- at least five selected chapters and no chapter above 60%;
- at least four evaluable chapters with eight or more selected rows.

Performance gates retain the H0C2 meaning:

- hard beats smooth by more than `1e-12` on at least 60% of selected rows;
- sample-group bootstrap 95% LCB of win rate is at least 50%;
- selected median relative error reduction is at least 20%;
- sample-group bootstrap 95% LCB of median reduction is at least zero;
- selected hard error is at most Delta E76 `5.0` median / `12.0` p95;
- full hard/fallback policy p95 may not exceed smooth p95;
- hard has lower median error in at least 60% of evaluable chapters.

Two complete runs must be hash-identical. All eligibility reasons and complete
query/retrieval lineage must be retained.

## Decisions

- **Pass:** `external_hard_canonicalizer_replication` and open only an RGB-input
  spectral-estimation/value audit. This still does not open film fitting or a
  photo renderer.
- **Fail:** `external_replication_failed`; close broad empirical-prior claims
  for this fixed candidate. Do not tune the radius, add USGS to the bank, blend
  neighbours or add a neural model to rescue the confirmatory result.
- **Invalid:** source hash, population, support or exactness failure; repair
  implementation/provenance only and rerun the same contract.

## Forbidden inputs and interpretations

Film pixels, project photographs, RGB previews, sample/chapter names as
features, witness outputs as features, CLIP/content embeddings, Top-K blending,
neural training and owner preference are forbidden. A pass is not an identified
reflectance-from-RGB operator, Velvia response, stock response, calibration,
visual safety or product value result.

## Claim ceiling

Cross-source mechanism replication of one frozen hard measured-spectrum prior
for a synthetic datasheet-prior witness on bounded measured reflectances.

