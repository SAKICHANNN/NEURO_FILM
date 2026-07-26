# U6.2B Boolean Grain Crop Frontier Results

**Date:** 2026-07-26

**Node:** `ULT > U6 > U6.2 > U6.2B`

**Decision:** **closed by visual severe-artifact veto**. Both frozen Boolean
policies pass the aggregate automatic gates, but neither is acceptable on
real-image crops. Keep U6.2A only as synthetic representation evidence and
retain the existing legacy `.018` heuristic as the clean comparator.

## Reproducibility

- implementation commit: `f938a040cf8dc54dd690f3171d15db840bc9dd59`;
- config SHA-256: `677dd625...410a3e`;
- two serial formal reports are byte-identical at
  `0d6794e5...94801d4`;
- visual review SHA-256: `50087b8d...8c430fe`;
- five source hashes, decoded sizes, crop boxes and seeds match the frozen U41
  lineage;
- repeat float output is exact for every Boolean candidate/sample;
- 9 focused U6.2A/B tests and all **882** CPU tests pass.

The first short-timeout invocation left its Python child alive. A second
background invocation was detected and terminated before evidence collection.
The authoritative byte-identity evidence comes from two later completed,
strictly serial runs.

## Automatic evidence

| Candidate | Mean-luma drift max | Luma RMS range | Low-pass RMSE max | New endpoints | Automatic |
|---|---:|---:|---:|---:|---|
| Boolean small `.22/.12` | `.00567` | `.00904–.01553` | `.00635` | `0` | pass |
| Boolean large `.38/.12` | `.00734` | `.01238–.02162` | `.00895` | `0` | pass |

Both outputs are finite and structurally bounded. Small-radius contexts contain
4,383–51,524 disks; large-radius contexts contain 1,467–17,290. These metrics
show that the fixed composition is visible, repeatable and low-frequency
bounded. They do not establish acceptable grain morphology.

## Full-resolution severe review

The small-radius policy has confirmed severe failures on 4/5 crops. The
large-radius policy fails 5/5. Decisive observations are:

- ID11: dense discrete bright points contaminate saturated red and dark
  surfaces;
- ID37: thousands of salt-like points contaminate a smooth sky gradient;
- `FS_FACE_01`: bright points corrupt skin, hair, hands and background;
- ID21: points contaminate face, uniform, flag and fine insignia;
- ID14: the large-radius circles remain objectionable over inherited
  high-ISO noise.

The legacy `.018` comparator has 0/5 confirmed severe failures. This is
autonomous B0 visual evidence, not owner or population preference.

## Scientific interpretation

The Boolean field is a valid stochastic-geometry representation on synthetic
flats, but this particular fixed low-sample composition exposes individual
coverage events as sparse bright spots. Aggregate luma RMS, sigma-4 low-pass
drift and endpoint metrics all miss that semantic morphology. Severe visual
adjudication therefore remains an independent promotion gate.

Do not rescue U6.2B by lowering strength, changing radius, increasing the Monte
Carlo count, changing crops/seeds or relaxing/replacing the visual veto. A
future grain candidate needs a separately motivated and preregistered
formulation. U6.2 still requires measured NPS, autocorrelation and repeat-scan
evidence before calibration claims.

## Claim boundary

This is B0 autonomous existing-image crop evidence for two fixed clean-room
Boolean compositions. It is not measured film grain, stock/process/scanner
calibration, physical realism, full-frame validation, preference, production
integration or a named-stock result.
