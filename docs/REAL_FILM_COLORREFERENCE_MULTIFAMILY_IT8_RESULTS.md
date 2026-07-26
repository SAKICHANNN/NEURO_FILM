# SF2.9R ColorReference multi-family IT8 results

Date: 2026-07-26

Node: `ULT > RF0.4 > SF2.9R`

Status: complete, limited source/measurement pass

Decision: `physical_spectral_and_target_manufacturing_prior_only`

## Executive result

Five frozen ColorReference IT8/ISO 12641 transmission-reference archives pass
all byte, ZIP CRC, deterministic parse, common-patch and spectral-support
checks. They provide 288 common target patch IDs and direct 380--780 nm
transmission measurements at 10 nm intervals. This is useful content-free and
scanner-RGB-free evidence that real target families and batches have measurable
spectral residuals.

It does **not** identify a photographic stock appearance operator. These are
batch-average scanner-calibration targets manufactured toward IT8 aims, and no
uncalibrated common recorder input, camera scene, exposure series or complete
recorder/process lineage is present. Cross-family differences can therefore
contain target manufacture, aim adjustment, batch, age and process nuisance.

No stock operator fitting, training, latent-mode study or production
integration opens.

## Frozen evidence

| Field | Value |
|---|---|
| Frozen config | `configs/sf2_9r_colorreference_multifamily_it8_v1.json` |
| Config SHA-256 | `03f8025030c0e55d36cc70c57cae99166e372bc151e00e8964b627a6e3af9a34` |
| Audit implementation | `de6f786dbf63cf034a98a8e4a4010e80c53bb3eb` |
| Expected / observed bytes | 1,468,016 / 1,468,016 |
| Archives | 5 |
| Common patches | 288 / 288 |
| Spectral grid | 41 samples, 380--780 nm, 10 nm |
| Formal report SHA-256 | `92246d2b2f1e0f3635734177c76a7f9cacbd26ad9d4b50d0c316d8ea6a39889f` |
| Repeat report SHA-256 | `92246d2b2f1e0f3635734177c76a7f9cacbd26ad9d4b50d0c316d8ea6a39889f` |

The two formal reports are byte-identical. Raw archives and reports remain
ignored local research artifacts.

## Archive identity and repeatability

| Archive | Measured material header | Production date | SHA-256 | Median / p90 within-batch mean Delta E76 |
|---|---|---|---|---:|
| `E240220.zip` | For Kodak Ektachrome Product Family | 2023:02 | `4535b861...f5b977a` | 0.54 / 0.773 |
| `V240219.zip` | Fujichrome Velvia (RVP 50) | 2019:01 | `3f601d71...ed1d8fc6` | 0.40 / 0.57 |
| `N230513.zip` | Fujichrome Velvia 100 (RVP 100) | 2017:03 | `a82d0970...611ba4a` | 0.52 / 0.82 |
| `A240301.zip` | Agfachrome RSX II | 2012:08 | `24ef8b4c...adb2fba` | 0.57 / 0.75 |
| `F240222.zip` | Fujichrome Provia 100F (RDP III) | 2016:08 | `ab87e6ac...356416e` | 0.34 / 0.48 |

The exact Ektachrome stock remains unknown because the measured header names
only the compatible product family. The other four headers identify a
material, but not a camera exposure pair, physical roll, independent process
replicate or unadjusted recorder input.

## Common-patch Lab differences

These are measured target-output differences, not stock-look effect sizes.

| Left | Right | Median | p90 | Maximum |
|---|---|---:|---:|---:|
| Ektachrome family | Velvia 50 | 1.675 | 9.150 | 20.024 |
| Ektachrome family | Velvia 100 | 4.223 | 8.940 | 20.178 |
| Ektachrome family | Agfachrome RSX II | 3.262 | 8.227 | 20.272 |
| Ektachrome family | Provia 100F | 1.521 | 4.210 | 15.616 |
| Velvia 50 | Velvia 100 | 3.807 | 8.748 | 17.781 |
| Velvia 50 | Agfachrome RSX II | 3.248 | 10.737 | 23.899 |
| Velvia 50 | Provia 100F | 1.818 | 8.305 | 18.745 |
| Velvia 100 | Agfachrome RSX II | 3.274 | 11.156 | 23.983 |
| Velvia 100 | Provia 100F | 3.637 | 7.249 | 24.889 |
| Agfachrome RSX II | Provia 100F | 2.175 | 9.198 | 19.257 |

The result establishes neither a monotone ranking of photographic saturation
nor a digital-to-film transform. Large tail differences are compatible with
different dye spectra, target manufacturing corrections and batch/process
residuals.

## Identifiability decision

### What improved

- Scene content and ordinary photograph composition are absent.
- The data are direct colourimetric, density and spectral measurements rather
  than scanner RGB pixels.
- All families share the same 288 nominal patch IDs and measurement schema.
- Batch-average repeatability fields quantify a real measurement/manufacturing
  noise floor.

### What remains missing

- the uncalibrated recorder input or target aim adjustment for each patch;
- a common camera scene or exposure series;
- independent physical-roll and process replication;
- complete recorder, chemistry, measurement and ageing lineage;
- reusable data rights for training or redistribution.

Consequently, subtracting one archive from another would conflate the film
material with how each calibration target was manufactured to meet aim values.
It would not recover the response a scene would have received on that stock.

## Branch decision

Allowed:

- retain the measurements as an internal physical spectral representation
  feasibility source;
- use them as scanner-profile, batch and target-manufacturing nuisance controls;
- use the schema to specify what a future common-input film measurement source
  must contain.

Forbidden:

- fit a stock look, LUT, curve or renderer from these archives;
- train a stock classifier, router or latent-mode model;
- describe the pairwise distances as calibrated stock distinctions;
- redistribute the archives or derived weights;
- integrate any result into the production renderer.

The next algorithm leaf must use either clean-room physical constraints or a
separately rights-cleared, independently lineaged common-input dataset. SF2.9R
does not change the current `operator_fitting_allowed=false` project state.

## Claim ceiling

Bounded internal schema, spectral-measurement, batch/family and
target-manufacturing-nuisance evidence only. No camera-scene pair, exact
multi-stock response, stock style, operator fitting, training, calibration,
latent mode or redistribution claim.
