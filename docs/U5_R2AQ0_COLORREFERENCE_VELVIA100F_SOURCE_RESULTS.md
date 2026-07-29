# U5.R2AQ0 ColorReference Velvia 100F source audit

## Decision

AQ0 passes only the source-integrity and semantics gate. It opens a separately
frozen `U5.R2AQ1` pair-table and repeated-target-set identifiability audit. It
does **not** open fitting, training, rendering, stock calibration or product
work.

The official ColorReference test-data surface provides five 763x591 RGB8
source TIFFs in the film-recorder device's unknown RGB space and six target
sets (`1,2,3,4,5,9`) with five individually measured Velvia 100F slides each.
Every slide has 288 IT8 XYZ/Lab/LCh rows and a 41-wavelength CGATS spectral
record. Two independent bounded download/audit passes are exact.

## Evidence

- Frozen config SHA-256:
  `b67cf59e5881a26fce3dc31f2cae61e89bbb7eaea94542cac7a297154dc19dee`
- Formal report SHA-256:
  `9004a7c3c4c406621cf32fd8c5793871840b172d6d3811d1a32ea56b4440d341`
- Software commit:
  `4e998e48ebab700c50e765f0db10cfb07e5f430d`
- Per pass: 17 payloads / 3,147,395 bytes; all expected sizes and file hashes
  repeat exactly.
- Source TIFFs: five RGB8 images, 763x591, samples 0--255, direct array
  preservation with no colour conversion.
- Measurements: 12 safe ZIP archives, 60 unique members, 30 complete
  test-set/slide keys, 288 rows per member.
- IT8 records: `SAMPLE_ID`, XYZ, Lab and LCh, nine fields.
- CGATS.5 records: nine base fields plus 41 wavelength/value pairs from
  380--780 nm, 91 fields; wrapped physical lines parse to 288 logical rows.
- All members declare `Fujichrome Velvia 100F (RVP 100F)`,
  `PROD_DATE "2005:05"` and diffuse `opal` geometry. Serial identities differ
  by target set and slide.

## Interpretation limits

The source is a film-recorder grid, not a digital-camera scene and explicitly
not sRGB. The targets are measured developed slides, not scanner RGB. The six
sets share one declared production date; they are independent target-set or
charge identifiers, not proven independent rolls, emulsion batches or process
sessions. Scanner data is excluded from this lane.

The page offers the files free for interested testing/development, but no
standard reusable data licence was observed. Internal audit is allowed;
redistribution, commercial training and public weights remain closed.

The maximum current claim is:

> controlled recorder-device-to-measured-developed-slide paired proxy
> candidate for Velvia 100F.

It is not a calibrated stock response, scanner-independent appearance model
or identified digital-camera-to-film operator.

## Next gate

AQ1 must freeze the 288-patch geometry before inspecting target variation,
construct all `6 x 5 x 288 = 8,640` rows, validate repeated-set support and
measure nuisance/measurement variability. Only an AQ1 pass may open a
separately preregistered held-target-set explicit-operator baseline.

Source: [ColorReference test data](https://www.colorreference.de/testdata/index.html).
