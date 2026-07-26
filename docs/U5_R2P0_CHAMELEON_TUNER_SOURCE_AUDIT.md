# U5.R2P0 ChameleonTuner Source and Method Audit

Date: 2026-07-26  
Node: `ULT > U5 > U5.R2 > U5.R2P0`  
Decision: **retain only as a future misaligned-pair calibration prior**

## Why it was audited

ChameleonTuner is a WACV 2026 method for optimizing an explicit 3D LUT from
reference photographs with limited field-of-view and point-of-view mismatch.
That superficially resembles the project's desired “find a related photograph
and learn how its colour should move” direction, so the exact supervision and
public availability were checked before any implementation.

Primary sources:

- [WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Tan_ChameleonTuner_Automatic_ISP_Color_Tuning_in_Subjective_Scenarios_WACV_2026_paper.pdf)
- [official repository](https://github.com/ZjTan4/ChameleonTuner)

## Exact method boundary

The method still requires paired images of the **same scene**. It relaxes
pixel alignment, not pair identity:

1. LSC segments each source and target into superpixels.
2. A pretrained LoFTR model finds local same-scene keypoint matches.
3. Keypoints vote for corresponding region pairs.
4. Gaussian blur, regional IQR filtering and averaging produce representative
   colour pairs.
5. NSGA-II searches source-supported control vertices of a `17³` trilinear
   LUT against CIEDE2000, hue and saturation objectives.

DPED supplies simultaneous smartphone/DSLR same-scene captures. FiveK supplies
raw/expert-C paired images. Neither dataset is film-stock truth. The paper
reports approximately one CPU day to converge on DPED.

This is useful evidence that exact pixel registration is unnecessary when
real same-scene pairs exist. It is not evidence that unrelated “similar”
photographs can identify an operator.

## Availability audit

The official CVF PDF is `3,490,714` bytes at SHA-256
`814ac185346ff87afe0ba9ed6fca1626bd7bab32d3df24003e77cb10100de370`.

The official repository at revision
`b6ca8954c3f2a0dd0e500a9272cbdaf4e9b06530` contains exactly one tracked
131-byte README. It says code will be released later. There is no
implementation, checkpoint, dependency freeze, example data or licence file.

Therefore a current reproduction cannot honestly claim paper compatibility.

## Decision

Do not implement or run ChameleonTuner now:

- current Commons/YFCC film pixels are closed by identifiability gates;
- the project has no rights-cleared same-scene digital/film pair set;
- applying LoFTR to unrelated archive photographs would convert content and
  geometry similarity into false colour supervision;
- DPED/FiveK could only be digital method controls already covered more
  directly by FilmSet/FiveK evidence;
- the one-day CPU search is unjustified without an eligible paired question.

Reopen only for rights-cleared same-scene digital/film or scan-pipeline pairs
with independent stock/roll/process groups, or after an official licensed
release and a separately frozen non-stock paired-control question.

This audit does not solve unpaired operator identification, authorize stock
learning, or add a runnable algorithm leaf.
