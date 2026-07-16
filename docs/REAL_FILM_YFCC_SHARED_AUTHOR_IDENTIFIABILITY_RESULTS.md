# SF1.3B shared-author stock-identifiability results

Date: 2026-07-16

Node: `ULT > RF1 > SF1.3B`

Decision: **fail; close this shared-author pixel pool for stock learning**

## Frozen evidence

- config SHA-256: `69504c7246908ad770504d25d1b3e1612c71c9899110ce1637957e91d72e634b`;
- report SHA-256: `bf303e68fe1c253e85ec88107d8e6c99e417534901cae25f30e946b2466968af`;
- 37 pixels, eight held-out UIDs and 16 equal-weight UID-by-stock units;
- no result-dependent threshold or feature change.

| Descriptor | Balanced accuracy | Permutation p |
|---|---:|---:|
| global RGB distribution | 56.25% | .464 |
| luma distribution | 62.50% | .318 |
| grayscale HOG | 43.75% | .802 |
| 4x4 low-frequency RGB | 68.75% | .206 |
| standardized RGB distribution | 68.75% | .240 |
| geometry/border/source | 62.50% | .314 |

Global RGB fails both the 70% accuracy and p<=.05 gates. The best nuisance
control is standardized RGB at 68.75%; RGB is 12.5 percentage points worse.
The paired bootstrap interval for RGB minus that control is [-31.25%, 0%], so
both the required +10-point margin and positive lower bound fail. Only minimum
group support and the HOG ceiling pass.

## Interpretation

Shared authors remove the strongest across-uploader shortcut but do not make
stock identity identifiable in this small pool. Scene colour, standardized
colour and geometry remain at least as predictive as raw global colour. This
does not prove Ektar100 and Velvia50 have no distinguishable appearance; it
proves this dataset/design cannot establish it.

## Binding branch

- close these 37 pixels for stock learning, operator fitting and latent-mode
  discovery;
- do not increase model capacity, train a router or cluster the images;
- retain SF1.1-SF1.3B as positive connectivity/rights evidence and negative
  identifiability evidence;
- continue Ultimate via other evidence-authorized stock/data work or the
  independent deterministic product path.

Training, operator fitting, LSM and authenticity claims remain forbidden.
