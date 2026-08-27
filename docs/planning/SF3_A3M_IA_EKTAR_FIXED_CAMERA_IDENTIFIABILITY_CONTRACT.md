# SF3.A3M — Internet Archive Ektar fixed-camera identifiability D0

## Question

Does Kodak Ektar 100 retain a reproducible appearance signal after fixing the
author and camera, and holding out an entire upload/item, relative to two other
Kodak colour-negative stocks from the same author and camera?

## Frozen source and roles

- source: Internet Archive items by `Matthew Paul Argall`;
- camera: exact stated `Kodak VR35 K4`;
- rights: exact item-level `CC BY 4.0`;
- target: Kodak Ektar 100;
- controls: Kodak UltraMax 400 and Kodak Pro Image 100;
- roles: two exact items per stock, one development and one confirmation;
- pixels: at most 12 hash-ranked IA-generated JPEG thumbnails per item;
- grouping: the whole IA item is the indivisible unit.

The item identities and roles are fixed in the JSON contract before any image
body is requested. Every selected thumbnail remains bound to the corresponding
original JPEG name, size and MD5 plus the derivative name, size and MD5.

## Frozen evaluation

The primary contrast is Ektar versus the pooled non-Ektar controls. A frozen
centroid classifier holds out one complete IA item at a time. The primary
descriptor is global RGB distribution. Luma distribution, grayscale HOG,
low-frequency 4x4 RGB, geometry/border/source facts and capture date are
nuisance controls. The exact 15 assignments of two Ektar labels among six item
groups form the permutation null.

Pass requires all of:

- source lock and all local file identities exact;
- 12 decoded thumbnails per item and six item groups total;
- primary balanced accuracy at least `0.8333333333333333`;
- Ektar item recall `1.0` and pooled-control recall at least `0.75`;
- exact permutation p-value at most `0.10`;
- primary advantage over the strongest nuisance control at least
  `0.16666666666666666`;
- date, HOG and low-frequency scene-colour controls each at most
  `0.6666666666666666` balanced accuracy.

## Stop and claim rules

- No item, frame, role, stock, threshold or descriptor may be replaced after a
  pixel has been read.
- Any failure closes this exact single-author/fixed-camera IA route. No larger
  model, more frames, alternate crop or threshold rescue is allowed.
- A pass is only weak, unpaired, low-resolution stock-identifiability evidence.
  It does not identify a digital-to-film operator, permit K=1 fitting, establish
  roll/process/scanner calibration, or promote a product profile.

