# SF1.3A YFCC shared-author pixel results

Date: 2026-07-16

Node: `ULT > RF0.4 > SF1.3A`

Decision: **pass bounded acquisition/integrity; open only preregistered SF1.3B identifiability diagnostic**

## Frozen-scope result

- 37 of 38 candidate derivatives retained, 7,975,459 bytes total;
- Ektar100: 21 files / eight UIDs;
- Velvia50: 16 files / eight UIDs;
- all eight UIDs retain at least one pixel for each stock;
- manifest SHA-256: `33395e5bfa765d0a28d0162e5ad40b79b9d54744995623122fe238d072768ba3`;
- final audit SHA-256: `176554b5fd12caa114c21e344bbfcbc4d5ec8c4fd5697ffe4b1a017230ada6b0`.

All 37 retained files pass recorded SHA-256, byte-count, decode and short-side
checks. There are zero exact duplicate groups and zero dHash<=4 pairs.

## Autonomous visual evidence

Both stock-separated contact sheets and the 12 original-resolution automatic
risk cases were reviewed. No confirmed decode corruption, colour blocks,
banding, posterization, geometry failure or other severe artifact was found.
One Ektar dumpster frame has strong source-native overlay/blur aesthetics; it
is input appearance, not corruption produced by this project.

This is autonomous Codex visual evidence, not population preference and not
stock identifiability. Content is visibly heterogeneous and unbalanced, with
same-author scene series.

## Provenance limitation

The acquisition records each live-page outcome and exact image lineage, but
the first implementation omitted a per-request UTC field. No timestamp was
imputed after the fact. SF1.2 and every SF1.3A image request did recheck the
current CC BY 2.0 page; future acquisition code must retain the exact UTC.

## Claim boundary and branch

This result proves only that a small rights-current, connected and clean pixel
pilot exists. It does not prove stock signal, operator identity, multiple
latent modes, calibration or authenticity. Training and operator fitting
remain forbidden.

The only opened child is `SF1.3B`: a frozen leave-one-UID-out diagnostic using
the existing RGB, luma, HOG, 4x4 scene-colour, standardized-RGB and
geometry/border descriptors. It must beat nuisance controls under the
preregistered gates; failure closes this pixel pool for stock learning.
