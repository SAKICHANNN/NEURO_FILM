# FILM-R v2 acquisition, integrity and use gate

## Decision

FILM-R v2 is admitted as a small, rights-clear **real-film scan and nuisance
set**, with the claim ceiling `real-film-derived/unknown-look`. It is not
admitted as named-stock truth, roll truth, paired digital-to-film supervision,
or a neutral restoration target.

The machine-readable decision is
`configs/real_film_filmr_decision.json`. It binds the acquisition, manifest,
integrity report and both visual-review sheets by SHA-256.

## Verified acquisition

- Source: Figshare article 21803304, version 2, CC BY 4.0.
- 44 authentic damaged 35 mm colour scans and 44 supplied expert restorations.
- 88/88 files and 437,570,872/437,570,872 bytes acquired.
- Every remote MD5 matched; every local payload has a recorded SHA-256.
- Acquisition report SHA-256:
  `07e3a26daec77d754203bff05376ff485b65d09821912a443f8f0622c284b79c`.
- Pair manifest SHA-256:
  `a9075014e4fe6063c0137dc4de6af5b8222c54480a237a513f42027ebcf5b6ba`.

## Integrity and visual audit

All 88 JPEGs decoded. Each original/restored pair has matching dimensions.
There are no exact cross-pair duplicates and no dHash cross-pair near duplicate
at distance 4 or less. None of the files carries an ICC profile or EXIF record.
The integrity report SHA-256 is
`f24bfe74a6eee979c3ec0f3d466da0419aa5f38558c22632be9b76d75cb6ec26`.

The two contact sheets and the largest-change pairs were visually reviewed.
No pair mismatch, geometry rewrite or subject-identity corruption was found.
The restored versions mostly remove scratches, dust and local damage. Several
pairs nevertheless change almost every pixel: the maximum mean absolute change
is 5.069/255 and the largest changed-pixel fraction is 0.9987. The most changed
`cinestill800t_half_13` pair removes dense vertical scratches while retaining
the same film-strip image; it demonstrates that an image-wide numerical change
does not imply content corruption, but also that the restored file is not a
neutral colour reference.

## Identifiability limits

Filename parsing yields 11 provisional families, but the distribution is small
and imbalanced: Cinestill 800T has 17 pairs and Velvia 50 has 8; five families
have only one pair. Physical roll, process, scanner, exposure and authoritative
stock labels are absent. Filenames are hints only.

Content is also confounded. Many images show film boxes, rolls, cameras, a film
strip or other archival objects; ordinary photographic scenes are a minority.
Consequently a family classifier or retrieval score can succeed by recognizing
content, damage or acquisition style rather than transferable film colour.

## Enforced use boundary

Allowed:

- characterize real-scan colour, damage and restoration nuisance;
- construct source/pair-held-out exploratory `unknown-look` tests;
- use originals as real-film style references and both siblings for robustness;
- stress artifact detection and content preservation.

Forbidden:

- learn or claim a calibrated response for any named stock;
- claim physical-roll or scanner-independent generalization;
- treat original/restored siblings as digital/film pairs;
- use restored images as clean colour truth;
- split siblings or strongly related filename-family content across train/test.

RF1.1 must therefore begin with nuisance and content separability, not model
training. If film-family signal does not survive content-balanced and
source-held-out controls, FILM-R remains an artifact/style stress set only.
