# SF2.6R paired-source availability reconnaissance

Date: 2026-07-26  
Status: complete; both sources remain unavailable for fitting  
Claim ceiling: metadata/method precedent only

## Question

Do either of the two primary publications that describe controlled
digital/film correspondences currently expose enough reusable data, lineage and
licensing to open a stock-specific fitting leaf?

This was a bounded read-only reconnaissance of official public surfaces. It was
not a pixel audit, licence decision, fitting experiment or request to an
external person.

## SillyStill / CineStill 800T

Primary surfaces:

- paper: <https://arxiv.org/abs/2411.15967>
- official repository:
  <https://github.com/mikasenghaas/sillystill/tree/b1ebdb971207b403b7126b759c1d30d30954a722>

Observed facts at the pinned repository revision:

- the paper describes 41 raw pairs and 38 processed pairs;
- the official README still labels both the Zenodo and Hugging Face dataset
  downloads `Not yet available`;
- the recursive repository tree contains no released `data/` corpus, release,
  tag or alternate public branch;
- only one illustrative pair is present:
  `imgs/table_cafe_digital.ARW` and `imgs/table_cafe_film.JPG`;
- GitHub reports no repository-level licence;
- `report/LICENSE` is an MIT text for the report/template subtree and does not
  establish a licence for the absent dataset or the example photographs.

Decision:

`dataset_unavailable_and_rights_unresolved`. One illustrative pair cannot
support held-out scene, roll, process or scanner evaluation and must not be
expanded into stock truth. No download, fitting or training opens.

## Emulating Emulsion / Velvia 100

Primary surfaces:

- official project page: <https://musicofmusix.github.io/siggraphposters25>
- official poster abstract:
  <https://musicofmusix.github.io/assets/misc/siggraph_abstract.pdf>
- public portfolio repository:
  <https://github.com/musicofmusix/musicofmusix.github.io/tree/6be041ad38fee8052d2f9eecb0414f9ec0c5c2f2>

Observed facts:

- the project describes a two-matrix, three-sigmoid analytic RAW-to-scan model
  with about 30 fitted parameters;
- it describes 33 chart pairs from one 36-exposure Velvia 100 roll, three
  illuminants, eleven exposure levels and 3,168 unique patch correspondences;
- the public project page and repository contain the paper/poster and rendered
  web figures, but no training measurements, fitted parameters, source-code
  repository or reusable data licence;
- the project itself states that only two real-scene exposures remained after
  the chart captures.

Decision:

`method_public_data_and_parameters_unavailable`. The analytic architecture is
valuable method precedent and may motivate a clean-room, data-independent
operator-family audit. It cannot be called a Velvia reproduction, fitted stock
operator, calibration source or latent-mode teacher without the missing
measurements and independent-roll validation.

## Binding branch

- Keep both sources in the evidence registry as unavailable candidates.
- Do not train on or fit from publication figures.
- Do not infer a dataset licence from a paper, a report-subtree licence or
  technical accessibility.
- A future source leaf may reopen only if an official dataset/parameter release
  supplies explicit rights and immutable lineage.
- A separate algorithm leaf may test the published functional form using
  original synthetic parameters, with the label
  `film-inspired/data-independent architecture witness` and no stock claim.
