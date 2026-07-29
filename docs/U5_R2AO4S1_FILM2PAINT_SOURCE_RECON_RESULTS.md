# U5.R2AO4S1 FILM2PAINT exact-source reconnaissance

Date: 2026-07-29

Decision: **strong controlled design; publication-only and rights-blocked**

Claim ceiling: publication-level method and data-design evidence only.

## Question

Does the independent FILM2PAINT study expose reusable measurement-level
positive-film chart correspondences that can extend the current 71-pair
author-rendered display proxy without increasing operator capacity first?

The preregistered audit was bounded to official publication and repository
metadata. It did not acquire a dataset payload, extract figure pixels, fit an
operator, contact an author, or infer rights from public readability.

## What is scientifically valuable

The official six-page paper documents two distinct controlled target families:

- a photographed 140-patch ColorChecker Digital SG on Velvia, Ektachrome and
  Kodachrome reversal films, including two stated Ektachrome exposures;
- a 288-patch Coloraid IT8 target generated with a film recorder on several
  E-6 families, with the same reported exposure across slides;
- hyperspectral digitisation converted using D65 and the CIE 1931 2-degree
  observer;
- explicit deterministic baselines (per-channel curves plus matrix and
  polynomial mappings) alongside a shallow neural correction baseline.

This is a materially stronger experimental design than an uncontrolled
community-photo pool. It is also independent of the Balica 2026 Figure 8/11
display-proxy source.

Two interpretation boundaries remain important even if the data later become
available:

- the film-recorder IT8 stimulus is device-space exposure, not a digital
  camera scene-linear input;
- the Digital SG films were shot at various exposures, while exact row-level
  exposure, roll, process and scanner-profile lineage is not published.

The source could therefore support controlled correction/operator research, but
would not automatically identify a general digital-camera-to-stock response.

## Availability and rights result

The exact official PDF is 2,002,643 bytes at SHA-256
`bad7ee283b3d8352faf45bdc567cb8166d9aaf4f5a5dc05d206300165643f0f7`.
It provides aggregate tables and rendered figures, not patch measurements,
raw scans, a manifest, code, weights or a data licence. Its copyright statement
is Society for Imaging Science and Technology copyright, not a reusable data
grant.

The paper states that Zurich Central Library provided the film-stock dataset
to the authors. Independent official metadata checks found:

- no Crossref related-resource relation;
- zero exact FILM2PAINT records in Zenodo;
- zero DataCite records related to the article DOI;
- the cited University of Basel source thesis has one original 24,242,582-byte
  PDF and no dataset bitstream;
- its 1,748-byte repository licence grants repository distribution and
  preservation, not reuse of the underlying experimental data.

Public visibility of the paper or thesis therefore does not satisfy the frozen
measurement-level or reusable-rights gates.

## Branch decision

Decision code:
`strong_controlled_design_but_publication_only_and_rights_blocked`.

- Do not extract publication figures as additional training or fitting pairs.
- Do not treat the film-recorder IT8 values as camera scene RGB.
- Do not fit, train, cluster modes, promote a stock response, or integrate a
  product candidate from this source.
- Retain the target topology and simple explicit correction families as future
  baselines.
- Reopen only if an official release supplies immutable per-patch film
  measurements or raw scans, target/reference values, exposure/process/scanner
  lineage, and explicit reusable data rights.

This closes only AO4S1. It does not weaken the existing AO4P/AO4C evidence and
does not stop the Ultimate Goal.
