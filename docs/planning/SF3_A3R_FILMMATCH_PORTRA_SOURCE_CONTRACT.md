# SF3.A3R FilmMatch Portra 400 controlled-source contract

Date frozen: 2026-08-31

Node: `ULT > SF3 > SF3.A3R`

Status: prospective zero-pixel source gate

## Question

Does FilmMatch's official PortraMatch publication expose the underlying
controlled Portra 400 / Sony FX3 observations with sufficient rights,
identities and independent group roles to open a stock-first physical-data
leaf?

This question is materially stronger than a preset or gallery audit. The
publisher describes five Portra 400 rolls, roughly 20,000 colour samples,
three illuminants, an eleven-stop exposure sweep, side-by-side film/digital
capture, linear negative scans and a later validation roll. It is also
distinct from the already-consumed FilmMatch Ektachrome practice dataset: this
leaf asks only whether the Portra source observations themselves are public
and admissible.

## Frozen sources

- `https://www.film-match.com/portramatch`
- `https://www.film-match.com/portramatchlearnmore`
- `https://www.film-match.com/shooting-charts`

Exact current response sizes, SHA-256 identities, required method text and
public Google Drive folder identities are frozen in
`configs/sf3_a3r_filmmatch_portra_source_v1.json`.

The audit may read only these three HTML responses. It must not request page
images, Google Drive metadata or bodies, LUT/product downloads, chart media,
pixels, fits, renders or scores.

## Admission gates

All gates are conjunctive:

1. All three official HTML responses retain their frozen byte identities and
   the Portra capture/scan/method facts.
2. A public Portra-specific paired observation payload is linked, rather than
   only the already-known generic/Ektachrome practice folder or a paid LUT.
3. The Portra observations have an explicit licence authorising research
   fitting and derived parameters.
4. Exact asset hashes or a member manifest are published.
5. Exact roll/process/scanner/source identities and a sealed independent
   confirmation role are published.
6. All forbidden media/pixel/model/product operations remain zero.

Failure closes only this exact public-source route. It does not dispute that
the private Portra dataset exists or that its controlled design is valuable;
an independently licensed exact export could reopen a new prospective leaf.

## Claim ceiling

Even a source pass would open only a separately preregistered data-integrity
audit. It would not establish a Portra operator, stock calibration,
generalisation, product profile or candidate 3.
