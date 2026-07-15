# RF0.3 FSA/OWI real-film source gate

**State:** acquisition contract frozen; metadata enumeration is the next ready
leaf. No bulk image download is authorized by this document alone.

## Why this source is materially better than FILM-R

The Library of Congress (LOC) describes about 1,600 FSA/OWI colour photographs
made in 1939--1944. The FSA subset contains 644 colour photographs, most of
them 35 mm Kodachrome slides; the OWI subset contains 965 colour
transparencies. The collection spans rural life and farm labour as well as
factories, railroads, aviation and women at work. The guide record describes
1,616 colour slides/film records and twenty named photographers.

LOC records that the original slides and transparencies were scanned in
2003--2004 with a Sinar 54 capture system. Slides were scanned at 3,600 dpi and
larger transparencies at 1,800 dpi. LOC says the archival scans received no
image enhancement or colour correction and were intended to represent the
film as it existed at scan time. Service JPEGs add mild sharpening. This gives
the collection a documented physical-film origin and a largely shared capture
procedure, while leaving ageing, original exposure, film format and derivative
compression as explicit nuisance variables.

The LOC rights page states that this collection is in the public domain and is
free to use and reuse. The Commons mirror exposes a stable API and carries LOC
`fsac.*` digital identifiers, creator/date fields and public-domain metadata.
It is an enumeration and bounded delivery mirror, not a replacement for LOC's
provenance claims.

## Evidence ceiling

This source can support `real-film-derived archive scan look` experiments and
creator/assignment holdouts. It cannot establish:

- a digital-input/film-output pair;
- an individual physical-roll label;
- a controlled development or scanner-settings group;
- a pure or calibrated Kodachrome emulsion response;
- a claim that Commons thumbnails are identical to LOC archival TIFF colour.

FSA and OWI must not be silently pooled as the same film stock. The guide says
most FSA colour images are 35 mm Kodachrome, while OWI includes larger colour
transparencies. Medium, series and source identifiers must remain observable
or the record is excluded from any stock-specific analysis.

## Staged gate

### Phase A -- metadata only

Enumerate the Commons category through its MediaWiki API, retain every API
continuation token, and write a deterministic manifest and report. Pass only
if all of the frozen thresholds in
`configs/real_film_fsa_owi_acquisition.json` hold. A category count is not
enough: LOC identifier, public-domain and LOC-source coverage are independent
gates.

### Phase B -- 64-image decode/visual pilot

Only after Phase A passes, choose a deterministic creator-balanced sample of
at most 64 Commons 1,280 px derivatives and enforce a 64 MiB aggregate cap.
The pilot audits:

1. broken files, dimensions, colour profiles and exact duplicates;
2. obvious crops, borders, restoration, retouching or modern colourization;
3. content balance by photographer and coarse scene class;
4. whether the extra Commons derivative step creates unacceptable artefacts;
5. whether LOC identifiers and captions remain correctly aligned.

The pilot is not training data and cannot promote an algorithm.

### Phase C -- bounded research corpus

Phase C requires an explicit recorded pilot decision. It stays below 1,100
images and 1 GiB, excludes records that fail provenance or derivative-lineage
checks, and never downloads the 59--210 MB archival TIFF collection by
default. Splits must hold out creators and, where recovered, shooting
assignments/LOT groups; duplicate identifiers and perceptual duplicates may not
cross splits.

## Stop rules

Stop before pixels if metadata cannot supply enough named creators, LOC IDs or
rights evidence. Stop after the pilot if content is too confounded with
creator/assignment, derivative edits are common, or the common scanner chain
dominates the observable signal. A successful data gate only enables the RF1
identifiability controls; it does not establish a learnable film transform.

## Primary sources

- LOC collection overview: <https://www.loc.gov/collections/fsa-owi-color-photographs/about-this-collection/>
- LOC guide record: <https://www.loc.gov/pictures/item/93845501/>
- LOC digitization details: <https://www.loc.gov/pictures/collection/fsac/digitizing.html>
- LOC rights and access: <https://www.loc.gov/collections/fsa-owi-color-photographs/about-this-collection/rights-and-access/>
- Commons FSA colour category: <https://commons.wikimedia.org/wiki/Category:Color_photographs_from_the_Farm_Security_Administration>
- LOC Flickr Commons album: <https://www.flickr.com/photos/library_of_congress/albums/72157603671370361/>
