# SF3.A3W — Inland Aperture Portra 400 / Ektar 100 source admission

## Role

This is a prospective zero-pixel source-admission audit for seven fixed Flickr
photo pages published by one author, Matt / Inland Aperture. The page metadata
explicitly labels four photographs as Kodak Portra 400 and three as Kodak
Ektar 100, all with Canon EOS1 / EOS 1, and exposes a per-work CC BY 4.0
license declaration.

The leaf asks whether this exact small same-author, same-camera, cross-stock
source is mechanically and legally addressable. It does not test colour,
grain, a stock response, an operator or a product profile.

## Frozen source and operations

1. Read only the seven exact Flickr photo-page HTML documents in the frozen
   forward or reverse order.
2. Parse the selected photo model already embedded in each HTML document and
   require the exact ID, title, description, license code, original dimensions
   and date fields in the frozen canonical manifest.
3. Independently parse the page's JSON-LD `ImageObject` and require its exact
   CC BY 4.0 URL, acquire-license page, author name and author URL.
4. Treat embedded `contentUrl`, thumbnail and original URLs only as strings.
   Do not request them, issue HEAD/range requests, or decode pixels.
5. Canonicalize rows by photo ID so forward and reverse request orders must
   serialize to identical reports.

## Admission gates

The source-structure gate requires all seven fixed pages, four Portra and three
Ektar labels, one exact author, one exact Canon EOS1 camera declaration and
seven per-work CC BY 4.0 declarations.

Stock-evidence admission separately requires independent authors/sources,
explicit roll/process/scanner identities, same-scene neutral/film pairing and
a sealed confirmation role. Geographic and trip adjacency are content/source
nuisance, not a same-scene pair or an independent group.

## Stop rules

- Do not request any `live.staticflickr.com`, buddy-icon or other media URL.
- Do not infer roll, process or scanner from upload order, dates, camera text,
  filenames, locations or shared authorship.
- Do not treat Portra/Ektar photographs from nearby locations as paired.
- Do not combine this single-source pool with already failed natural-label
  association families to rescue stock identifiability.
- Do not fit, render, score, tune a threshold or open candidate three.

## Claim ceiling

At most this can establish a publicly addressable, per-work CC BY 4.0,
same-author/same-camera Portra 400 and Ektar 100 appearance-source manifest.
It cannot establish source-independent stock identity, a stock response,
digital-to-film mapping, calibration, learning eligibility, three-stock
evidence or product capability.
