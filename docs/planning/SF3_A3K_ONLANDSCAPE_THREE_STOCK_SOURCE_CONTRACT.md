# SF3.A3K — On Landscape three-stock source admission

## Question

Does the public *Colour Film Comparison Pt. 3* page expose a same-scene
physical-film observation for Velvia 50, Portra 400 and Ektar 100 with rights
that permit the controlled stock-specific fitting programme?

## Frozen observation

- article: `https://www.onlandscape.co.uk/2011/06/colour-film-comparison-pt-3/`
- publisher terms: `https://www.onlandscape.co.uk/terms-and-conditions/`
- author/publisher: Tim Parkin / Landscape Media Limited
- scene: one Bamburgh Castle large-format sequence developed by Digitalab and
  scanned on a Howtek drum scanner
- required exact stock labels: `Velvia 50`, `Portra 400`, `Ektar`

The article states that the film sheets depict one scene, while also recording
minor time/light changes between exposures. Transparency scans are described
as calibrated; negative scans use operator-selected shadow/highlight points and
some displayed variants add curves, gradients or noise reduction. These are
separate scan/interpretation nuisances, not stock response.

## Execution order and gates

1. Request only the article HTML and official terms HTML.
2. Verify the exact page identity frozen in the config and the required
   capture/develop/scan statements.
3. Parse the comparison widgets into `(group, asset, stock label)` tuples and
   require all three target stocks in the same `all` group.
4. Verify that the terms explicitly permit dataset extraction, operator
   fitting, derived parameter/weight use and redistribution or commercial
   product use. Private/business research access to one copy is insufficient.
5. If the rights gate fails, stop before every image request, pixel decode,
   fit, render or score.

## Stop rules

- Do not register, subscribe, log in, accept new terms or contact the publisher.
- Do not request any comparison image, thumbnail or downloadable scan in this
  leaf, even when the URL can be constructed from the public HTML.
- Do not infer identical illumination, exposure, negative inversion, roll,
  process or scanner response beyond the article's statements.
- Do not treat an internal-research permission as training, redistribution,
  released-parameter or commercial-product permission.
- Failure closes this exact source without relaxing rights or substituting a
  different Portra/Ektar/Velvia asset.

## Claim ceiling

At most, this leaf can establish a public metadata-level observation that one
same-scene physical comparison contains the three requested stock labels. It
cannot establish a rights-cleared dataset, stock response, calibration,
operator fit, product profile or publication/release permission.
