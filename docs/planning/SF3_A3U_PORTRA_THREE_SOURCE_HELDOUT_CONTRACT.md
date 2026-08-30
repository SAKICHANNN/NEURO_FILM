# SF3.A3U — Portra three-source held-out identifiability

## Role

This is a prospective, pixel-level source-held-out diagnostic for current
Kodak Portra 400. It exists because three independently published source
families are now available under explicit reuse terms:

1. the already consumed and author-balanced Wikimedia Commons SF3.A1D2 pool;
2. the R1HG `nicknicknicknick.net/on-film-2023` CC BY 4.0 family, which also
   supplies a same-author Kodak Ultramax 400 wrong-stock control; and
3. the unread Luminant CC BY 4.0 Portra inventory.

The leaf asks whether the unchanged SF3.A1E low-capacity RGB-distribution
descriptor, trained only on Commons Portra-versus-Ektar, generalizes to the
unseen nicknick author/source and beats luma, HOG, low-frequency colour,
standardized-colour and geometry/source controls. Luminant is a gated
positive-only second confirmation and cannot provide specificity.

This is not an operator fit, a same-scene digital-to-film mapping, a calibrated
stock response or a product experiment.

## Frozen roles

- **Development, already consumed:** the exact 21 clean Commons Portra 400 and
  22 clean Commons Ektar 100 rows frozen by SF3.A1E. No Commons reselection is
  allowed.
- **Primary unseen confirmation:** nicknick indices `39..50`, labelled exactly
  `Portra 400`, plus twelve Ultramax controls selected from indices `11..38`
  by ascending `SHA256(canonical_url)`. The older/ambiguous `Portra Pro 400`
  indices `01..10` are excluded from every metric and remain unread.
- **Conditional second confirmation:** only if every primary nicknick gate
  passes, select twelve Luminant items labelled exactly `Kodak Portra 400` by
  ascending `SHA256(five_digit_id)`. `Portra 400` and `Kodak Portra 400 +2`
  labels are excluded. Luminant is scored for positive recall only.

All newly selected media must be downloaded through repository-relative
`data/...`, rehashed into a local manifest and decoded only after its source
identity, licence, byte count and content type pass. Media roles are determined
before the first body request.

## Frozen method and controls

The descriptor family and image normalization are inherited unchanged from
SF3.A1E (`src.real_film.connected_stock_identifiability.extract_descriptors`).
For each descriptor, standardization statistics and class centroids are fit on
Commons development rows only. No parameter, feature, threshold or role may be
chosen from nicknick or Luminant pixels.

The primary descriptor is `rgb_distribution`. Controls are:

- `luma_distribution`;
- `hog_grayscale`;
- `low_frequency_rgb_4x4`;
- `standardized_rgb_distribution`;
- `geometry_border_source`;
- deterministic development-label permutation; and
- reversed Portra/Ektar class centroids.

The nicknick primary result must pass every frozen recall, specificity,
balanced-accuracy, permutation, nuisance-delta, HOG-ceiling, finite, source
integrity and replay gate. Failure closes before any Luminant body request.
Passing nicknick opens only the twelve-row Luminant positive-recall check.

## Stop rules

- Do not read nicknick `01..10` or unselected Ultramax bodies.
- Do not read any Luminant body unless nicknick passes every primary gate.
- Do not infer roll, process, scanner or physical Portra generation from URL
  order, page sections, camera text or author identity.
- Do not alter the SF3.A1E descriptor family, class model, thresholds, sample
  counts or roles after pixels are read.
- Do not rescue a failure with another classifier, embedding, crop, colour
  space, threshold, stock control or additional source.
- Do not fit or promote a colour operator, candidate three, calibrated profile
  or product capability from this result.

## Claim ceiling

At most, a pass establishes a bounded private appearance association for
current-labelled Portra 400 that generalizes from the fixed Commons source to
one unseen same-author Portra-versus-Ultramax source and retains positive
recall on one further Portra-only source. It does not establish the physical
stock response, scanner-independent colour, roll/process transfer, a digital
to film operator, calibration, product authenticity or completion of the
three-stock programme.
