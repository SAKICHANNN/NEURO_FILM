# U4.2A Current Three-Look Style/Appeal Contract

## Question

Do the unchanged, user-selectable Velvia 50, Portra 400 and Ektar 100
`film-inspired / Look Approximation` outputs provide visible product value over
an exact identity control on the current nine-input U4.1A gold set?

This leaf measures autonomous visual evidence only. It is not a population
preference study, a stock-identification test, a target-film comparison, a
calibration result or a product-promotion decision. RF3.D15's failure to prove
robust three-stock distinguishability remains authoritative and immutable.

## Frozen parents and arms

- U4.1A evidence commit: `4af18939d6567a96144d8df218974c0a64859227`.
- U4.1A formal-source commit: `7938967269f7b08672cf6e0384eb95eefd58bcd5`.
- Accepted U4.1A report SHA-256:
  `90ae1784c24d855f888d1f91110d46ed37e23c3c80e7f9a8dca0ee96a58ff5b5`.
  Its formal status is `PASS_OPEN_AUTONOMOUS_VISUAL_REVIEW`.
- Final U4.1A adjudication evidence SHA-256:
  `af1ee870de1f749a62d2a39ca1f519fc66724fa7cf2d1e0d4c1d7755ac00972a`,
  with status
  `PASS_NO_CONFIRMED_SEVERE_ARTIFACT_ACROSS_81_ARM_ROTATED_GOLD_VERDICTS`.
- Frozen samples: `01`, `05`, `08`, `09`, `11`, `18`, `21`, `29`,
  `FS_FACE_01`.
- Candidate arms: the exact U4.1A `velvia_50`, `portra_400` and `ektar_100`
  outputs, without rerendering.
- Control arm: the same product profile and input-decoding path at
  `look_amount=0`, written as PNG with all optional effects disabled. This is
  an exact-identity product-path control, not a neutral film stock.

## Blinding and observation

1. Build exactly three deterministic rounds. Each sample receives a fresh
   permutation of four anonymous labels `A`-`D`.
2. The private label mapping remains unread until all 108 observations are
   durably recorded.
3. Each observation records `severe` (`no`, `yes`, or `uncertain`), film-style
   strength (1-5), and photographic appeal (1-5).
4. Review happens from anonymous sheets only. No extra round, revote, threshold
   change, rerender, look-amount change or post-reveal editing is allowed.
5. Any `yes` or `uncertain` is reported and cannot support the product-value
   pass. U4.1A's existing original-resolution severe result is retained rather
   than silently overwritten.

## Frozen gates

The report is complete only when all 108 observations exist and the two fresh
formal processes are byte-exact. Each named Look must independently pass all
of the following:

- zero `severe=yes` and zero `severe=uncertain` observations;
- median-of-sample style score at least `3.0`;
- median paired style advantage over identity at least `1.0` point;
- paired style advantage at least `1.0` on at least `6/9` samples;
- median paired appeal advantage over identity at least `0.0` points;
- appeal is at least identity on at least `6/9` samples.

The portfolio passes only if all three named Looks pass. A partial result is
reported per Look and does not justify relabeling, dropping or tuning a failed
arm inside this leaf.

## Stop rule and claim ceiling

Failure closes only the current v1 product-value claim on this population. It
does not reopen the exact cohort for tuning and does not stop the long-lived
Look Approximation product Goal. A later challenger must be separately
versioned and use newly frozen development/confirmation roles.

The maximum positive claim is: three unchanged deterministic product Looks
show autonomous style/appeal value over an exact identity control on this
nine-input gold set while retaining the existing severe-artifact pass. It is
not human preference, general product quality, stock distinguishability,
calibrated stock response, physical-film reproduction or authenticity.
