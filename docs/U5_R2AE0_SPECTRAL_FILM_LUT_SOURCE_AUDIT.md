# U5.R2AE0 — spectral_film_lut source and profile-lineage audit

## Decision

**An isolated synthetic operator-bank audit is feasible; profile truth,
redistribution and product reuse remain blocked.**

The pinned `spectral_film_lut` revision
`02ecafd78c4a97bd0708d69a9f5ec39caf492d2e` is a distinct deterministic
spectral simulator. It exposes a headless `film_conversion` / `create_lut`
path and separates camera negative, print, reversal and grain stages. The
repository is MIT-licensed and its current source tree contains embedded
digitized curve data for a broad set of named materials.

This is useful as an independent algorithm comparator, but it is not a
verified stock-response bank. The current `FilmData` schema has no source URL,
document identity, page/figure coordinate, digitization method, uncertainty
or per-profile licence field. Some profiles explicitly describe their data as
unreliable, experimental or less accurate. A current named object therefore
cannot be promoted merely because its module is present.

## Exact inventory

At the pinned revision:

- the four exported lists contain 93 entries but only 86 distinct objects and
  names;
- seven objects are duplicated across or within lists: Kodak 5222, its four
  development variants, Kodak EXR 100T 5248 and Kodak Vision 2383;
- the underlying modules contain 66 base `FilmData` constructions:
  42 camera-stage and 24 print-stage;
- all 66 provide characteristic curves, 61 provide log sensitivity, 42
  provide spectral density and 38 provide MTF fields;
- the public lists contain 46 negative, 7 reversal, 37 print and 3 reversal
  print entries before identity de-duplication.

The package runs without its GUI. An isolated Python 3.14 runtime generated
finite float32 5-cube outputs for Ektar 100 + Endura Premier and direct
Velvia 50. This proves only headless executability, not numerical correctness.

## Historical datasheet lineage

The complete Git history contains 31 datasheet PDFs totalling 22,442,549
bytes immediately before commit
`887caa50f3e54d4ea0eed846d498967a466e2d65` removed the directory on
2026-06-21. For example, the original Ektar profile commit added
`Kodak_Ektar_100.pdf` and its Python module together. The deletion commit says
only `remove datasheets`; no licence, provenance or profile-to-figure mapping
was added in its place.

This historical association improves auditability for a subset of profiles,
but it does not establish permission to redistribute the manufacturer PDFs or
their digitized values, and it does not cover all current profiles. The
historical objects remain in the ignored external checkout only. No PDF,
profile array, LUT or external implementation is copied into K-MCFM.

## Non-duplicate next leaf

AE1 may run one source-bound, synthetic-only structural diversity audit:

- use the exact pinned external revision through the headless API;
- restrict the primary bank to camera/output chains for which the Git history
  contains identifiable datasheet artifacts for every material in the chain;
- use a fixed synthetic RGB cube and fixed pipeline parameters;
- de-duplicate exported objects by identity and name before selection;
- require two independent complete runs and exact serialized-array replay;
- measure range, non-finite values, clipping, monotonicity/folding and
  pairwise operator separation after matching simple global/basic controls;
- include identity, duplicate-object and strength/basic-transform negative
  controls;
- freeze candidate chains, thresholds and failure branches before inspecting
  the resulting operator metrics.

AE1 may answer only whether this external simulator supplies a reproducible,
structurally nontrivial comparison bank. It cannot identify a real
digital-to-film operator, validate named-stock authenticity, fit current
pixels, train a router, act as a teacher or authorize product integration.
Profiles without exact historical document association, including current
UltraMax 400 and Velvia 50 entries, remain outside the primary source-bound
bank even though the code can execute them.
