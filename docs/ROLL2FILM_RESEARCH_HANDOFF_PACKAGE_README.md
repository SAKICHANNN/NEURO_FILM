# Roll2Film Web Research Handoff Package

Upload this package together with `PROMPT_CN.md` to the web researcher.

## Read first

1. `PROMPT_CN.md`
2. `CHECKSUMS.csv`
3. `visual_history/ALL_SCHEMES_NUMBERED.png`
4. `visual_history/NUMBER_MAP.csv`
5. `normalized_anchors/manifest.json`
6. `red_highlight_counterfactual/README.md`
7. `roll2film_e0/ROLL2FILM_CT1_E0_FOUNDATION.md`

## Evidence boundaries

- Historical scheme numbers are display identifiers, not independent style classes. `02`, `03`, and `33` are smoke-only cues.
- Normalized anchors use identical frozen inputs and disable grain. They are anchor-inspired replays, not pixel-identical historical outputs.
- Red-highlight diagnostics locate possible speckle but do not constitute visual adjudication.
- E0 is a restricted synthetic SPD-affine pass, not real-film, named-stock, general transfer, per-photo adaptation, or product-safety evidence.
- FilmSet is a Capture One recipe dataset, not physical film truth.
- `CHECKSUMS.csv` records every packaged relative path, byte count and SHA-256.

## Current-machine correction

The Windows workspace checked on 2026-07-15 contains the full FilmSet tree: 21,140 files / 11,262,805,356 bytes. Every train directory has 4,657 files and every test directory has 628. Older documents saying 638 test images conflict with the archive and with `5,285 - 4,657 = 628`; the researcher must resolve this against primary sources.
