# RF1.1 FILM-R signal/content/nuisance audit results

## Decision

`structurally_unidentified`. FILM-R remains useful as a real-film unknown-look,
damage and artifact stress set, but it cannot identify a reusable filename-
family colour signal under the frozen content controls.

No new image feature pixels were decoded and no classifier was trained. The
metadata/content support gate failed first, as preregistered.

## Reproducibility

- 44/44 visual content labels exactly cover the integrity manifest.
- Dataset decision, integrity report and both review-sheet hashes verified.
- Report: `outputs/real_film/filmr_v2/signal_audit/report.json`.
- Report SHA-256:
  `88c3d771617469a624e57729320e63462332ca0525f605cbd747e2679c055ba8`.
- Complete rerun was byte-identical.
- Software commit: `2a8304f9a7f42093677572a779edb27180d7f412`.

## Family-by-content matrix

| Filename family | Total | Product/equipment | Urban/architecture | Landscape/nature | Other/interior |
|---|---:|---:|---:|---:|---:|
| Cinestill 50D | 1 | 1 | 0 | 0 | 0 |
| Cinestill 800T | 17 | 16 | 0 | 0 | 1 |
| Ektachrome 100 | 3 | 3 | 0 | 0 | 0 |
| Ektar 100 | 1 | 0 | 1 | 0 | 0 |
| Gold 200 | 2 | 0 | 1 | 1 | 0 |
| Portra 160 | 2 | 1 | 1 | 0 | 0 |
| Portra 400 | 4 | 2 | 1 | 0 | 1 |
| Provia 100 | 4 | 0 | 1 | 2 | 1 |
| Superia 400 | 1 | 0 | 0 | 1 | 0 |
| Ultramax 400 | 1 | 0 | 1 | 0 | 0 |
| Velvia 50 | 8 | 0 | 2 | 5 | 1 |

The family/content association is large: Cramér's V is 0.6298. The descriptive
chi-square p-value is 0.00698, though small expected cell counts make effect
size and the structural matrix more important than that p-value.

Five filename families have at least three pairs. Under the frozen requirement
of two samples in each of two content categories, only Velvia 50 crosses
content. The largest pairwise comparable clique—Cinestill 800T, Ektachrome 100
and Portra 400—shares only the product/equipment category; none of those three
can be tested across content. There are zero mutually comparable cross-content
families.

## Why classifier training stops

A random image split would reward recognition of film boxes, cameras and
landscapes. A grayscale content classifier could therefore look like a film-
family classifier. Because the support matrix cannot estimate family effects
after holding content out, colour-only versus structure-only accuracy would not
repair the causal ambiguity.

This is a data-design failure, not a software or ML failure. Training on the 44
images would average or memorize the observed content and would not answer the
Ultimate question.

## Propagation

- Keep FILM-R for real-scan appearance, damage, restoration nuisance and severe-
  artifact stress tests.
- Do not train filename-family, named-stock, Roll2Film or retrieval experts from
  FILM-R alone.
- RF2 remains gated on a larger, content-balanced real-film source with honest
  source/roll holdouts.
- RF0.3 becomes the next active leaf: metadata-first Apollo/DOCUMERICA and other
  obtainable real-film source audit, followed only by bounded justified data
  acquisition.
