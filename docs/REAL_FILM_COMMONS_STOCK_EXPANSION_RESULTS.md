# SF0.6A Commons exact-stock expansion results

The ten-category metadata sweep produces one new pass: Kodak UltraMax 400.
Together with SF0.5 Ektar100, the program now has two provisional `S0` source
passes, still one short of the minimum comparative stock set. No pixels were
downloaded and no training opens.

- 10 categories / 393 current file rows;
- snapshot SHA-256 `59820e7deea0880f37f6c5dfcf62e03656ee1f71e9e231fe0fcf021eb24c5ecb`;
- two byte-identical offline audits, report SHA-256 `54abe0f394c669f1120ab0de9939e9c3e0eb0ad71d50b31e8903e79100524f00`;
- UltraMax 400: 84 files, 15 normalized authors, 35.71% largest author and 51 strict derivative-rights rows.

The failures are informative rather than arbitrary. Fuji C200 has seven
authors but only five derivative-rights rows; ProImage 100 has 30 strict rows
but one author contributes 66.67%; Superia Reala has only three authors;
Provia 100F and Sensia 100 have good author counts but only two strict rows
each. Uploaders were never substituted for authors.

`SF0.6B` next audits exact Kodachrome 25/64, Ektachrome Elite 100/200 and
Vision3 50D/250D categories. At least one must pass the same true-author and
derivative-rights gates before any second pixel pilot is frozen.

Machine decision:
`configs/real_film_commons_stock_expansion_decision_v1.json`.
