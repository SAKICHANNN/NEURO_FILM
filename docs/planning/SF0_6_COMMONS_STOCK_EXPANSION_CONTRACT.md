# SF0.6 exact-stock source expansion contract

## Question

Can current exact-stock Commons categories yield at least two additional
rights-complete, true-author-diverse 1600px derivative lanes, so Ektar is no
longer the only passing stock and a later comparative identifiability test is
scientifically meaningful?

The sweep is metadata-only. The ten candidates are Kodak UltraMax 400,
ProImage 100 and Vision3 500T; Fuji C200, Provia 100F, Sensia 100, Sensia II
100, Superia 200, Superia Reala and Industrial 100. Portra is excluded because
the large Commons parent is mixed and the exact Portra 160/400 subcategories
have only one direct file each. B&W stocks remain a separate future mode.

## Frozen gates

Each candidate independently requires:

- at least 20 current files and five uploaders;
- no uploader or normalized visible author over 60%;
- at least five normalized author groups;
- at least eight strict pixel candidates with explicit CC0/CC BY licence URL,
  or public-domain usage terms, complete source metadata and a 1600px URL that
  differs from the original URL;
- at least 95% with short dimension >=512, no more than 10% frozen non-scene
  title flags, and complete Commons SHA1/dimensions/source/free-licence fields.

Uploader diversity never substitutes for author diversity. Parent categories,
family labels, camera simulations, LUTs and visual guesses do not count as an
exact stock.

## Stop and continuation rules

No pixels are fetched during the metadata sweep. If fewer than two new stocks
pass, close those sources and continue source discovery without training. If
at least two pass, freeze a separate derivative-only manifest capped at 128
files/512 MiB and 12 rows per author and uploader; then repeat the SF0.5
integrity/content/visual gates. No GPU or colour fit is authorized here.

Machine contract:
`configs/real_film_commons_stock_expansion_audit_v1.json`.
