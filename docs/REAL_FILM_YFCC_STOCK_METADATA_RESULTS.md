# SF0.8A YFCC15M exact-stock metadata results

Date: 2026-07-16  
Decision: **new source passes candidate discovery; no pixels or learning are open yet**.

## Frozen source

YFCC100M is a Flickr/Yahoo research release whose rows carry a Creative Commons licence and user-supplied metadata. The bounded source here is the public `mehdidc/yfcc15m` Parquet conversion of the CLIP YFCC subset, not the complete YFCC100M population. Ten Parquet files were frozen: 7,350,000 rows and 1,738,329,293 bytes. No images were downloaded or decoded.

Primary context: [YFCC100M paper](https://doi.org/10.1145/2812802), [official Flickr description](https://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-creative-commons-100-million-dataset/), [bounded subset page](https://huggingface.co/datasets/mehdidc/yfcc15m).

The local manifest SHA-256 is `b6e507005540769badcf0a52eff3eebbd63415df215f3c7770b8b34f6652d581`. The exact-phrase audit was executed twice and produced the identical report SHA-256 `32f5e2865bddd0e6b4ddfffd455adfd65de1d27dd0567fbca649d6e97fc33ef9`.

## Result

Only static-photo rows licensed under CC BY 2.0 were considered. Exact stock phrases were searched in user title, description and tags after a frozen punctuation normalization. The unchanged gate was at least eight rows, at least five Flickr author UIDs and no UID above 60%.

| Candidate | Rows | Author UIDs | Largest share | Raw gate |
|---|---:|---:|---:|---|
| Fujifilm Velvia 50 | 51 | 26 | 13.73% | pass |
| Kodak Gold 200 | 74 | 16 | 29.73% | pass |
| Kodak Portra 160 family string | 58 | 9 | 43.10% | pass, generation-mixed |
| Kodak Ektachrome E100VS | 35 | 10 | 37.14% | pass |
| Fujifilm Pro 400H | 30 | 8 | 30.00% | pass |
| Fujifilm Pro 160S | 12 | 6 | 25.00% | pass, process-contaminated |
| Kodak Portra 400 family string | 11 | 6 | 27.27% | pass, generation-mixed |

The remaining seven configured stock strings fail at least one unchanged gate. These counts are source-discovery evidence only. A UID is a grouping key, not proof that two accounts are two people.

## Adjudication and next gate

Velvia 50 is selected because it has the strongest author diversity, is directly relevant to the target looks and does not require merging NC/VC or other Portra generations. A prospective metadata exclusion removes explicit cross-processing, E-6-as-C-41, HDR/tonemapping, multiple exposure and B&W developer rows. Six rows are excluded, leaving 45 rows across 23 UIDs with a 15.56% largest share.

SF0.8B may select at most 32 rows and at most four per UID. Before any pixel is retained it must reverify the live page, direct image response and licence; then pass decode, exact/perceptual duplicate, author/content concentration and repeated visual severe-artifact review. Training remains forbidden. If live attrition takes the pool below eight rows or five UIDs, the pilot stops rather than weakening gates.
