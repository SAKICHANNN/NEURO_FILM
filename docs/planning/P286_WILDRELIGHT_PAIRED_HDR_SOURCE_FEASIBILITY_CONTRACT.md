# P286 WildRelight paired-HDR source feasibility contract

## Question

Does the official 2026 WildRelight release provide a rights-compatible,
exactly inventoried and group-isolatable physical observation pairing real HDR
photographs with co-located HDR environment maps and capture metadata?

## Frozen official sources

- Hugging Face dataset `Lez/wildrelight`, revision
  `90ab579145da9f3ea998d706c8defd0d4b87641f`.
- Revision-pinned repository API, recursive expanded tree, `README.md`, and one
  fixed metadata witness `small-aligned/bench2/meta.json`.
- Official project/paper identity: WildRelight, CVPR Findings 2026 / arXiv
  `2605.11696`.

Only repository and file metadata plus the two named UTF-8 text files may be
requested. EXR/DNG/image bodies, pixels, code, models, training and inference
are forbidden in P286.

## Frozen source gates

All gates must pass simultaneously:

1. dataset id, revision, public/ungated state and README identity are exact;
2. README binds the complete dataset to CC BY 4.0;
3. the recursive manifest is complete, non-truncated and every large file has
   exact size plus LFS SHA-256 identity;
4. the manifest contains exactly 30 scene groups in aligned/unaligned and
   original/small variants, with paired photo/envmap EXRs and per-scene JSON;
5. the metadata witness exposes scene id, reference shutter, HDR method,
   shooting times, ISO, exposure times, source DNG names and alignment method;
6. photograph and environment-map time roles are matched and contain at least
   three capture times;
7. total public payload size is at most 210 GB and a future bounded acquisition
   can select one `small-aligned` scene without archive download;
8. two fresh forward/reverse reports are byte-identical.

The release contains merged HDR EXRs and metadata naming source DNGs, but no DNG
payloads. Passing P286 therefore cannot establish RAW-bracket availability.

## Stop rule and claim ceiling

Failure of any gate yields
`FAIL_CLOSED_WILDRELIGHT_PAIRED_HDR_SOURCE_FEASIBILITY` and opens no payload
read. Passing yields only
`PASS_PRIVATE_WILDRELIGHT_PAIRED_HDR_SOURCE_FEASIBILITY` and permits a separate
prospective, hash-selected, bounded one-scene source-observation D0.

P286 cannot establish relighting/HDR quality, scene radiometry, independent
generalization, a shared colour operator, A1/A4/A5, public package/schema/
capability/product admission or candidate 3.
