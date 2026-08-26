# P237 — InRetouch RTD Paired-Source Eligibility Audit

## Position

P237 is a metadata-only latest-source gate under the bounded final-cycle
authority. The WACV 2026 InRetouch Retouch Transfer Dataset (RTD) is relevant
because one Lightroom preset is applied across many different MIT-Adobe FiveK
scenes, which could support a source-independent shared operator. Relevance does
not establish product eligibility.

## Frozen question

Before any dataset file, pixel, preset, checkpoint, or model is downloaded, does
the exact official RTD release simultaneously provide:

1. a publicly enumerable and reproducibly identified payload;
2. rights permitting commercial model/product research and derived weights;
3. explicit scene and preset grouping sufficient for file-disjoint,
   preset-disjoint development and confirmation roles;
4. authoritative file sizes and content hashes for bounded acquisition; and
5. a non-gated acquisition path that does not require accepting additional
   unseen terms or sharing contact information?

All five conditions are mandatory. Failure of one condition closes product-cycle
admission before pixels. Academic/private feasibility may be recorded but cannot
increment or reopen candidate 3.

## Fixed official sources

- GitHub repository `omarAlezaby/InRetouch`, default branch HEAD.
- Hugging Face dataset `omaralezaby/Retouch_Transfer_Dataset`.
- WACV 2026 paper/project page only for method and grouping facts.

## Allowed reads

- Git refs, GitHub repository metadata, README and LICENSE text.
- Hugging Face dataset API/model-card metadata and file descriptors.
- Paper/project-page HTML or PDF metadata.

Dataset payload bytes, images, presets, weights, archives, LFS objects, training,
inference and pixel decode are forbidden.

## Decision

- `PASS_PRIVATE_SOURCE_ELIGIBLE_FOR_SEPARATE_PIXEL_PREREGISTRATION` only if all
  five mandatory conditions pass twice with identical normalized reports.
- Otherwise `NOT_ELIGIBLE_PRODUCT_CYCLE_SOURCE_RIGHTS_OR_PAYLOAD_GAP` and no
  pixel leaf opens.

No license reinterpretation, alternate mirror, manual gate acceptance, author
contact, row substitution, or private credentials may rescue this leaf.

## Claim ceiling

Metadata-only public-source and rights eligibility. No data quality, preset
authenticity, operator recovery, A1/A4/A5, package, capability, product, film or
stock claim.
