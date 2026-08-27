# P266 DMEB multi-view HDR source readiness contract

## Question

Does the latest official ECCV 2026 DMEB release provide a rights-compatible,
exactly inventoried and independently executable synchronized multi-view
exposure-plus-depth observation that can open a separate prospective physical
HDR experiment?

## Frozen official sources

- Project/repository: `divisonofficer/dmeb`, commit
  `559e52f4c4bc02a1285b5491fcacb5cd4c5844eb`, tree
  `8e81f34d70fbfd0aa18484b9077c314d1787c50d`.
- Commit-pinned `CITATION.cff`, `LICENSE`, `dataset/DATASET_CARD.md`,
  `dataset/benchmark_protocol.md`, `code/README.md`,
  `code/checkpoints/README.md`, and `docs/index.html`.
- The paper-level observation is materially different from prior after-only or
  temporal-flow leaves: synchronized cameras receive different exposures and
  are geometrically aligned using calibrated depth before confidence-weighted
  HDR fusion.

Only bounded GitHub API/raw responses may be requested. Google Drive folders,
archives, manifests, images, depth, checkpoints, third-party backbones, pixels,
training and inference are forbidden in P266.

## Frozen readiness gates

All gates must pass simultaneously:

1. repository commit/tree and citation identity are exact;
2. official sources define synchronized multi-view varying-exposure inputs,
   exposure/gain metadata, depth, calibration, valid masks and HDR references;
3. code has explicit commercial-compatible rights covering the executable
   reference implementation;
4. every required dataset subset has explicit commercial-compatible rights;
5. current public sources expose exact scene/frame counts rather than template
   placeholders;
6. exact public archive inventory and checksums are present before payload
   access;
7. the reference checkpoint has a concrete public locator and checksum;
8. official scene/session splits can support group-disjoint development and
   confirmation;
9. two fresh forward/reverse reports are byte-identical.

The statement that archives will contain manifests is not an exact public
inventory. Public Google Drive links are locators, not proof of anonymous
member access, identity or licence completeness. An absent or noncommercial
licence is not permission.

## Stop rule and claim ceiling

Failure of any gate yields
`NOT_READY_DMEB_RIGHTS_INVENTORY_OR_CHECKPOINT_GAP_NOT_SCIENTIFIC_RESULT`.
It authorizes no Drive request, payload/model download, pixel read, training,
inference, mirror use, rights inference, candidate-3 consumption or product
mapping.

A complete pass could open only a separately preregistered payload-integrity
and source-observation preflight. P266 cannot establish HDR quality, arbitrary
RAW/HDR support, a shared operator, A1/A4/A5, package/schema/capability or
product admission.
