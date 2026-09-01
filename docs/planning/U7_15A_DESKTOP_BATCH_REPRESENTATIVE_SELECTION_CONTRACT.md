# U7.15A Desktop Batch Representative Selection Contract

## Role

`U7.15A` repairs one product-decision limitation in the existing private native
desktop batch.  U7.11A canonically sorts one to 100 selected photos and always
uses the first canonical path as the only preview representative.  This is
deterministic and visible, but the user cannot deliberately choose a more
informative member before selecting one Look Approximation for the whole batch.

The repair adds one explicit representative choice among the currently selected
photos.  The representative authorises only the already-existing input-basis
and three-look preview.  Canonical binding, child render order, output names,
look amount, output format, image bytes, strict recipes and aggregate batch
receipt remain unchanged.  Callers that omit the new optional argument retain
the exact historical canonical-first behaviour.

## Parent state and ownership

- Frozen parent HEAD: `ff1c0d720a9afaf86b440ee5b6b86d240c323b50`.
- U7.14B remains exact at evidence SHA-256
  `da8a3918d2bc5dbf6189d15c6a44fa8928ec40ea0cc93031602de934b9cc7ab0`.
- Core baseline `src/inference/product_desktop.py`: 52,965 bytes, SHA-256
  `149ed425c0770afcbbb1d33f36ba42341545d635d860c391a6fb45f42b45f587`,
  Git blob `dac918b7f110a478c23dbea8286b8b28eaa634fa`.
- UI baseline `src/inference/product_desktop_ui.py`: 31,225 bytes, SHA-256
  `85ef883d03393135c39618ba3553aaa38398591a4635182b7110625f4677c417`,
  Git blob `7ef2bf0ee2a9657ac526929f003fe717c3db856c`.
- Root owns this contract/config, the two product modules, one focused test,
  later audit/evidence files and only strictly required parent-test edits.
  Producer/consumer collaborators reported no exact overlap.
- Existing `.codex/` and `tmp/` content is foreign and must remain untouched.

## Frozen interface and behaviour

1. `ProductDesktopWorkflow.render_batch_previews` gains a keyword-only optional
   `representative_path`.  `None` preserves the canonical-first member exactly.
2. A supplied representative is resolved and must match exactly one already
   bound batch member by canonical path and filesystem identity.  A missing,
   duplicated, aliased, external or post-selection changed member rejects before
   preview render.
3. `export_batch` accepts a preview state bound to any still-valid member of the
   exact canonical input tuple.  It continues to revalidate every input before
   every child and publication.
4. The native UI exposes a read-only keyboard-reachable representative selector
   only when two or more photos are selected.  Labels remain unambiguous even
   when basenames repeat.  Changing it invalidates any existing preview and look
   selection and requires a fresh preview.
5. Single-photo behaviour, omitted argument behaviour and the previous
   canonical-first batch path stay byte compatible.

## Frozen success gates

1. The parent limitation reproduces: reversed selection order still previews
   the canonical first path.
2. An explicit non-first member becomes the input-basis and three-look preview
   source while the returned batch tuple remains canonically ordered.
3. Batch export after that preview keeps the exact historical child order,
   output/recipe bytes and aggregate receipt identity for the same sources,
   look, amount, format and destination identity model.
4. Batch-external, missing, identity-aliased and mutated representatives reject
   before preview renderer or publication; sources and sentinels remain exact.
5. The UI selector is visible/read-only/keyboard reachable only for a batch,
   names the chosen member, and a real selector change clears preview authority,
   selected look and export eligibility.
6. Single-photo and omitted/default core/UI paths remain unchanged.
7. U7.10A, U7.11A, U7.12A/B/C/F/G and U7.14A/B behavioural suites pass in
   fresh processes; forward/reverse committed-head scientific payloads are
   exact; tracked state and owned residue gates pass.

## Stop rule and claim ceiling

Any gate failure closes this exact representative-selection route.  Do not
rescue it by changing canonical child order, receipt schemas, preview geometry,
look math, amount, renderer, encoders, recovery, cache, scratch policy or
publication semantics.  Do not infer that one selected photo represents the
whole batch aesthetically; the choice is explicit user authority only.

A pass establishes only private Windows/Python batch preview selection for
deterministic `film-inspired / Look Approximation` output.  It is not calibrated
camera rendering, calibrated stock response, physical-film reproduction,
stock distinguishability, automatic aesthetic routing, arbitrary-media quality,
installer, public release, cross-platform GUI or product-value evidence.

## Commits and verification

1. Commit this contract/config before implementation or formal controls.
2. Commit core/UI/test changes as one reversible implementation leaf.
3. Commit the formal controller before committed-head forward/reverse runs.
4. Run focused and fresh-process parent suites, Ruff, format, compile, JSON,
   diff and owned-residue checks.
5. Commit evidence/binding tests, then minimally propagate tracker/agent log.
6. Do not push.
