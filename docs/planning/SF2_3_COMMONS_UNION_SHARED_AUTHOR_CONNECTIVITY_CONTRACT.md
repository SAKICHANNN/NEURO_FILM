# SF2.3 Commons-union shared-author connectivity contract

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.3`

Status: frozen before implementation or formal execution

## Question

Do the three already frozen Wikimedia Commons metadata snapshots contain a
redundant exact-stock/shared-author graph strong enough to justify a later,
separately frozen live page, label and rights preflight?

This is a metadata-only connectivity audit. It does not reopen SF0.4-SF0.7,
does not use their pixels, and cannot establish stock appearance or
identifiability.

## Parent evidence and development/confirmatory split

- Commons and YFCC pixel pools are closed because content, scene colour and
  source fingerprints dominate stock labels.
- The earlier Commons audits evaluated each acquisition batch and stock
  independently; they did not union the three immutable snapshots into a
  cross-stock author graph.
- A read-only development census saw 294 strict scene rows, 30 normalized
  author strings, 17 exact-stock categories and eight strings occurring under
  multiple stocks. Most observed stock pairs had only one author and one
  author formed a large star.

Those numbers establish DoR only. They were seen before this freeze and are
explicitly forbidden as confirmatory evidence. The formal audit must use this
committed contract/config without changing thresholds after its result.

## Frozen inputs and filtering

Use only the three SHA-256-pinned SF0.4/SF0.6A/SF0.6B snapshots and their
source configs. No network, page or image request is allowed.

A strict row must come from an `exact_stock_community_category`, satisfy the
per-source config's permissive derivative licence list, have a nonempty
normalized author, source/original/1600px derivative URLs, a derivative URL
different from the original, and a licence URL or public-domain usage terms.
Rows matching the source config's frozen non-scene title patterns are removed.

Author normalization reuses the conservative Commons function: unescape HTML,
strip tags, casefold and collapse whitespace. Raw normalized strings remain
separate identities. No author may be inferred from uploader, and no apparent
name reversal or other alias may be merged without independent profile/source
evidence frozen in a future contract. The current alias map is empty.

## Frozen support, redundancy and identity gates

An individually eligible stock needs at least eight strict rows, five authors
and no author above 60%, reusing the established Commons source thresholds.

A passing component additionally needs all of:

1. at least three eligible exact stocks;
2. at least five shared authors overall;
3. at least two authors on every retained stock-stock edge;
4. at least two shared authors incident to every component stock;
5. at least three retained edges and cycle rank at least one;
6. no one shared author supporting more than 40% of retained edges;
7. zero cross-stock overlap in page ID, file-page URL, original URL or API SHA1.

The two-author edge, cycle and author-edge-share rules prevent a single
photographer star from masquerading as leave-one-author-out connectivity.
Graph support is still only feasibility: stock may remain aligned with an
author's trip, date, content or scan workflow.

## Branches

- **Pass:** open only a separately frozen bounded Commons live page, exact
  label, rights and author-alias preflight for the best component. It does not
  authorize pixels.
- **Insufficient connectivity:** close this union as a data-gap result; do not
  merge unverified aliases, relax graph gates or download pixels.
- **Identity overlap or input mismatch:** fail closed and preserve the
  deterministic evidence.

Every branch keeps operator fitting, training, LSM and stock claims false.

## DoD and claim ceiling

Commit this contract/config before auditor code. The implementation must hash
all six immutable inputs, validate fail-closed flags, run twice byte-identically,
report every stock/edge/component and identity conflict, and have focused tests
for pass, insufficient graph, alias non-merging and cross-stock overlap. After
formal execution, propagate the result and run the full CPU suite.

At most SF2.3 can establish that frozen Commons metadata has a conservatively
normalized, redundant shared-author graph suitable for later live verification.
It cannot establish person identity, current rights, physical stock use,
content balance, stock signal, an operator, latent modes, `S1/S2`, calibration,
authenticity or product value.
