# U7.21B Desktop Runtime-Scope Session Binding Contract

## Problem

The native desktop seals eleven selected renderer files and the current Git
commit when previews are created.  The installed launcher separately protects
the complete U7.9D product runtime scope (`src`, `configs`, both product
entrypoints and `requirements-product-v2.txt`), but that launcher validation
ends after the long-lived desktop process starts.  A later uncommitted change
to an unsealed decoder, encoder or other runtime module can therefore retain
the same `HEAD`, pass the current preview-session check and enter export while
the strict recipe still records the old commit.

This is a preview-to-export provenance defect, not an algorithm or format gap.

## Frozen scope

U7.21B may modify only the private native-desktop session validation in
`src/inference/product_desktop.py`.  It must reuse the existing authoritative
U7.9D runtime scope and may add dedicated tests, audit and evidence.  It must
not change renderer pixels, input/output formats, look definitions, strength,
effects, recipes, receipts, UI controls, runtime installation or claims.

## Required behavior

1. The default desktop runtime scope is read from the exact U7.9D scope
   contract; a second divergent scope list is forbidden.
2. Before preview pixels are decoded, every tracked file in that scope must
   equal the session `HEAD`, and no untracked member may exist in the scope.
3. Preview reads, detail inspection, single export and batch export must repeat
   the same validation against the unchanged session `HEAD`.
4. A modified, staged, deleted or untracked file anywhere in the scope rejects
   before export command execution and preserves the current preview workspace
   for inspection/cleanup under its existing ownership rules.
5. Changes outside the U7.9D runtime scope, including documentation, evidence
   and tests, remain allowed when `HEAD` is unchanged.
6. The existing individual file seals remain defense in depth and injected
   test bindings retain their explicit behavior.
7. All existing Velvia 50, Portra 400 and Ektar 100 output bytes, desktop
   behavior and `film-inspired / Look Approximation` claims remain unchanged.

## Stop rule

Any product-output drift, acceptance of runtime-scope drift, rejection caused
only by out-of-scope files, weakening of individual seals, new product feature,
or material preview/export latency regression closes U7.21B without expanding
the scope, suppressing Git errors, hashing only a subset or rewriting history.

## Claim ceiling

A pass establishes only private same-session product-source consistency for
the repository-bound Windows/Python desktop.  It is not a sandbox, hostile-host
guarantee, standalone/public release, cross-platform guarantee, calibrated
stock response or physical-film reproduction.
