# U7.14B Desktop Canonical Scratch Boundary Contract

## Role

`U7.14B` repairs a concrete product-storage defect. The native desktop launcher
documents `--scratch-root` as a repository-relative generated-artifact root,
but currently accepts any existing directory. A normal or installed-runtime
launch can therefore place preview workspaces on `C:`, a drive root, or another
project instead of the canonical repository `tmp` junction.

The repair is limited to the public desktop entry point. It resolves the
current repository-relative `tmp` entry and accepts only that directory or an
existing descendant after full path resolution. `ProductDesktopWorkflow`
retains its explicit scratch injection for internal tests and private cores;
renderer, decoder, preview, look, strength, export, recipe, receipt and batch
semantics remain unchanged.

## Parent state and ownership

- Frozen parent HEAD: `9ab788dc4999fea054b0d4e76499fca54818a29a`.
- U7.14A remains exact at evidence SHA-256
  `2c8257efb5ae49e779577c61a49a03997857121badb1ea261d4609f17ddb166f`.
- The public desktop entry point is
  `scripts/open_product_desktop.py`, 1,549 bytes, SHA-256
  `c92f0a1c81ca896d7065144961fabdf0183b6a51e3368ac50df651da3ace4064`.
- Root owns the U7.14B contract/config/entry/test/audit/evidence paths and the
  narrow U7.10A launcher-smoke fixture update. Producer and consumer tasks
  reported no overlap.
- Existing `.codex/` and `tmp/` content is foreign. Tests and formal execution
  may create only uniquely named descendants of repository-relative `tmp` and
  must remove every still-owned member.

## Trigger and invariant

The exact defect is reproducible before rendering: pass an existing directory
outside the resolved repository `tmp` tree through `--scratch-root`; the
launcher constructs `ProductDesktopWorkflow` with it and subsequent preview
workspaces are created there.

The frozen invariant is:

> Every product-desktop generated scratch member must enter through the
> repository-relative `tmp` storage binding. A CLI override may select only the
> canonical root or one of its existing resolved descendants.

Validation occurs after argument parsing and before importing `tkinter`,
creating a window, inspecting an input, instantiating the workflow, or creating
scratch members. The error must name the canonical boundary without disclosing
unrelated directory content. The default remains `ROOT / "tmp"`.

## Frozen accepted and rejected controls

Accepted controls:

1. omitted `--scratch-root`, resolving through the current `tmp` junction;
2. explicit repository-relative `tmp`;
3. an existing normal descendant of the resolved `tmp` tree;
4. the same descendant supplied through an equivalent path containing `.`;
5. one ordinary launcher smoke run using an owned canonical descendant.

Rejected before `tkinter` import and input/scratch reads:

1. repository root, `data`, `outputs`, and a sibling of canonical `tmp`;
2. an existing directory on `C:` outside the repository;
3. an existing drive root;
4. `tmp/../outputs` after normalization;
5. a symlink or junction path whose resolution escapes canonical `tmp`;
6. a missing path and a regular file.

Rejected paths and any sentinel files within them must remain byte-exact. No
candidate media, renderer, recipe, receipt or output is read or written in the
negative controls.

## Frozen success gates

1. The defect reproduces from the frozen parent Git object: an external
   existing directory crosses argument parsing and reaches workflow
   construction.
2. The corrected helper accepts exactly canonical `tmp` and existing resolved
   descendants and rejects every frozen escape/control.
3. Rejection happens before `tkinter` import, input inspection, workflow
   construction or scratch publication; foreign sentinels remain exact.
4. The default command and one explicit canonical descendant reach the real
   desktop smoke mainloop and leave zero owned residue.
5. Existing U7.10B/U7.11B formal launcher descendants remain admissible because
   they are under the same resolved repo-relative `tmp` tree.
6. U7.14A, U7.12C/F/G, U7.11A and U7.10A behavioral suites pass in fresh
   processes. Historical evidence stays immutable.
7. Forward/reverse committed-head scientific payloads are exact; source
   bindings, tracked cleanliness and zero owned residue pass.

## Stop rule and claim ceiling

Any gate failure closes this exact entry-boundary repair. Do not rescue it by
allowing `C:`, a drive root, `data`, `outputs`, environment-variable escape,
symlink/junction escape, arbitrary temporary directories, a second fallback
flag, or a machine-specific `P:` path. Do not modify the workflow core,
renderer, source, look, strength, preview, output, recipe, receipt, batch,
installer or storage junction.

A pass establishes only private Windows/Python desktop scratch routing through
the current repository storage binding. It is not a filesystem sandbox,
standalone installer, public release, cross-platform GUI, calibrated camera
rendering, calibrated stock response, physical-film reproduction, stock
distinguishability, HDR/wide-gamut or product-value claim. Every named output
remains `film-inspired / Look Approximation`; AO6 remains only the Velvia 50
display-proxy Look Approximation baseline.

## Commits and verification

1. Commit this contract/config before implementation or formal control runs.
2. Commit the entry repair and focused tests as one reversible leaf.
3. Commit the formal controller before committed-head forward/reverse runs.
4. Run focused and fresh-process parent regressions, Ruff, format, compile,
   JSON, diff and owned-residue checks.
5. Commit evidence/binding test, then minimally propagate tracker and agent log.
6. Do not push.
