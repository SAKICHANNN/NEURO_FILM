# U7.15C Desktop Batch Representative Path-Disclosure Contract

## Purpose

`U7.15C` repairs a concrete privacy regression in the native batch workflow.
`U7.15A` added an explicit representative selector, but its visible values use
the complete selected path (`001 · C:\\...\\photo.png`). This violates the
existing `U7.10A` desktop invariant that user-visible input identity is limited
to a basename and does not disclose its parent path.

The repair changes only selector presentation. The internal mapping continues
to bind the exact resolved path, and the numeric selection index keeps repeated
basenames unambiguous. It does not change preview pixels, representative
authority, canonical batch order, look strength, export bytes, recipes,
receipts, renderer, profile, output format, or claims.

## Frozen parent and defect

- Parent HEAD: `020bb364d9d86824b8af4bb22418bb7a1e27c9ea`.
- U7.15A evidence SHA-256:
  `029122a26a0331854f92821017a72e52ee1a0cae6f06053495121c9eb47c6da8`.
- U7.15B evidence SHA-256:
  `b472598327a61a5884828a406becb46f51da4369eebe8c061d290ed2a1fa0015`.
- Parent UI Git blob `f34ec4b2fa28e00d13f1919d433156579df04d64`
  contains `label = f"{index:03d} · {path}"`, so every batch member's
  absolute parent is visible in the read-only combobox.
- The existing single-photo U7.10A audit binds
  `absolute_path_disclosed=false`; U7.15C restores that invariant for the
  later batch selector.

Existing `.codex/` and `tmp/` are foreign untracked trees and remain untouched.
Formal scratch may use only an owned repo-relative `tmp/u7_15c_*` subtree and
must remove it.

## Frozen implementation

1. Only `src/inference/product_desktop_ui.py` may change in production code.
2. Each explicit selector label is exactly
   `"{one-based-index:03d} · {path.name}"`.
3. The automatic entry remains exactly `Automatic · canonical first`.
4. The numeric index makes equal basenames in different directories distinct;
   the private label-to-path mapping still resolves the selected exact member.
5. No visible combobox value, `input_text`, status text, or successful
   representative-selection message contains a selected parent path.
6. Invalid/free-form selector text still rejects rather than becoming a path
   authority.
7. Single-photo behavior and the selector's busy/disabled states remain exact.

## Frozen gates

1. The committed parent reproduces disclosure of both distinct selected parent
   directories in combobox values.
2. Current values show only automatic copy or index plus basename, with no
   absolute parent, drive root, or path separator from the parent directory.
3. Two exact files with the same basename in different directories remain
   separately selectable and resolve to the correct internal path.
4. Explicit non-first selection still becomes the input-basis and three-look
   representative, while canonical batch input/output order remains unchanged.
5. U7.15A representative output/recipe/receipt identities and U7.15B visible
   strength semantics remain exact.
6. The product core and renderer are unchanged; source files are immutable.
7. Forward/reverse committed-head reports are byte-identical, all focused and
   parent behavioral suites pass, and owned residue is zero.

## Stop rule and claim ceiling

Any gate failure closes this exact disclosure repair. Do not rescue by showing
relative directories, hashes, thumbnails inside the selector, editable paths,
automatic representative scoring, or another selector widget.

A pass establishes only private Windows/Python Tk basename-only presentation
for the existing explicit batch representative. It is not filesystem secrecy,
a sandbox, public/cross-platform release, calibrated stock response, physical
film reproduction, stock distinguishability, arbitrary-media quality, or
product-value evidence. All three names remain user-selected `film-inspired /
Look Approximation` labels.

## Execution and rollback

1. Commit this contract/config before implementation.
2. Commit the one production-line repair and focused tests separately.
3. Commit the formal controller before clean committed-head execution.
4. Run forward/reverse formal processes, targeted fresh-process parent tests,
   Ruff, compile, JSON, diff, source-immutability and residue checks.
5. Commit evidence and its binding test, then minimally propagate tracker and
   agent log. Do not push.
