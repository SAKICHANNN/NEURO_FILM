# U7.2P Product CLI Research-Dependency Isolation Contract

Status: prospectively frozen before implementation  
Date: 2026-09-01  
Parent: U7.2O first-class product Look Approximation CLI entry

## Product defect

The safe product command currently imports the analytic Y/chromaticity research
profile at module import time. That profile imports a broad `src.eval` graph and
research-only packages such as `pypdf` and its cryptography provider even for
`--list-product-looks` and ordinary `--product-look` renders. A product command
must not require an unrelated research dependency graph merely to start.

## Single permitted change

Remove the two top-level analytic-profile imports from `scripts/render_film.py`
and import their callables only after the already-existing
`--color-engine analytic-y-chromaticity` branch has been selected and its
preconditions have passed. Do not copy or modify the analytic implementation.

No other product, research, profile, catalog, recipe, output, dependency or
claim behavior may change.

## Frozen controls and gates

1. Bind the exact U7.2O contract, implementation, formal runner, evidence and
   both 7,617-byte formal reports before implementation.
2. In fresh subprocesses, install an import blocker for `src.eval`, `pypdf`,
   `cryptography`, `torch`, `torchvision`, `diffusers`, `transformers`, `peft`,
   `wandb` and `pytorch_lightning`.
3. Under that blocker, `--list-product-looks` must succeed byte-exactly to the
   U7.2O discovery output and must import none of the blocked roots.
4. Under that blocker, Velvia 50, Portra 400 and Ektar 100 at look amounts
   `0`, `.5` and `1` must reproduce the exact U7.2O image hashes and normalized
   recipe semantics. Every recipe remains `film-inspired`,
   `claim_calibrated=false`, and profile `safe-rich-product-v1`.
5. The U7.2O full recipe/layers/metrics product bundle remains byte/semantic
   exact under the blocker and leaves zero owned runtime residue.
6. Selecting `analytic-y-chromaticity` without the blocker still exercises the
   existing analytic branch and passes its existing integration suite. With
   the blocker it must fail only when the analytic branch is actually selected,
   before output publication.
7. U7.2O conflict, invalid-value, legacy output, catalog discovery, source
   immutability and predecode behavior remain unchanged.
8. Two complete committed-head formal processes in forward/reverse enumeration
   order must produce byte-identical reports and pass every gate.

## Stop rule

Any product byte, normalized recipe, full-bundle semantic, catalog, legacy,
analytic-route or claim drift is `FAIL_CLOSED`. Do not compensate by weakening
the blocker, deleting analytic support, changing dependencies, changing a
profile, changing output bytes, tolerating recipe drift or adding a wrapper.

## Claim ceiling

A pass proves only that the existing private product CLI no longer imports one
unselected research dependency graph. It does not prove a minimal install,
package, installer, public API, release readiness, cross-platform environment,
calibrated stock response, physical-film reproduction or product value.

