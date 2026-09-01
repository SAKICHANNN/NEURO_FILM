# U7.9A — Private Windows Product Runtime Install Contract

## Parent and question

Parent: `ULT > U7 productization > U7.9A`.

The first-class Look Approximation CLI and its exact binary-only dependency set
are proven, but a user must still assemble and invoke them manually. U7.9A asks
whether one standard-library installer can create a repository-bound private
Windows runtime and launcher without weakening source, ownership, claim or
failure-atomicity boundaries.

This is an installation and launchability leaf, not another render wrapper,
colour experiment, stock calibration or public release.

## Frozen inputs

- Windows CPython `3.12.x`, invoked explicitly by the caller;
- `requirements-product-v2.txt`, SHA-256
  `78b2ec79ffd4053c42c90a5db765baf715fed261efa39a5e762e013145718517`;
- the repository-bound `scripts/render_film.py` entry point and its current
  product catalog;
- source parent commit `045d36e05880ef86ca8120ce66240014e14716ac`;
- one caller-selected absent destination under the P-backed repository-relative
  formal scratch root.

## Required behavior

1. Reject a non-CPython-3.12 caller before creating the destination.
2. Reject an existing destination of every entry type without changing it.
3. Claim the destination once, create a private virtual environment, and install
   exactly the pinned product requirements with `--only-binary=:all:`. A caller
   may supply an explicit local wheelhouse; normal index access remains an
   explicit installation-time dependency.
4. Run `pip check`, import every required product distribution, and verify that
   the research-only OmegaConf/ANTLR packages are absent.
5. Publish a deterministic `product-runtime.json` receipt and a Windows command
   launcher that invokes the repository-bound product CLI from any current
   directory. The launcher must not imply portability: it binds the exact
   repository path, source parent commit and requirements hash.
6. Before every launch, reject repository HEAD drift and requirements drift.
   `--list-product-looks` must work without media; an exact tiny RGB input must
   render all three available looks through the launcher with the same output
   bytes as the direct installed interpreter.
7. On any failure, remove only the still-owned installation directory. If the
   destination path is replaced after creation, preserve the foreign
   replacement and report the displaced owned directory rather than deleting
   either one.
8. The installer, launcher and formal audit may write only to the caller's
   explicit destination and P-backed repository-relative scratch/output roots.
   Source files are immutable and no C-drive or drive-root artifact is allowed.

## Frozen gates

- exact requirements/source identities and CPython 3.12;
- destination create-only behavior;
- successful binary-only install, `pip check`, required-import and forbidden-
  dependency checks;
- launcher execution from a foreign current directory;
- deterministic catalog JSON and direct-versus-launcher equality for all three
  tiny product renders;
- exact Look Approximation claim labels in catalog and receipts;
- existing-destination, requirements-drift, HEAD-drift and injected-install-
  failure rejection;
- late foreign replacement preservation and zero still-owned failure residue;
- forward/reverse report byte identity after normalization of timing only.

Failure closes this exact installer design. Do not rescue by changing Python,
dependency versions, package source policy, repository binding, roles, controls
or gates after observing the result.

## Claim ceiling and stop rule

A pass establishes only a private Windows CPython 3.12 repository-bound install
and launch mechanism for the existing deterministic `film-inspired / Look
Approximation` product. It is not a standalone/portable installer, signed
package, public release, license clearance, GUI, calibrated stock response or
physical-film reproduction. The repository has no root license, so public
distribution remains fail-closed. Stop adjacent bootstrap/launcher variants
after this exact leaf; the next product leaf must address a different observed
user-flow, performance, compatibility or safety defect.
