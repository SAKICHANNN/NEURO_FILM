# U7.13A — private Linux product CLI runtime contract

## Question

Can the unchanged first-class K-MCFM product CLI run under the already-present
WSL2 Ubuntu 22.04 x86_64 CPython 3.12 runtime, using the same twelve pinned
product dependency versions as Windows, and reproduce deterministic,
strictly replayable Velvia 50, Portra 400 and Ektar 100 Look Approximation
renders from one exact real JPEG with decoded RGB16 parity to Windows?

This is a cross-platform product-mechanics question. It is not a request to
change renderer math, profile parameters, output encoding, the Windows
runtime, the desktop UI or the public dependency set.

## Frozen source and runtime

- Distribution: WSL2 `Ubuntu-22.04`, x86_64.
- Interpreter: `/usr/bin/python3.12`.
- Product dependencies: the exact twelve names and versions already present in
  `requirements-product-v2.txt`, materialized only from the exact official
  PyPI wheel files locked in the U7.13A config.
- Input: exact U4.1A gold row `01`, a real JPEG already present under the
  repo-relative `outputs/...` junction. Its path, size and SHA-256 are locked
  before any U7.13A render.
- Looks: exact currently available `velvia_50`, `portra_400` and `ektar_100`,
  each at `look_amount=1.0` through `safe-rich-product-v1`.
- Output: PNG16 plus strict recipe. AO6 is not executed and remains only a
  Velvia 50 display-proxy Look Approximation baseline.

The wheel archive is project-owned and P-backed through the repo-relative
`data/vendor/linux/u7_13a-py312/wheels` path. Execution roots and reports use
repo-relative `tmp/` and `outputs/`; no durable data or generated artifact is
written to C: or a drive root.

## Frozen execution

1. Before acquisition, validate the config, official PyPI URLs and the
   project/source bindings that do not depend on the eventual formal commit.
2. Acquire only the twelve locked wheels. Every file must match its frozen
   filename, byte length and SHA-256 before it can be used.
3. From one tracked-clean committed HEAD, build a fresh Linux virtual
   environment in an owned P-backed temporary root and install with
   `--no-index --find-links` from the locked wheel archive. `pip check`, exact
   versions and the absence of OmegaConf/ANTLR must pass.
4. In both forward and reverse look order, run each look once with the Windows
   project environment and once with the fresh Linux environment, then replay
   every strict recipe on its originating platform to a second create-only
   output.
5. Hash complete PNG and recipe files, decode RGB16 and embedded ICC, normalize
   only machine/run-specific recipe paths and output-byte identities for the
   cross-platform semantic comparison, and remove owned execution roots.
6. Persist source facts and reports only after every frozen check is computed.

## Gates

All gates are binding:

- exact contract/config/source/profile/assets/CLI identities;
- exact twelve wheel filenames, sizes, SHA-256 values and installed versions;
- exact WSL distribution, architecture and CPython 3.12 runtime;
- `pip check` passes and OmegaConf/ANTLR remain absent;
- the three catalog rows remain available and retain their current
  `film-inspired / Look Approximation` claim ceilings;
- Windows and Linux each produce one complete PNG16 plus strict recipe per
  look and replay each recipe byte-exactly on its originating platform;
- forward/reverse complete PNG, decoded sample, ICC and normalized recipe
  identities are exact within each platform;
- Windows-versus-Linux decoded RGB16 samples and embedded ICC are exact for all
  three looks; cross-platform compressed PNG byte equality is diagnostic only;
- every output remains finite after decode and introduces zero exact 0/65535
  boundary samples relative to the current product contract;
- unavailable `generic_bw` rejects before input decode/publication on Linux;
- source bytes and tracked worktree remain unchanged;
- formal network reads are zero and owned environments, reports, media and
  stage files leave zero residue after accepted evidence is bound.

## Stop rule

Any wheel, runtime, install, import, source, catalog, render, replay,
determinism, decoded-sample, ICC, boundary, invalid-control, ownership or
cleanup mismatch closes this exact U7.13A leaf. Do not change dependency
versions, Linux distribution, interpreter, source, look, amount, output
format, tolerance, recipe normalization, gate or role after a result. A failure
does not invalidate Windows product evidence and does not open a rescue leaf.

## Claim ceiling

A pass proves only private WSL2 Ubuntu 22.04 x86_64 CLI mechanics for one exact
real JPEG and the three unchanged deterministic product Look Approximation
choices. It does not prove arbitrary Linux distributions/devices/inputs,
cross-platform GUI or installer, byte-identical compressed PNG interchange,
public packaging/release, population preference, stock distinguishability,
calibrated stock response, physical-film reproduction or authenticity.
