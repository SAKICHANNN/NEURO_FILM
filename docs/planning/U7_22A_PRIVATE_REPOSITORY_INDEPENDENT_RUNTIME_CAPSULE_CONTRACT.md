# U7.22A Private Repository-Independent Runtime Capsule Contract

Status: frozen before implementation.

## Parent and observed product gap

U7.9A-D and U7.20F provide a verified private Windows runtime, but both its
CLI and desktop launchers still require the original source checkout, live Git
history, and a clean runtime-source scope. Moving or removing the checkout
therefore makes the installed product unusable even though its pinned Python
environment remains intact.

U7.22A tests one additive, create-only source capsule inside the exact existing
P-backed U7.21E runtime. It reuses that runtime's interpreter and distributions;
it must not create another virtual environment, copy datasets or research
outputs, modify the old receipt/launchers, choose a project licence, or authorize
public distribution.

## Frozen source and layout

- parent runtime receipt: `outputs/private-product-runtime-u7-21e-8f07545/product-runtime.json`,
  3,726 bytes, SHA-256
  `548a80583811329260083bd37d0abd122b033141b1c358ef700f07ee044ea756`;
- parent runtime Python: `runtime/Scripts/python.exe`, 274,424 bytes, SHA-256
  `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`;
- current tracked-clean committed source is the sole build input;
- all tracked `src/**/*.py` files plus
  `scripts/pipeline_color_baseline.py` are stored in one deterministic ZIP;
- the CLI and desktop entry points, pinned requirements, the two render profiles,
  colour statistics, guardrails, legacy profile YAML, and runtime-scope config
  are the only separately materialized source files;
- `tmp/` is an empty owned scratch directory inside the capsule;
- the capsule destination must be absent and a normal directory after creation.

## Required behavior

1. Before creating the capsule, verify the exact parent receipt, parent Python,
   receipt schema/claim, runtime containment, installed distribution versions,
   current requirements identity, tracked-clean Git state, and P-backed parent.
2. Read packaged source from committed Git blobs, not untracked or modified
   working files. The archive and every external file are listed by relative
   path, byte length, and SHA-256 in canonical `runtime-source-capsule.json`.
3. Generate deterministic isolated Python and native Windows launchers. Before
   importing product code they must verify the manifest, archive, external
   files, parent receipt, parent Python, and containment, then run only the
   capsule entry point with the capsule archive first on `sys.path`.
4. Repository execution remains governed by the existing Git checks. Capsule
   identity is accepted only when the root has no `.git` entry and the isolated
   launcher supplies the exact manifest path and hash. An environment variable
   must never bypass a real checkout's Git checks.
5. Product code must revalidate the complete capsule identity at session start
   and immediately before every existing publication boundary that currently
   revalidates repository scope. Recipe `software.commit` remains the committed
   source SHA recorded by the capsule manifest.
6. From a foreign current directory, with the original checkout made unavailable
   to the child process, the capsule must list the exact catalog and render all
   three available Look Approximation outputs. Output, recipe, replay, and claim
   bytes/fields must equal a direct committed-source oracle.
7. CLI and desktop smoke must reject archive, manifest, external-config, parent
   receipt, and parent-Python drift before product code or pixel decode. Existing
   destination, late foreign replacement, and injected build failure preserve
   foreign entries and remove only still-owned capsule residue.
8. The accepted build must add materially fewer than 100 materialized files and
   less than 64 MiB logical data. It may not create another environment or write
   C-drive/generated drive-root artifacts.

## Frozen gates and stop rule

All identity, containment, deterministic-archive, create-only, isolated-startup,
foreign-CWD, checkout-unavailable, catalog, three-look render, recipe, replay,
desktop-smoke, tamper, cleanup, file-count, size, and claim gates must pass.

Any failure closes this exact capsule design. Do not rescue it by copying the
whole source tree, creating a second environment, weakening hashes or Git-mode
checks, allowing dirty sources, changing product pixels/recipes/looks, selecting
a licence, or adding data/model assets.

## Claim ceiling

A pass establishes only one private P-backed Windows runtime tree that remains
usable without the original repository. It is not a public or redistributable
package, signed installer, hostile-host sandbox, cross-machine/cross-platform
proof, calibrated stock response, physical-film reproduction, or legal
clearance. Every output remains `film-inspired / Look Approximation`.
