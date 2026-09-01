# U7.18A Desktop Fast Export Policy Contract

## Purpose

`U7.18A` integrates the already-proved exact bounded tile-parallel tier into the
actual private desktop launcher. The current desktop uses tile size `256` and
one worker for both small previews and full-resolution exports. Existing
U7.2B/U7.2D evidence establishes exact 8-worker execution and a measured 512
tile at 24MP, but the user-facing desktop does not select that tier.

This leaf keeps preview work at `256/1` and changes only formal export work
(single photo, batch children, and exact 1:1 detail) launched by
`scripts/open_product_desktop.py` to `512/8`. PNG compression remains `6`.
There is no new user control, scheduler, algorithm, format, or claim.

## Parent and ownership

- Frozen parent HEAD: `68e4503613701a662d407b268a2809256208b79a`.
- Parent `src/inference/product_desktop.py` Git-object SHA-256:
  `3caa294dfe2317629c2dcbce0fb77be574241dba6d5161f19d0511ededb356de`.
- Parent `scripts/open_product_desktop.py` Git-object SHA-256:
  `316413043cc43c204f5bb2a95007ad49e01c0e12f345aef279646edbd0bfc1fe`.
- U7.2B parallel-tile evidence Git-object SHA-256:
  `6cd0c404ac5148c579c9f3e6f3eead1d39002933b2cf3d945c76ca1a63fd1a5a`.
- U7.2D 512/8 export evidence Git-object SHA-256:
  `2035428421149ffc6b19f23abd425963403e98c67bc4e810306b6a071f9743c3`.
- U7.17A finishing-control evidence Git-object SHA-256:
  `4ef53dda83152ba3dccd9a516ed4ce074c36e5df3716c9e245926da67dd8f7eb`.
- Root is sole writer/integration owner. Consumer explicitly reported no path
  overlap. Existing `.codex/` and `tmp/` entries are foreign and untouched.

## Frozen behavior

1. Preserve the existing `tile_size` and `tile_workers` constructor parameters
   as preview policy and as the export policy for callers that do not opt in.
2. Add optional positive `export_tile_size` and `export_tile_workers` values.
   `None` means exact legacy fallback to the preview values. Bool and nonpositive
   values reject during workflow construction.
3. Preview rendering always receives the preview policy. Export command
   construction always receives the resolved export policy.
4. The real desktop launcher alone opts into export `512/8`; it retains preview
   `256/1` and PNG compression `6`.
5. Single export, batch children, and U7.16A/U7.17A exact detail use the same
   resolved export policy through the existing command builder. No separate
   detail or batch exception is allowed.
6. Output pixels, encoded PNG16 bytes, ICC, Look, amount, effects, input hash,
   claim ceiling, create-only behavior, cleanup, and strict replay remain exact.
   Recipe execution metadata may differ only in the frozen tile size/workers.
7. Existing tests/audits that instantiate the workflow without export-specific
   values retain their command and output identities exactly.

## Frozen formal roles and gates

The primary source is the already-consumed 4032x6048 JPEG
`data/preference/repid/u5_r2repid3_shared_operator_v1/fit/winner/a4593-kme_0276.jpeg`,
SHA-256 `7ea3d1ed37b7df518c0466c1379f08aa34ea5ae00eb7dc8208fcff2cfe68d1c6`.
Use Ektar 100 Look Approximation, amount `.65`, PNG16 compression `6`, and the
U7.17A maximum deterministic effects (`grain=.05`, simple halation `.15`,
`dust=.02`, seed `7`).

Two committed-head controllers run the prospective order in both directions:
baseline `256/1`, candidate `512/8`, candidate `512/8`, baseline `256/1`; the
reverse controller reverses that sequence. Each output is newly created below
the repository-relative P-backed `tmp/` root and removed after verification.

Success requires:

1. Parent Git objects and prior evidence bindings are exact and tracked diffs
   are clean before each accepted controller.
2. Preview manifests and commands retain `256/1` while every single, batch, and
   detail export command uses candidate `512/8` only when explicitly configured.
3. Baseline and candidate PNG16 files, decoded RGB16 samples, ICC bytes, and
   normalized non-execution recipe semantics are exact; strict replay of each
   recipe reproduces its own output exactly.
4. Both candidate outputs repeat byte-exactly; input bytes remain immutable and
   all owned scratch/media/recipe artifacts are removed.
5. Candidate median wall time divided by baseline median wall time is at most
   `.85` in both forward and reverse controllers.
6. Candidate peak process-tree RSS is no more than baseline peak plus
   `536870912` bytes and no more than `4294967296` bytes.
7. Constructor invalid controls reject, legacy fallback is exact, and the
   U7.10A/U7.11A/U7.12B/C/G/U7.14A/U7.16A/U7.17A behavioral suites pass.
8. Accepted forward/reverse scientific reports are byte-identical after
   removing order/timing observations; Ruff, format, compile, JSON, diff, and
   evidence-binding checks pass.

## Stop rule and claim ceiling

Any gate failure closes U7.18A and the implementation is reverted completely.
Do not rescue by trying worker counts, tile sizes, compression levels, formats,
effects, Looks, strengths, sources, thresholds, native/GPU kernels, caching, or
new schedulers. Do not expose a resource-tuning UI or open another adjacent
tile-performance leaf.

A pass establishes only a private local Windows/Python desktop export policy
for deterministic `film-inspired / Look Approximation` output. It is not
calibrated stock response, physical-film reproduction, stock distinguishability,
cross-platform performance, public release, arbitrary-host performance, or a
new colour/effect algorithm. AO6 remains only a Velvia 50 display-proxy Look
Approximation baseline.

## Execution

1. Commit this contract/config before implementation or formal execution.
2. Commit the narrow workflow/launcher integration and focused tests.
3. Commit the formal controller, then run forward/reverse from committed HEAD.
4. Commit evidence/binding tests and minimally propagate tracker and AGENT_LOG.
   Do not push.
