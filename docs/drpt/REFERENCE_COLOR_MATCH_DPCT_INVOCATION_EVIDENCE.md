# Fixed D-PCT Invocation Consumer Evidence

Date: 2026-07-28  
Node: P43A-D  
Status: exact-wheel local candidate invocation complete

## Result

Neuro-Film can now invoke the fixed producer package without importing the
mutable D-PCT checkout. The consumer verifies the wheel filename, size and
SHA-256 before any temporary transaction exists, verifies the exact runtime,
safely expands those wheel bytes into a temporary private directory, writes a
domain-separated canonical request, runs the package and independently
reconstructs the response, artifacts and existing P27 candidate.

The package remains a local research candidate. This result does not satisfy
A1/A4/A5, promote an algorithm, authorize P30, stage P33, apply FilmFX or
deliver a product file.

## Pinned authority

| Item | Identity |
|---|---|
| producer stable snapshot | `e22725d` |
| package source commit | `01ef0616610b16246623236301234ff3b4c4a7f2` |
| installed fixture commit | `eb4b889bc18f1aba078e19e7e62fbf8c4add6e54` |
| producer package lock | `300b95b05954b8f1de7aaec6f86429bf584a40088e15d07dd439384c73090287` |
| exact wheel | 95,994 bytes; `fd995ad88c9f30f2136508f7c3879ce6768fde7ff2e149537d2b58f78e36c292` |
| request schema | `bac2866842d31c1d42e8b813da5c6632bdb9de49ef296e7542209639aa9b37b8` |
| response schema | `204da4f668e69beba7705f8ab0d7966d962c5d6933870bde481bbaee8a15473e` |
| installed fixture | `eab24eaed81e3602de519ce4ba89e640bf8185a452d4104b2577b54fdf15b295` |
| consumer lock | `df39b9f8905c5e53ec653cc62e8edf6ce013f3eddc859dcbd724da43a2452301` |
| consumer lock schema | `eaa2fa49228975c9435fd34b0fdf653f45ef05043255a1a98484144ef3d0d29b` |
| consumer adapter | `db52a210449d2148c2e630deb0dd3b10d8cf9c5101c9145f0aeaa9c7ed137398` |

The runtime is exactly CPython 3.12, NumPy 2.4.4 and Pillow 12.1.1. The sole
mapped capability/profile is
`zhuise.dpct-chroma.cpu-reference.v1` /
`zhuise.display-linear-srgb-d65-relative-f32.v1`.

P27 `neuro-film.dpct-consumer.v2` remains the lower envelope compatibility
authority. P43 adds invocation identity and never relabels the P27 producer
commit alias as the package source commit.

## Fail-closed evidence

- a stale same-name 94,548-byte wheel is rejected before scratch mutation;
- an unpinned Python runtime is rejected before invocation;
- request IDs bind exact source/reference views and change with source bytes;
- output directory files must be exactly response, payload and output for a
  candidate, or response alone for a failure;
- response/package/request IDs, source/reference views, artifact filenames,
  sizes and hashes are independently checked;
- P27 independently rebuilds transform, diagnostics, output, result and
  consumer receipt;
- every temporary package/request/output byte is removed on return.

Direct ZIP import initially failed because producer build fingerprinting reads
its source files. The final consumer path therefore safely expands the already
verified wheel into its temporary transaction, matching installed-wheel
semantics without trusting an installation directory or mutable checkout.

## Verification and propagation

- 24 dedicated/adjacent invocation and P27 tests pass.
- 187 D-PCT/core/product-chain tests pass.
- 438 combined color-match/FilmFX tests pass.
- Full suite: 1333 passed, one skipped and the unchanged 36 isolated-worktree
  output/asset failures; no P43/color-match/FilmFX failure.
- Latest main: `2afa6c07c76bd5faea7bca9baac13a785e51a60c`.
- Common base: `c03c321b9fc642e2e092d59e20dd1b145b96192d`.
- Consumer/main changed paths: 199/141 with zero exact overlap.
- Clean merge tree: `1ce7695f0a75aaab12974cd45d24412b74d2e2e7`.
- A fresh detached synthetic merge passes 38 P42/P43 focused tests and was
  removed. Main's concurrent dirty research files were read-only.

Producer later advanced to `e73ad14` for ROGR development research; no
invocation schema/package change was observed or consumed.

## Remaining boundary

The producer repository has no project-level release grant, so the wheel is
not a redistributable product dependency. There is no native ABI or
Android/macOS/iOS runtime evidence. Most importantly, the invoked candidate
has not passed genuine A1/A4/A5; all product authorization and delivery paths
remain fail-closed.
