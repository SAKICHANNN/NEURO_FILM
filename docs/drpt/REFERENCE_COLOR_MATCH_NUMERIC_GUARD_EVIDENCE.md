# Reference Color Match Exact-Receipt Numeric Guard Evidence

Date: 2026-07-28
Node: P29A-D
Status: stable consumer candidate; no delivery or applied state

## Boundary

P29 consumes only the exact relative display-linear sRGB P27 receipt and the
P28 ordered one-reference/N-source resolution. It does not invoke D-PCT,
decode media, write image files or establish visual/aesthetic safety.

The consumer retains producer DiagnosticsV2 facts without reconstruction:

- out-of-gamut fraction before output policy;
- fraction of channel samples actually changed by clipping;
- non-clipping projection fraction.

It adds one consumer-owned metric: the fraction of output pixels newly placed
on a `[epsilon, 1-epsilon]` boundary relative to the exact source pixels.

Default v1 limits are:

| Metric | Maximum |
|---|---:|
| producer out-of-gamut fraction | 0.25 |
| producer clipping fraction | 0.05 |
| producer projected fraction | 0.25 |
| consumer new-boundary fraction | 0.05 |
| boundary epsilon | `1 / 65535` |

Equality passes. Non-finite values, invalid fractions, unsupported schemas,
identity mismatch or any exceeded limit fail closed.

## Single-source contract

`CoreNumericGuardDecisionV1` revalidates the prepared source/reference,
producer compatibility pin, exact output receipt, acceptance and admission
bindings before reading pixels. It binds policy, admission, receipt, source
and output identities into a canonical decision ID.

Its only actions are `eligible-for-transaction` and `identity-fallback`. Its
fixed claim ceiling is `numeric-only-no-visual-claim`.

Schema SHA-256:
`4178f792256ef0c85f99e7b873865355687d47d4783f12118fa06ec252676964`.

The corrected producer success fixture reports 25% clipping, so it fails the
default 5% policy. A deliberately relaxed policy is used only to prove the
success path. This prevents the conformance vector from silently becoming a
product threshold waiver.

## Atomic batch contract

`CoreNumericBatchGuardV1` accepts decisions only when P28 is
`pending-product-guard`. It requires:

- exactly one decision per contiguous source row;
- exact source, receipt and admission identity binding;
- original user order;
- one shared numeric policy;
- an already accepted A1/A4/A5 admission for every row.

All rows passing yields only `eligible-for-transaction`. One row failing
makes the complete batch `identity-fallback`; no partial eligibility exists.
An upstream P28 fallback forbids numeric decisions and remains a not-evaluated
full fallback.

Schema SHA-256:
`77a6049d3fb3bebd77efe31034430461d20edf7dbcc92436a8beeb1daf0355b4`.

## Verification

Adversarial coverage includes threshold equality, source/output pixel
mutation, source/decision swaps, policy/reason/action/claim/ID mutation,
missing and reordered decisions, mixed policies, upstream short-circuit,
strict JSON and schema roundtrip.

- 65 combined P27-P29 focused tests pass.
- Full suite: 1218 passed, one skipped, 36 unchanged environment failures.
- No colour-match or P29 test fails.
- Latest main is `a33526e`; common base is `c03c321`.
- Consumer/main changed-path overlap is zero.
- Merge tree: `39d578eca046bb5eb3e87c3e37cf8e605d15b23d`.
- Detached synthetic merge `d8ae8caa...` passes all 65 selected tests.
- The temporary merge worktree was removed after verification.

## Producer propagation

D-PCT later proved its separate absolute display-linear BT.2020 HDR Sparks
arithmetic on Windows x64 at producer HEAD `6c7118c`, including exact
Python/native output identity. That rail remains explicitly unmapped:
arithmetic portability is not a relative-SDR colour-state bridge, JSON ABI,
mobile/Apple runtime result or Neuro-Film product admission.

P29 therefore changes no producer schema, producer threshold, HDR mapping,
media decoder, FilmFX path or main-worktree file.
