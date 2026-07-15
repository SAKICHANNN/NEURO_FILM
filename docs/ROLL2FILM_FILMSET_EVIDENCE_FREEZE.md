# Roll2Film FilmSet evidence freeze

> Date: 2026-07-15
> Nodes: `ULT > U5.CT2/U5.CT4`
> Decision: **pass for internal paired-blind recipe research**

## What was frozen

The local decompressed FilmSet image tree was inventoried without modifying its
source bytes. The evidence builder at software commit
`ae4ef606007d79054c321af48657a51b467b5f9c` produced four role-isolated,
git-ignored manifests under `outputs/roll2film/filmset_evidence/`.

| Evidence | Observed |
|---|---:|
| Image files | 21,140 |
| Image bytes | 11,262,805,356 |
| Train identities per domain | 4,657 |
| Final identities per domain | 628 |
| Source-only train identities | 2,096 |
| Target-only train identities | 2,096 |
| Internal paired-dev identities | 465 |
| Final-lockbox identities | 628 |
| Source tree SHA-256 | `e47c254ce550099a52518f6863fa28a1f9077e889593c0a1b1cd690cda8f1619` |

The paper-reported 638 test count remains provenance only. The local runtime
contract is 628. Train and test reuse 80 basenames; identities are therefore
namespaced by distributed split. No train/test input payload shares a SHA-256.

## Pair-blind and lockbox result

- train-input duplicate discovery produced 4,637 clusters; the largest has 7 identities;
- exact/dHash/frozen-embedding cross-pool leakage is zero;
- source/target/dev content-ID and duplicate-cluster intersections are zero;
- source training exposes only input images, while target training exposes only the three recipe domains;
- both evaluator manifests are rejected by the training loader;
- the final 628 lockbox contains 2,512 rows and every test payload remains undecoded, unrendered and unembedded (`decoded_test_payloads=0`).

The four physical manifest SHA-256 values match the values declared in
`report.json`:

| Manifest | SHA-256 |
|---|---|
| `source_train.jsonl` | `4788ecffcc0d05df9baf614e159fd5e2bd6f5e5ca76e2bd3109ed993aca6f865` |
| `target_train.jsonl` | `740019507339bb156a7b73061571209145447a3774f1c5319dbc4f9555c0db4f` |
| `internal_dev_lockbox.jsonl` | `fc78bb22e9a36972c405f52fd22d7b505fd179155af294c3718a5b693b61b0a2` |
| `final_628_lockbox.jsonl` | `7bba03a6f9ae4a70c4f0f128bf12b14116918ac7898dc656f324113110d9b0b2` |

A cache-backed repeat produced byte-identical manifests and report. The final
report SHA-256 is
`f3ff191622592996d36033ca2acda10c39510c79f4520f28033b48df2ca471b0`.

## Boundaries and next gate

This passes `U5.CT4` as a data/access foundation. It does **not** validate a
Roll2Film model, physical film response, named stock, public redistribution or
commercial rights. FilmSet is Capture One recipe evidence only, and per-image
release rights remain unresolved.

The official 628 stays sealed until the complete CT5-CT7 policy is frozen. The
next data-bearing work may use only source/target training roles and the
internal paired-dev evaluator. `U5.CT5` remains blocked on the nonlinear CT3
identifiability gate and matched-strength baseline freeze, not on FilmSet data
availability.
