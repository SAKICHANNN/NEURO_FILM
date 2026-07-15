# Roll2Film FilmSet manifest and lockbox contract

> Date: 2026-07-15
>
> Nodes: `ULT > U5.CT2/U5.CT4`
>
> State: frozen implementation contract; generated manifests remain ignored

## Purpose

Use the local FilmSet archive for honest paired-blind recipe-transfer research
without allowing a training process to recover aligned source/target identities
or inspect the official 628 final targets before the complete policy is frozen.

FilmSet is Capture One recipe data. Nothing in this contract promotes it to
physical film, stock, process, scanner or named-stock truth.

## Local source

Expected image root:

```text
data/raw/filmset/FilmSet/
  train/{input,Cinema,ClassNeg,Velvia}/
  test/{input,Cinema,ClassNeg,Velvia}/
```

Observed archive contract:

- 4,657 identities in every train domain;
- 628 identities in every test domain;
- exact basename parity across the four domains within each distributed split;
- 21,140 image files and 11,262,805,356 bytes in the decompressed tree;
- paper-reported test count 638 is provenance only; runtime count is 628.

Any mismatch fails closed and writes no final manifests.

## Generated artifact separation

All generated files live under ignored
`outputs/roll2film/filmset_evidence/`:

```text
inventory_cache.jsonl       # resumable append-only file metadata cache
source_train.jsonl          # input images only; source-only content clusters
target_train.jsonl          # recipe targets only; target-only content clusters
internal_dev_lockbox.jsonl  # paired train identities, evaluator-only
final_628_lockbox.jsonl     # official test identities, final evaluator-only
report.json                 # counts, hashes, leakage gates and provenance
```

The source and target training manifests never contain the same content or
duplicate-cluster ID. Training code must not accept either lockbox manifest.
Lockbox separation is an application-level capability boundary, not an OS ACL;
the final evaluator still needs an isolated process/access log before use.

## Inventory row schema

Every finalized row contains:

| Field | Meaning |
|---|---|
| `schema_version` | `roll2film.filmset_manifest.v1` |
| `dataset_id` / `archive_version` | stable FilmSet source identity |
| `distributed_split` | `train` or `test` from the archive |
| `domain` | `input`, `cinema`, `classneg` or `velvia` |
| `content_id` | normalized basename identity; never used to create cross-pool pairs |
| `duplicate_cluster_id` | cluster frozen before pool assignment |
| `research_pool` | `source_train`, `target_train`, `internal_dev_lockbox` or `final_628_lockbox` |
| `path` | path relative to the FilmSet image root |
| `bytes` / `mtime_ns` / `sha256` | resumable integrity evidence |
| `width` / `height` / `icc_sha256` | decoded only for internal train input canaries; final test payload stays sealed |
| `dhash64` / `visual_embedding_hash` | train-input duplicate/leakage canaries |
| `allowed_use` | `internal_research_only` |
| `redistributable` | always `false` pending per-image release clearance |
| `payload_access` | training role or evaluator-only lockbox role |

Official test rows may be byte-hashed for integrity, but the evidence builder
must not decode, render, embed or visually inspect their payloads.

## Pair-blind split

Only the 4,657 distributed-train input identities are used to discover duplicate
clusters. Cluster evidence is built from:

1. exact input SHA-256;
2. 64-bit difference hash;
3. a deterministic low-resolution RGB embedding projected to a frozen vector
   space, with projection seed/hash recorded in `report.json`.

Clusters, not individual files, are assigned deterministically to:

- source-only training, target proportion 45%;
- target-only training, target proportion 45%;
- internal paired dev lockbox, target proportion 10%.

Source training exposes only `input`. Target training exposes only the three
recipe domains. Internal dev contains all four domains but is rejected by the
training manifest loader. Near-duplicate evidence must be zero across the three
pool boundaries after clustering.

The frozen embedding is a leakage canary, not a semantic-quality model. Its
threshold is recorded and must not be tuned after observing hidden recipe
metrics.

## Resume and determinism

- Hashing 11.26GB is resumable through `inventory_cache.jsonl`.
- A cache row is reusable only when relative path, bytes and `mtime_ns` match.
- The final manifests are sorted and rewritten atomically from validated cache
  records; the source dataset is never modified.
- Split seed, duplicate thresholds, projection parameters, source tree hash,
  every manifest SHA-256 and software commit are recorded.
- A second run with an unchanged source tree must produce byte-identical final
  manifests and report, except for explicitly excluded wall-clock fields (v1
  contains none).

## Hard gates

The evidence freeze passes only when:

- all four domain counts are `4657/628`;
- basename parity is exact and train/test basename overlap is zero;
- total file count and bytes match the local archive contract;
- every file has a SHA-256;
- source/target/dev cluster intersections are empty;
- exact, dHash and frozen-embedding cross-pool leakage counts are zero;
- training loader accepts only source/target manifests and rejects both lockbox
  roles;
- final 628 rows have `payload_access=final_evaluator_only` and no decoded
  metadata/embedding fields.

An embedding or perceptual canary failure blocks the split; it is not repaired
by deleting individual inconvenient pairs after evaluation. Change thresholds
only by creating a new manifest generation and recording why.

## Release boundary

Kaggle metadata labels the archive MIT and the paper describes license-free
samples, but there is no per-image rights manifest. Internal research may
proceed. Public weights, example images, redistributed manifests containing
sensitive lineage, or commercial claims require a separate release review.
