# SF2.2R institutional open-metadata source reconnaissance

Date: 2026-07-17

Node: `ULT > RF0.4 > SF2.2R`

Status: complete; no formal source-audit child opens

## Question

Does an authoritative institutional open-metadata corpus expose enough exact
named photographic-stock evidence and connected creator/collection groups to
justify freezing a new metadata-only stock feasibility audit?

This is bounded source reconnaissance, not a preregistered stock experiment.
It cannot establish that a term is absent from a full corpus, that a pictured
object was captured on the named stock, or that any pixel is reusable.

## Authoritative source facts

- Smithsonian Open Access publishes staff-created collection metadata through
  an official repository and an anonymous AWS Open Data mirror.
- The mirror is line-delimited JSON, hash-sharded into 256 files per owning
  unit. Metadata and media rights are separate fields; CC0 metadata does not
  automatically make every linked image reusable.
- The official S3 listings report the following complete metadata sizes for
  the seven photography-relevant units considered here:

| Unit | Bytes | Role considered |
|---|---:|---|
| NASM | 11,399,479 | aviation and space photographic collections |
| HSFA | 31,703,910 | human-studies film and slide collections |
| NPG | 75,278,034 | photographic portrait objects |
| EEPA | 236,396,975 | photographic archive |
| AAA | 1,742,064,352 | artists' archival collections |
| SIA | 2,043,947,846 | institutional archival collections |
| NMAH | 2,501,226,483 | photographic-history objects and archives |

The combined full metadata scope would be 6,642,017,079 bytes. No image URL
was requested and no response body was retained in the repository.

Primary source checkpoints:

- https://www.si.edu/openaccess/devtools
- https://www.si.edu/openaccess/faq
- https://github.com/Smithsonian/OpenAccess
- https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/index.txt

## Bounded reconnaissance

The same eight deterministic hash shards (`00`, `20`, `40`, `60`, `80`,
`a0`, `c0`, `e0`) were inspected in memory for each unit. Case-insensitive
search terms were `Kodachrome`, `Ektachrome`, `Fujichrome`, `Velvia`,
`Portra 400`, `Portra 800`, `Tri-X`, and `Agfachrome`.

This 8/256 shard sample is a density and schema probe only. It was selected
before reading matches, but it is not a complete census and has no power to
prove a corpus-wide zero.

| Unit | Relevant sampled evidence | DoR interpretation |
|---|---|---|
| NMAH | 71 `Kodachrome` occurrences; no second searched stock family | family-level, single-unit concentration; no comparative edge |
| EEPA | 3 `Kodachrome` occurrences; no second searched stock family | sparse family-level support only |
| SIA | 3 `Ektachrome` occurrences; no second searched stock family | sparse family-level support only |
| NASM | no searched stock occurrence in the fixed sample | no density evidence for a bounded audit |
| HSFA | no exact searched stock; a loose `Portra` probe produced text noise | no exact variant support |
| NPG | no exact searched stock; loose `Portra` is dominated by `portrait` | lexical false-positive warning |
| AAA | no searched stock occurrence in the fixed sample | no density evidence for a bounded audit |

Public collection pages confirm that Smithsonian metadata can describe real
Kodachrome or Fujichrome objects, but examples are commonly family-level
media descriptions, collection-level summaries, or rights-restricted images.
They do not repair exact product identity or create same-source/different-stock
connectivity.

## Decision

`no_formal_audit_dor`.

Do not download or scan the 6.64 GB corpus. The observed evidence is split by
owning unit and mostly names broad historical stock families rather than exact
`film_stock_id` variants. A cross-unit union would turn institution/unit,
collection, era, scanner and content into near-perfect stock proxies. That is
the same shortcut class already rejected by SF1.0B, not a new connected design.

This decision does **not** claim that Smithsonian contains no additional stock
records. It says the bounded, pre-contract evidence does not justify the cost
or scientific claim of a formal full-corpus audit.

## Branch and claim boundary

- no formal `SF2.2A` metadata acquisition opens;
- no image, IIIF, thumbnail or media request opens;
- no stock evidence grade changes;
- no operator fitting, training, clustering, LSM or authenticity claim opens;
- NMAH/EEPA/SIA examples remain source-design evidence only;
- a future child requires a materially new index or structured field that
  exposes exact product variants and connected independent groups before any
  large metadata transfer is frozen.

Ultimate remains active. The next legal leaf is another independently
evidence-gated source design or a deterministic product leaf; failure of this
source does not weaken existing gates or reopen any frozen experiment.
