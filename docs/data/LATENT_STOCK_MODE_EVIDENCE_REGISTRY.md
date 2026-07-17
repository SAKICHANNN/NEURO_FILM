# Latent stock mode evidence registry

Date: 2026-07-16

Node: `ULT > RF stock-first > LSM0/LSM1`

Status: schema and feasibility ledger only; no latent stock mode is currently
established and no current row authorises training, clustering or operator
fitting.

## 1. Separation invariant

Observed evidence and latent inference are different accounts. A latent result
must never overwrite, enrich or backfill observed source metadata. In
particular, visual appearance cannot turn an unknown exposure, EI, illuminant,
push/pull, process, scanner or emulsion generation into an observed fact.

The observed stock authority remains
`docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`. This registry adds only the
conditional feasibility and inference ledger required by
`docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md`.

## 2. Observed evidence schema

Each source row records only independently verifiable fields:

| Field | Meaning |
|---|---|
| `film_stock_id` | evidence-graded canonical stock identifier |
| `label_evidence_grade` | `S0/S1/S2/S3` under the stock registry |
| `source_id` / `source_revision` | immutable source identity |
| `uploader_group` / `source_group` | observed or explicitly unknown grouping |
| `physical_roll_group` | observed roll identity or `unknown` |
| `rights_snapshot` / `allowed_use` | live or frozen rights evidence and scope |
| `pixel_availability` | none, bounded, quarantined or eligible under a named contract |
| `duplicate_sibling_groups` | exact/perceptual/sibling lineage state |
| `content_cells` | observed/audited content support, not mode labels |
| `process_type` / `development_batch` | observed value or `unknown` |
| `scanner_settings_profile` | observed value or `unknown` |
| `connectivity_edges` | same-source/different-stock and same-stock/different-source support |
| `stock_identifiability` | passed, failed, metadata-only or unidentified |
| `operator_fitting_allowed` | explicit frozen boolean; false by default |
| `evidence_artifacts` | exact manifests, configs, reports and hashes |

## 3. Current observed feasibility ledger

| Candidate/pool | Observed state | LSM1 decision | Operator fitting |
|---|---|---|---:|
| Commons Ektar100/UltraMax400 plus YFCC Ektar100/Velvia50 connected pool | 104 bounded pixels; stock/source/content graph audited; source geometry and low-frequency scene colour dominate | `unidentified`; source/content confounded | false |
| SF1.1-SF1.3B full YFCC shared-author lane | source/repeat scan, bilateral rights and 37 clean pixels pass; held-out-UID RGB 56.25%/p=.464 loses to 68.75% nuisance controls | `unidentified`; connectivity survives but stock signal does not | false |
| BlueNeg Gold archive display-proxy pairs | 47 exact pairs/six rolls; archive preview-to-display mapping only; process/scanner interpretation unknown | not an eligible latent-mode teacher; RF2.S0 is frozen negative transplant evidence | false |
| BlueNeg physical-roll pilot | correct-roll advantage fails replication and loses to content-similar wrong-roll retrieval | `unidentified` for reusable roll information | false |
| FILM-R | physical scans but family/content structural confounding; roll/process/scanner unknown | `source/content confounded` | false |
| LOC FSA/OWI | historical physical film; stock, roll, process and scanner settings unknown | historical/unknown nuisance lane, not named-stock mode study | false |
| Apollo 7 SF2.0A | 63/63 authoritative SO-368/SO-121 pages validate; stock/magazine/filter support passes but only one of two required shared content tags survives | `source/content confounded`; closed before pixels or identifiability | false |
| NASA/JSC STS098 SF2.0B0 | exact VELVI/5775/5776 public-table snapshot contract frozen; no formal snapshot result yet | `metadata only`; later content/date/focal/roll controls required before identifiability | false |
| 53/55/56 deterministic anchors | same-input normalisation indicates near-collinear direction; not real-stock observations | required strength-path negative control; not modes | false |

Current trusted metadata does not provide actual exposure offset, metered EI,
box-speed deviation, push/pull, process session, illuminant spectrum, scanner
profile, filtration or reciprocity state. Occasional user text is a weak hint to
audit, never a physical label.

## 4. LSM1 support/connectivity matrix schema

Before any stock can receive `mode-study eligible`, record:

- stock grade and label evidence source;
- total rows, bounded pixels and independent groups;
- source/uploader distribution and maximum group share;
- effective group count;
- content cells and stock-by-content/source structural zeros;
- same-source/different-stock edges;
- same-stock/different-source edges;
- known and unknown process/scanner fields;
- duplicate/sibling and cross-split leakage state;
- allowed-use scope and live-rights status;
- stock-identifiability evidence against colour/content/geometry/source
  nuisance;
- exact config, manifest, report, software and hash identities.

Allowed decisions are exactly:

- `mode-study eligible`;
- `insufficient connectivity`;
- `source/content confounded`;
- `rights blocked`;
- `metadata only`;
- `unidentified`.

An eligible decision requires all stock, connectivity, identifiability and
pixel/rights gates. It does not establish `K>1`; it opens only the preregistered
LSM2 residual-appearance pilot.

## 5. Latent inference schema

Latent rows exist only after the applicable LSM stage passes. The initial
registry is empty.

| Field | Rule |
|---|---|
| `stock_id` | must reference an eligible observed stock row |
| `mode_id` | `<stock>/Mode A/B/C`; never a physical cause without evidence |
| `status` | candidate, unidentified, descriptive-only, product-eligible or rejected |
| `support_groups` / `effective_group_count` | held-out independent support |
| `feature_signature_version` | frozen mode-space definition and hash |
| `development_transform_hashes` | normalisation/PCA/whitening fitted only on development groups |
| `bootstrap_stability` | membership/medoid/operator-direction intervals |
| `source_content_association` | explicit nuisance association diagnostics |
| `basic_strength_residual` | evidence after EV/WB/contrast/saturation/luma/strength controls |
| `medoid_or_operator_signature` | bounded explicit artifact, not final RGB generator |
| `ood_support` | confidence and fallback evidence |
| `visual_description` | nonphysical descriptive label, if useful |
| `physical_interpretation` | `unknown` by default; otherwise `hypothesis_only` |
| `interpretation_evidence` | independent metadata source, never appearance alone |
| `oracle_value` | gain over the stock global champion, evaluated separately from mode existence |
| `severe_artifact_state` | fail-closed full-resolution state |
| `claim_ceiling` | never above `film-inspired/unpaired-evidence` without a separate calibrated lane |

## 6. Initial latent registry

No rows. The project has not proved two or more transferable modes for any
stock. `K=1` remains the formal default until a complete LSM4 confirmatory
evidence bundle says otherwise.

## 7. Negative-control registry

| Control | Expected interpretation | Failure signal |
|---|---|---|
| 53/55/56 same-direction anchors | one mode plus a strength path | method splits them into stable separate modes |
| shuffled labels/random partitions | no stable cross-group mode | comparable stability to proposed modes |
| source/uploader-only clusters | nuisance | promoted as stock/emulsion modes |
| content/scene-colour/HOG clusters | content shortcut | promoted as mode evidence |
| scanner/geometry/border/resolution clusters | output/source nuisance | physical interpretation assigned |
| strength-only splits | continuous control axis | multiple experts created |

ID 11's 09/53/55/56 red-speckle/posterisation evidence is a required
worst-case artifact regression. It is external autonomous visual evidence, not
a completed internal blind adjudication or population preference.

## 8. Promotion and rollback

Latent inference may be appended only through the stage-specific development
and confirmatory contracts. An LSM result never mutates the observed ledger.
If support, nuisance control, stability, Oracle value or severe-artifact gates
later fail, mark the latent row rejected/descriptive-only and fall back to the
stock global champion or safe-rich; preserve all configs, reports and prior
decisions.
