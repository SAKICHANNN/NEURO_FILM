# Real-film stock evidence registry

Date: 2026-07-15

Node: `ULT > RF0.4`

Status: active stock-first data authority

## 1. Purpose

Ultimate is a **stock-first real-film** programme. Its primary unit is a
specific, evidence-backed `film_stock_id`, not a generic film domain, physical
roll, photographer, archive, Capture One recipe or camera Film Simulation.

Historical or otherwise unknown-stock film is a valid independent class:

```text
historical-film / unknown-stock
```

It supplies archive-look, ageing, scanner-nuisance and artifact evidence, but
it never substitutes for named-stock coverage and is never counted as a
specific stock.

This registry is an evidence ledger, not a claim that every listed candidate is
ready to train. Pixel acquisition and fitting remain behind source-specific
contracts and frozen gates.

## 2. Canonical hierarchy

```text
film_stock_id
  -> physical_roll_id
      -> frame_id
      -> scene/content
  -> process_type
  -> lab/development_batch
  -> scanner_id/settings/profile
  -> photographer/source/uploader
  -> capture_date/era
  -> ageing/damage/restoration state
```

`film_stock_id` chooses the product/research expert. Roll, process, scanner,
source and content are nested repeated-measure or nuisance variables. Missing
fields remain explicit `unknown`; they are never inferred from appearance and
then reused as truth.

At inference the user chooses the desired stock. A router may select only among
bounded experts or cases inside that stock. It must not guess that the input
digital photograph already belongs to a stock.

## 3. Label evidence grades

| Grade | Minimum evidence | Allowed use | Forbidden claim |
|---|---|---|---|
| `S0 claimed` | physical-film source is credible; stock is filename, caption or other weak hint | audit, search prioritisation, stress analysis | stock expert or authenticity |
| `S1 research-eligible` | stock label is supported by dataset/author/archive/edge-code evidence and content/source controls are viable | provisional stock-specific real-film-derived expert | transferable or calibrated stock response |
| `S2 transferable` | multiple independent rolls or sources; unseen-roll/source gain survives pooled, wrong-stock, retrieval and nuisance controls | `real-film-derived/<stock>` | `calibrated-reference` |
| `S3 calibrated` | controlled digital/film measurements, multiple rolls/process-scan sessions and whole-roll holdout | `calibrated-reference/<stock>` within the measured contract | universal response outside the measured contract |
| `H historical/unknown` | real physical-film provenance, but authoritative stock is absent | historical/unknown-stock expert and stress data | any named-stock coverage or fidelity claim |

Named-stock coverage and historical/unknown coverage are reported separately.
The first research milestone is at least three distinguishable `S2` stock
experts plus one independent `H` expert; it is a milestone, not a stop rule.

## 4. Current evidence ledger

| Source | Physical film | Stock evidence | Roll/source structure | Current grade/use | Blocking fact |
|---|---:|---|---|---|---|
| FilmSet Cinema/ClassNeg/Velvia | no | Capture One recipe labels | exact digital identities | auxiliary digital-recipe control only | not physical film |
| BlueNeg bounded core | yes | dataset-declared `Kodak Gold 100-5` | four matched physical rolls | provisional single-stock mechanism evidence; below `S2` | roll signal loses to content retrieval; restoration/scanner nuisance |
| BlueNeg full metadata | yes | 13 dataset film strings | 53 rolls, but most lack matched operator controls | acquisition/search candidate only | current matched evidence does not support multi-stock transfer |
| FILM-R v2 | yes | 11 filename families | 44 source pairs, one contributor; roll/process/scanner unknown | `S0` hints plus damage/artifact evidence | family/content confounding; no authoritative label |
| LOC FSA/OWI | yes | no reliable per-image stock | creator/location/sequence leakage guards | `H historical/unknown`; Phase C nuisance/stress lane | cannot establish physical roll, stock, process or scanner setting |
| Local Flickr-derived set | unverified per row | directory/caption claims only | legacy audit found no durable grouping | quarantined | rights, lineage and grouping fail |
| Apollo flight-film archive | yes | documented mission/magazine film types require local authoritative mapping | magazine/frame structure; narrow lunar/space domain | `S0` candidate pending RF0.4 audit | content domain and product/derivative rights need separation |
| SillyStill | claimed yes | one claimed Cinestill stock | small paired capture described by paper | blocked candidate | full data unavailable; repository lacks a root data license |
| Emulating Emulsion | yes | controlled Velvia 100, one roll | chart pairs under controlled capture/scan | strong method precedent; candidate only | public data and reusable licence not verified |
| Manufacturer sensitometry | physical prior | authoritative stock/process documents | no scene-level RGB targets | operator prior/constraint | not an end-to-end image target |

## 5. Stock eligibility contract

Before any stock-specific fit, freeze:

1. exact `film_stock_id` and its evidence source;
2. source URL/revision, licence snapshot, credit and allowed use;
3. physical-film verification and scan interpretation;
4. roll/source/process/scanner fields and explicit unknowns;
5. content cells and stock-by-content support matrix;
6. exact/perceptual duplicate and sibling grouping;
7. whole-roll or strongest available independent-source holdout;
8. pooled-all-stock, wrong-stock, generic historical, retrieval and simple
   WB/contrast/saturation controls;
9. stock-specific claim ceiling and fail branch;
10. manifest/config/report hashes and software identity.

A filename-only family, one source with stock/content structural zeros, or an
unknown-stock archive cannot pass this contract by using a larger model.

## 6. Coverage and evaluation contract

Every report must show two independent ledgers:

- `named_stock_coverage`: count and list by `S0/S1/S2/S3`;
- `historical_unknown_coverage`: count of independent real-film archives and
  sources, never converted into a stock count.

Each stock is evaluated separately with, where supported:

- leave-one-roll-out;
- leave-one-source/photographer-out;
- leave-one-process/scanner-out;
- pooled-all-stock and wrong-stock experts;
- generic historical-film expert;
- source/content retrieval control;
- matched WB/contrast/saturation and matched style strength;
- shuffled stock labels;
- full-resolution severe-artifact veto;
- pairwise stock distinctiveness after removing overall saturation, contrast
  and mean cast.

If outputs from different stocks collapse to one average look, or if content,
source or scanner controls explain the apparent stock signal, the stock branch
fails regardless of model capacity or visual saturation.

## 7. Immediate gate

`RF0.4` has selected four bounded BlueNeg audit pilots in the machine-readable
registry: Kodak Gold 100-5, Kodak GA 100 5095, Fuji NPH400 and Konica Super XG
100. Kodak Gold 400-5 was removed when whole-test-roll sealing left only one
eligible roll/frame. Selection authorises metadata/acquisition gating, not an
expert claim.
The comparison, rationale, experiment DAG and failure branches live in
`docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md`.

No stock is currently promoted to `S2`; only Kodak Gold 100-5 has locally
acquired `S1` pixels. LOC is the independent sealed `H` lane and can never fill
the missing named-stock slots.
