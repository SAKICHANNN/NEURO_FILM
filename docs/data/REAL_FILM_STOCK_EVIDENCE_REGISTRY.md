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

Within-stock latent modes are recorded only in the separate
`docs/data/LATENT_STOCK_MODE_EVIDENCE_REGISTRY.md`. `H-LSM-1` is a conditional
hypothesis: no current stock has proved `K>1`, and latent inference must never
be copied back into this observed-evidence registry.

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

This rule includes exposure offset, metered EI, box-speed deviation, push/pull,
illuminant spectrum, process session, filtration, reciprocity state and scanner
profile. User text may be retained as an auditable weak hint, but never becomes
structured physical truth without independent evidence.

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
| Wikimedia Commons stock categories | claimed physical film per file/category | exact Ektar100 and UltraMax400 pass pixels; Kodachrome64 stops | SF2.3 metadata union has 294 strict rows/30 author strings, but only Ektar100 and UltraMax400 pass support and share one author; zero redundant edges | `S0`; two provisional unpaired candidates plus connectivity-negative evidence | community labels, insufficient shared-author redundancy and content/scanner nuisance block learning and `S1` |
| YFCC exact user text + live Flickr rights | claimed physical film per user metadata | YFCC15M pixels remain shortcut-negative; full YFCC adds Ektar100/Velvia50 shared-author connectivity | 37 clean SF1.3A pixels; SF1.3B RGB 56.25%/p=.464 versus best nuisance 68.75% | `S0`; connectivity/rights evidence and negative identifiability evidence | UID is not person identity; shared-author pool is closed for learning |
| FILM-R v2 | yes | 11 filename families | 44 source pairs, one contributor; roll/process/scanner unknown | `S0` hints plus damage/artifact evidence | family/content confounding; no authoritative label |
| LOC FSA/OWI | yes | no reliable per-image stock | creator/location/sequence leakage guards | `H historical/unknown`; Phase C nuisance/stress lane | cannot establish physical roll, stock, process or scanner setting |
| Local Flickr-derived set | unverified per row | directory/caption claims only | legacy audit found no durable grouping | quarantined | rights, lineage and grouping fail |
| Apollo flight-film archive | yes | NASA/JSC documents Apollo 7 SO-368/SO-121 magazine, filter and frame ranges | 63/63 pages validate across two SO-368 and five SO-121 magazines | `S0` metadata negative control; SF2.0A content-confounded close | only spacecraft/hardware passes the frozen shared-content rule; no page/pixel expansion or learning |
| NASA/JSC astronaut photography | yes | official exact media codes `VELVI`, `5775`, `5776`; 13,255/397/122 rows across 25 missions, zero ID overlap | only STS098 joins both primaries; Velvia 2 supported rolls versus Portra 400NC 10 | `S0` metadata negative control; SF2.0C0 no-candidate close | no other mission repairs the frozen four-roll edge; no photo acquisition or learning |
| Openverse exact-stock search | no; weak indexed title/tag only | 960 returned rows for Ektar100/Velvia50/Portra400/UltraMax400 | source plus creator URL; Ektar has 19 repeated identities | metadata negative control; SF2.1A contract-mismatch close | unstable relevance pagination and creator dominance; indexed licence is not live upstream rights |
| Smithsonian Open Access metadata | yes for some described objects/collections; pixels not inspected | sampled records use broad Kodachrome/Ektachrome family descriptions, not connected exact product variants | owning unit/collection/record; sampled family evidence is split across NMAH/EEPA/SIA | metadata source-design negative control; SF2.2R no-DoR close | 6.64GB full metadata union would preserve unit/collection/content shortcuts; media rights remain separate |
| Newgrain public application | controlled community stock catalogue; physical use remains user-claimed | public catalogue exposes 385 approved entries and high apparent support for several exact stocks | pseudonymous user plus optional lab/scanner/process fields are technically exposed | source-design negative control; SF2.4R terms-blocked no-DoR close | Terms prohibit automated queries/scraping/mining and user content remains uploader-owned; no retained metadata or pixel acquisition |
| PROV VPRS 17684/17690 | institutional master negatives plus a linked register described as tracking film stock | series-level description only; two-pass SF2.5R finds all 30 register volumes physical-only and no per-negative stock rows online | agency/date/negative-number ranges exist; photographer/process/scanner detail remains unknown | source-design/data-gap evidence; no learning | official API metadata is non-commercial CC-BY-NC; image rights/use require separate audit; no pixels requested |
| ColorReference Velvia 100F Set 3 | exact stock and target-set statement | one physical target set; five slides observed by four scanner/software pipelines | strong same-slide scanner/software connectivity, no independent rolls | internal scanner-nuisance control only; SF2.7A raw median .06055, bounded global residual .01446 | pair medians span .01464-.09865 in unknown device RGB; raw clusters are nuisance, free testing/development statement has no standard redistribution licence, no stock fitting/training |
| ColorReference Velvia 100F recorder-source six-set lane | exact material header on all 60 measurement members | six target-set/charge IDs x five slides; all share `PROD_DATE 2005:05`, physical sheet/roll/process unknown | five common film-recorder RGB grids and 30 measured slide identities, 288 patch IDs each | AQ0 internal paired-proxy source/semantics pass; AQ1 identifiability only | source is unknown-profile recorder RGB, not camera/sRGB; target is measured slide not scanner RGB; no formal open licence, independent-roll claim, fitting/training or calibration |
| NTNU controlled E100/Velvia 50 study | exact stocks in publication and thesis | two paintings and controlled condition cells; roll/process session unknown | matching hyperspectral/film-MSI acquisition described, but no public raw data | publication/method evidence only; SF2.8R data-unavailable close | six analysed frames omit same-illumination cross-stock comparison; stock and illuminant remain confounded; no dataset licence/fitting/training |
| ColorReference multi-family IT8 references | four exact material headers; Ektachrome family only for one | five batch-average manufactured scanner targets, one per family | 288 common patch IDs; direct Lab/density and 41-point spectra; no common uncalibrated recorder input | internal physical spectral / target-manufacturing nuisance evidence only; SF2.9R limited pass | calibrated-target manufacture, batch/age/process and aim adjustment remain confounded; no camera pair, independent rolls, fitting, training, LSM or redistribution |
| Color Precision public comparison metadata | URL-derived stock-looking labels only; physical truth unverified | 927 embedded image URLs / 20 normalized labels / 471 filename condition keys | 456 filename-implied Frontier/Noritsu pairs, but roll/process/profile unknown and 15 keys incomplete | metadata/source-design evidence only; SF2.10R rights-and-replication blocked | scanner auto-adjustment and minor exposure edits disclosed; no explicit comparison-pixel reuse grant, verified pairing, digital counterpart manifest or independent replication; zero image requests |
| SillyStill | claimed yes | one claimed CineStill stock; paper says 41 raw / 38 processed pairs | SF2.6R finds one illustrative pair in the pinned official repo | blocked candidate; no fitting | dataset links remain placeholders and no repository-level data licence exists |
| Emulating Emulsion | yes | controlled Velvia 100, one roll; 33 chart pairs / 3,168 unique patches described | publication-level capture/scan design only | strong method precedent; no fitting | SF2.6R finds no public measurements, fitted parameters, source code or reusable data licence |
| Manufacturer sensitometry | physical prior | authoritative stock/process documents | no scene-level RGB targets | operator prior/constraint | not an end-to-end image target |
| spektrafilm pinned external simulator | no observed film pixels in this project | data-sheet/paper-derived spectral profiles with heuristic coupler parameters | no roll/source/process observations; display-sRGB proxy controls only | RF2.C0 external Look Approximation comparison; Ektar100/fixed-e0 retained | not evidence for stock identity, response, calibration, fitting, training or latent modes |

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
11. before any latent-mode study, a separate connectivity and stock-
    identifiability pass under the LSM1 contract.

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
