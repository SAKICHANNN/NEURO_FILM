# Ultimate real-film mainline reopening report

Date: 2026-07-15

Node: `ULT > RF`

Status: active planning authority for Ultimate evidence

> **Stock-first correction, 2026-07-15:** specific, evidence-backed
> `film_stock_id` classes now own the main learning and coverage objective.
> Historical/unknown-stock film remains a valid independent class, but it
> cannot replace or count toward named-stock coverage. The active evidence
> schema and current ledger live in
> `docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`; the executable source and
> experiment plan is `docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md`.

## 1. Corrected success condition

Ultimate succeeds only when a bounded explicit colour system learns distinct
experts for multiple specific, evidence-backed photographic-film stocks from
verifiable physical-film scans, remains visibly stylised on unseen real rolls
or independent sources, beats matched-strength pooled, wrong-stock, generic
historical and nuisance controls, and passes original-resolution
severe-artifact review.

FilmSet, camera Film Simulations, Capture One recipes, hand LUTs and
pseudo-teachers are controls or pretraining material. They are never final
film truth. Unknown process/scanner data can support only
`real-film-derived/unknown-look`; grouped archive scans can support only an
archive/scanner-specific `roll-look`. `calibrated-reference` requires controlled
stock, process, scanner and whole-roll holdout evidence.

Historical or unknown-stock film is retained as the separate
`historical-film/unknown-stock` class. It may support a generic archive-look
expert, ageing/scanner analysis and stress evaluation, but it never substitutes
for a specific stock and is reported in a separate coverage ledger.

Within an evidence-backed stock, `H-LSM-1` allows but does not presume two or
more stable latent appearance/operator modes. No current stock has proved
`K>1`, and no current unpaired method has identified a true digital-to-film
operator. `K=1` is a formal branch that retains the stock global champion plus
bounded strength. The complete conditional gates and observed/latent ledger
separation live in `docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md` and
`docs/data/LATENT_STOCK_MODE_EVIDENCE_REGISTRY.md`.

The first stock-first research milestone is at least three distinguishable,
transferable `real-film-derived/<stock>` experts plus one independent
historical/unknown expert. It is a milestone, not a stopping condition; the
programme continues to add as many scientifically eligible stocks as practical.

### 1.1 Primary class and nested nuisance contract

The highest-level training/product class is `film_stock_id`: manufacturer,
product line, nominal ISO, negative/reversal/motion/B&W type and, when
available, emulsion generation or catalogue code. `physical_roll_id`, process,
lab/development batch, scanner/settings/profile, photographer/source, content,
era, ageing and restoration are nested repeated-measure or nuisance variables.

At inference the user selects a target stock. Content-aware routing may choose
only among bounded cases or experts inside that stock. It does not classify the
input digital photograph as if the input already had a film identity.

Filename, folder, Capture One, Film Simulation, LUT or visually inferred labels
never become authoritative stock truth without independent provenance.
Exposure, metered EI, illuminant, push/pull, process and scanner interpretations
also remain `unknown` unless independently observed. A latent cluster may be
called only `Mode A/B/C` or given a nonphysical visual description; it cannot
backfill those fields.

## 2. Evidence ledger

| Evidence | Physical film? | Group/control quality | Current valid conclusion |
|---|---:|---|---|
| FilmSet train/internal/final | no; Capture One digital recipes | exact digital pairs and frozen identities | explicit operators can reproduce some strong digital recipes safely; auxiliary method control only |
| CT8 final-628 | no | 628 untouched digital identities | Cinema Lab and ClassNeg pooled-L2 beat the fixed basic; Velvia pooled-L2 loses fidelity despite stronger style; no real-film conclusion |
| BlueNeg | yes, archive negatives/scans | 491 frames, 53 physical rolls, but usable matched pilot reduced to four Kodak Gold 100-5 rolls and restoration/scanner nuisance | archive/scanner-specific roll information is ambiguous; not digital-to-film or stock truth |
| local Flickr-derived files | claimed film, not sufficiently verified | 4,210-row audit has zero eligible rows and no reliable roll/scanner lineage | quarantined; no training or truth use |
| manufacturer sensitometry | physical prior, not image target | stock/process curves and partial spectral information | initialise/constrain a model only |
| FILM-R candidate | yes; 44 4K 35mm colour scans | stock-like filenames, one contributor; roll/process/scanner unreported | rights-clear real-film target/style and artifact evidence after local acquisition; not stock truth |
| Apollo candidate | yes; original NASA flight film | physical magazines and documented film types; SF2.0A validates Apollo 7 metadata but closes content-confounded | narrow archive/stress source only; no Apollo 7 page/pixel expansion, while raw scans remain huge and processed products have stricter rights |
| DOCUMERICA candidate | yes; federal documentary slides/negatives | photographer and archival metadata, but no reliable roll/scanner/process fields | diverse public-domain real-film archive look; uploader/era/scan nuisance dominates stock claims |
| SillyStill candidate | yes, paired digital/Cinestill claimed | paper reports 41 raw / 38 processed pairs | SF2.6R pins the current official repo: dataset links remain placeholders, only one illustrative pair is present and no root licence exists |
| Emulating Emulsion candidate | yes, controlled Velvia 100 | one 36-exposure roll, 33 chart pairs / 3,168 unique patch correspondences | SF2.6R confirms strong method precedent but no public measurements, fitted parameters, code or reusable data licence |
| NTNU controlled reversal candidate | yes, controlled Ektachrome E100 and Velvia 50 | two paintings, two illuminants, exposure variants, matching hyperspectral captures and ten-band film scans | SF2.8R finds no released raw cube/TIFF, measurements, code, manifest or dataset licence; the six analysed frames lack a same-illumination cross-stock control |
| ColorReference multi-family IT8 targets | exact material for four archives; Ektachrome family only for one | 288 common manufactured calibration patches, direct Lab/density and 41-point transmission spectra | SF2.9R passes integrity/schema but lacks common uncalibrated recorder input, camera scenes and independent roll/process replication; physical/nuisance evidence only |
| Color Precision comparison metadata | 20 URL-derived stock-looking labels; physical truth unverified | 927 embedded comparison URLs, 471 condition keys and 456 filename-implied Frontier/Noritsu pairs | SF2.10R repeats exactly offline but scanner auto-adjustment is disclosed and rights, verified pairing, digital counterpart manifest and independent roll/process replication fail before pixels |

### CT8 auxiliary closure

The one-shot report is
`outputs/roll2film/ct8_final_628/report.json`, SHA-256
`04f3e254a16316af62f30f380651fdcf134ef8e9d38bbdf9cf11c7eb2d8acefe`.

| Domain | Primary vs fixed basic target Delta-E00 | 95% improvement interval | Style ratio | Decision |
|---|---:|---:|---:|---|
| Cinema | 1.5738 vs 1.7842 | [0.1931, 0.2276] | 1.005 | digital-recipe control pass |
| ClassNeg | 1.8994 vs 4.6588 | [2.7065, 2.8122] | 1.150 | digital-recipe control pass |
| Velvia | 2.9997 vs 2.8182 | [-0.2648, -0.0982] | 1.707 | fail; stronger but less faithful |

All 467 frozen Velvia automatic triggers were rendered at original resolution
and scanned over 10 pages. No severe geometry/text/object corruption, seams,
colour blocks or unstable speckle was confirmed. That safety observation does
not rescue the failed fidelity gate and does not establish real Velvia.
Supplement report SHA-256:
`b0fc5691cb8f6dd198a5c8ef33fd5fcc8afe922dad2f267c9248b3415090beeb`.

## 3. Why BlueNeg is ambiguous and the minimum discriminating experiment

The correct-roll mean improves over the composite control by only `0.0447`
Delta-E00 and its physical-roll bootstrap interval is `[-0.2201, +0.3095]`.
The two held-out rolls reverse sign: `19960817H` loses `0.2201`, while
`19970620C` gains `0.3095`. Four usable rolls cannot separate roll information
from date, location, deterioration, restoration, scanner and scene coverage.

The minimum diagnostic using already available metadata is a preregistered
nested leave-one-frame-out study on all four non-test matched rolls:

1. use every eligible query once with a fixed three-frame support budget;
2. compare correct roll, pooled, every same-stock wrong roll, shuffled support,
   scene/content retrieval and scanner/restoration-only features;
3. match style strength and common colour support;
4. report roll-wise effects, roll x query variance and roll-cluster intervals;
5. use no additional model capacity.

This can determine whether the sign reversal is mainly support sampling or a
repeatable roll interaction. It cannot establish population-level stock signal
with four rolls. A confirmatory claim requires several additional independent
same-stock rolls or a one-shot whole-roll evaluation on separately frozen
BlueNeg rolls under a new access contract. The latter remains an
archive/restoration mechanism test, not fresh-film emulation truth.

## 4. Obtainable real-film candidates

| Candidate | Exact/advertised scale | Rights state | Metadata strengths | Cost and decision |
|---|---:|---|---|---|
| FILM-R, Figshare 21803304 v2 | 88 files; 437,570,872 bytes; 44 damaged + 44 restored real scans | CC BY 4.0 | filenames identify 11 stock/format families; one contributor | acquired; family/content gate failed, so stress only |
| LOC FSA/OWI colour archive | 558 canonical public scans; 258 bounded derivatives currently retained | US federal public domain | creator/location/sequence guards; no reliable per-image stock | sealed partial `H historical/unknown`, never a named-stock pilot |
| FILM-AA, Figshare 21803292 | 20 files; 122,164,960 bytes; 10 empty damaged frames + annotations | CC BY 4.0 | authentic damage, no scene/roll colour truth | optional artifact-control sibling, not colour training |
| BlueNeg exact revision | 491 previews, 53 rolls, 13 film strings; current bounded lane 118,929,719 bytes | custom attribution license already snapshotted | physical roll/date/location/film string, alignment metadata | already acquired bounded subset; expand only under a preregistered whole-roll gate |
| NASA Apollo flight-film scans | almost 25,000 catalog images; raw examples about 1.2 GB each | NASA raw scans public domain with credit; ASU processed products restrict derivatives/commercial use | magazine, mission, film type, frame; scanner provenance | SF2.0A Apollo 7 sample closes content-confounded; do not expand pages or download pixels |
| DOCUMERICA | about 15,981 online public-domain scans | US federal public domain | photographer, place/date/series; Kodachrome/Ektachrome collection-level history | metadata-only audit, then small photographer/source-stratified sample |
| SillyStill | paper: 41 raw pairs, 38 processed | official repo has no root license and full dataset links remain unavailable | paired tripod scenes, one claimed stock | blocked for training; metadata/reference only |
| Emulating Emulsion | 33 chart image pairs / 3,168 unique patches, one Velvia 100 roll | publication accessible; public data/license not found | controlled illuminant, exposure, camera, D50 scan | method precedent only until data becomes verifiably available |
| archival motion collections | 44 to 81,576+ frames depending source | source-specific and often unclear | reel/frame continuity, authentic damage | artifact research only until stock/scanner/license audit passes |

Candidate availability is not stock eligibility. `RF0.4` now audits label
authority, stock-by-content support, independent rolls/sources, process/scanner
metadata, rights and attainable evidence grade before choosing the first 2-4
stock pilots. BlueNeg Kodak Gold remains provisional; no second stock is
currently promoted. LOC FSA/OWI is an `H historical/unknown` auxiliary lane and
cannot fill a named-stock pilot slot.

Primary sources:

- FILM-R project and license: https://figshare.com/articles/dataset/Authentically_damaged_film_scans/21803304
- Film damage project: https://daniela997.github.io/FilmDamageSimulator/
- BlueNeg repository revision: https://huggingface.co/datasets/ttgroup/blueneg-release/tree/b038a1ae68f42067ff12b5e79ddbe62919b7af23
- NASA raw-scan rights and sizes: https://apollo.im-ldi.com/ABOUT_SCANS/index.html
- NASA film/magazine documentation: https://www.nasa.gov/history/astronaut-still-photography-during-apollo/
- DOCUMERICA/NARA: https://www.archives.gov/publications/prologue/2009/spring/documerica.html
- SillyStill official repo: https://github.com/mikasenghaas/sillystill
- Emulating Emulsion: https://musicofmusix.github.io/siggraphposters25

## 5. Candidate algorithm matrix

| Family | CPU role | GPU challenger | What makes it more than saturation/contrast | Fail condition |
|---|---|---|---|---|
| global explicit expert | monotone density curves + matrix + smooth 3D LUT; Lab/Bures/OT/quantile/L2 baselines | optional differentiable LUT optimiser | hue-dependent and luminance-conditioned colour interaction with neutral-axis/Jacobian constraints | loses to matched WB/contrast/saturation or collapses to one mean cast |
| hierarchical mixed effects | fit shared stock/source component plus roll/process/scanner random effects with shrinkage | amortised posterior over the same finite parameters | explicitly subtracts nuisance and tests reusable shared signal | shared component unstable under leave-roll/uploader/scanner-out |
| similar-case retrieval | frozen semantic/achromatic features choose hard Top-1 explicit medoid/expert | small contrastive reranker or set encoder, never RGB decoder | input-dependent expert choice based on matched scene/colour support | content-only retrieval equals correct stock/roll or routing memorises uploader |
| conditional LUT/curves | kernel/ridge prediction of bounded coefficients | compact hypernetwork predicts curves/LUT weights | allows scene-dependent highlight, hue and exposure response inside explicit bounds | no gain over hard global experts at matched style strength |
| sparse mixture of experts | deterministic hard router + OOD fallback | Deep Sets/attention predicts sparse expert weights | preserves multiple stable modes instead of dataset averaging | Oracle routing value absent or shuffled groups perform equally |
| bounded local residual | only after global residual is demonstrated; bilateral grid with zero-init and amplitude/TV limits | lightweight grid coefficient predictor | corrects supported local illumination interactions without spatial generation | seams, unsupported colour moves, or no held-out gain over global |

No candidate directly generates final RGB. Every GPU model predicts a bounded
operator, expert selection, LUT weights or grid. The deterministic global
fallback and severe veto remain mandatory.

## 6. Gates and stop conditions

| Gate | Pass evidence | Stop/fallback |
|---|---|---|
| RF-G0 provenance | physical film verified; source/revision/license/hash/credit recorded | quarantine unverifiable or unclear-rights bytes |
| RF-G1 stock label/grouping | authoritative `film_stock_id` plus roll/source/scanner/process fields support leakage-free holdouts | grade `S0` or `H`; no stock expert/claim |
| RF-G2 stock identifiability | correct stock beats pooled, wrong-stock, shuffled, generic historical, nuisance and retrieval controls with group-aware CI above zero | close or downgrade that stock; do not scale model |
| RF-G3 style | visible style exceeds safe bland baseline and matched simple enhancement | reject saturation/contrast-only gain |
| RF-G4 transfer | unseen real rolls/sources retain the stock-specific gain after style matching | source-specific or `S1` label; no `S2` promotion |
| RF-G5 artifacts | no confirmed severe issue on frozen full-resolution gold; stress rate reported with CI | deterministic fallback or candidate rejection |
| RF-G6 content | face/text/object/geometry retained; intended grain/halation judged separately | reject spatial residual/model |
| RF-G7 claims/coverage | label matches `S0/S1/S2/S3/H`; named-stock and historical coverage remain separate | fail closed to lower claim; never count `H` as a stock |
| RF-G8 reproducibility | configs, seeds, hashes, split groups, software and evidence reports frozen | no promotion |

## 7. Dependency-ordered execution plan

```text
RF0 truth/data gate
  RF0.1 acquire and hash FILM-R under CC BY 4.0
  RF0.2 build source/rights/group manifest; unknown fields remain explicit
  RF0.3 LOC FSA/OWI historical/unknown metadata, pixels and nuisance gate
  RF0.4 authoritative stock registry and obtainable named-stock audit
    -> RF1 signal audits
       RF1.1 FILM-R family/content audit (stopped: unidentified)
       RF1.2 BlueNeg roll diagnostic (closed: content retrieval wins)
       RF1.3 LOC creator/location holdouts for the `H` auxiliary lane
       RF1.4 per-stock label/content/nuisance identifiability for first 2-4 pilots
         -> LSM0/LSM1 conditional ontology and feasibility
            -> LSM2-LSM8 only after stock, connectivity, identifiability,
               pixel/rights and leakage gates all pass
         -> RF2 CPU explicit expert parent
            RF2.H independent historical/unknown expert
            RF2.S stock-specific global -> hierarchical -> retrieval -> conditional
              -> RF3 GPU bounded challengers only after a stock-specific RF2 residual
                 -> RF4 unseen stock/roll/source style and artifact confirmation
                    -> RF5 stock-selectable product integration with OOD fallback
```

Each leaf freezes a hypothesis, controls, metric, gate and failure branch before
pixel access. FilmSet remains a software regression and pretraining/control
lane; it cannot promote RF nodes.

## 8. Nearest ready leaves

The named-stock data mainline has advanced through SF1.1. The exact full-YFCC
metadata index passes SHA/S3 integrity and two byte-identical scans;
Ektar100/Velvia50 has 16 shared UIDs. `SF1.2` confirms eight authors with a
current CC BY page for each stock without requesting image URLs. `SF1.3A`
retains 37 clean derivatives across all eight bilateral UIDs with no
exact/dHash<=4 duplicates or confirmed severe artifact. The current P0 leaf is
`SF1.3B`, a preregistered leave-one-UID-out stock-identifiability diagnostic.
It fails: global RGB is 56.25%/p=.464 and loses to 68.75% nuisance controls,
so this pool is closed for learning. The local ready branch returns to the
independent deterministic U1 high-precision product path while other stock/data
evidence is researched; training, operator fitting and LSM remain closed.

That stock/data research completed the metadata-only `SF2.0A` leaf. All 63
NASA/JSC Apollo 7 pages pass ID, stock, magazine, exposure-support and
filter-free bridge checks, but only `spacecraft_hardware` satisfies the frozen
cross-stock/two-magazine content rule; two shared tags were required. The
deterministic decision is `content_confounded`. Apollo 7 page/pixel expansion
closes, no image payload was requested, and fitting/training/LSM remain false.

The broader NASA/JSC database was tested as a materially different source, not
an Apollo 7 rescue. `SF2.0B0` obtains 168/329/10 STS098 rows for exact codes
`VELVI`/`5775`/`5776` with zero overlap or query error. Velvia, however, spans
only two rolls versus the frozen four-roll minimum. The edge closes for
insufficient independent support; no B1, photo pages, images, fitting, training
or LSM opens.

`SF2.0C0` then tests the same exact codes across the full result tables without
reopening STS098. The bounded aggregate-only census finds 13,255 Velvia50, 397
Portra400NC and 122 auxiliary Portra400VC rows across 25 missions. Only STS098
contains both primary stocks, and its Velvia arm still has two supported rolls
against the frozen minimum four. The decision is `no_candidate_mission`:
NASA/JSC cross-mission expansion closes with no photo pages or images requested,
and no fitting, training, LSM or stock claim opens.

`SF2.1A` tests Openverse as a separate openly licensed discovery index. The
bounded four-stock search returns 960 metadata rows without requesting images,
but Ektar relevance pagination repeats 19 identities and violates the frozen
stable-census contract. Only UltraMax passes the independent strict-row/creator
gate; Velvia has one strict row and Ektar/Portra exceed the 40% creator-share
ceiling. The decision is `query_contract_mismatch`; no live-page preflight,
pixels, fitting, training, LSM or stock claim opens.

`SF2.2R` then checks whether an authoritative institutional open-metadata source
has a materially better pre-contract design. Smithsonian's official hash-
sharded mirror is machine-readable, but deterministic 8/256-shard probes across
seven relevant units expose broad Kodachrome evidence in NMAH/EEPA and sparse
Ektachrome in SIA rather than exact product variants connected inside common
groups. The complete selected metadata scope is 6,642,017,079 bytes. The
decision is `no_formal_audit_dor`: do not download a disconnected corpus whose
unit/collection/content signatures would stand in for stock. No media, fitting,
training, LSM or stock claim opens.

`SF2.3` closes the frozen Commons union without network access. The audit finds
294 strict rows across 19 exact-stock categories and 30 author strings, but
only Ektar100 and UltraMax400 pass the established per-stock support gate. Their
single shared author is below the frozen two-author edge minimum, yielding zero
retained edges and no component. Do not merge aliases or lower the gate after
the result; no live preflight, pixels, fitting, training or LSM open.

`SF2.4R` then checks a film-specific community source with a controlled stock
catalogue. A minimal anonymous pre-contract probe confirms 385 approved stock
entries and technically useful pseudonymous-user/lab/scanner/process metadata,
but Newgrain's published Terms expressly prohibit automated searches, requests,
queries, scraping and mining. The decision is
`terms_blocked_no_formal_audit_dor`: exposed frontend access is not permission,
so no client, retained snapshot, shared-user graph, pixel request, fitting,
training or LSM opens. A future child requires written platform permission or
an officially licensed research export and separate pixel rights.

The independent `RF2.C0` external spectral-prior control is complete without
changing this data stop. One pinned spektrafilm Ektar100/fixed-e0 chain is
visibly stylised and non-basic while passing the provisional gold severe veto,
so it is retained only as an external comparison control. Auto-exposure chains
show much larger apparent style but unstable roughly 0.2x--4.2x scene-luma
response and near-duplicate profile behaviour; they are nuisance/adaptation
evidence, not stock experts. No external output is a teacher, identified
operator, stock response or calibration target.

The clean-room `U5.R2H0A` datasheet sibling now supplies a stronger negative
identifiability result without changing any stock-data gate. Its Velvia 50
base witness is strongly nonlinear, but D65-colour-matched bounded spectra
produce output differences much larger than the witness effect and its fixed
neutral axis also fails. Display-sRGB therefore does not identify the
stock-layer exposure even when official sensitivity and dye graphs exist. No
visual frontier, fitting, calibration or teacher opens. A measured-natural-
reflectance pilot may study a narrower empirical prior, but can never erase
the theoretical ambiguity or become stock truth.

That measured-reflectance child, `U5.R2H0C1`, now supplies a positive but
strictly subordinate result. Across 263 cross-scene CAVE cell pairs at D65
input Delta E76 <=1, the overlap-only witness differs 1.38 median / 5.47 p95,
only 1.66% / 3.79% of H0A's adversarial spread; scene-bootstrap gates pass.
This supports testing one simple empirical canonicalizer, not physical spectral
recovery, stock truth, film fitting or visual/product promotion. External
measured-spectrum replication remains mandatory for a broader claim.

The fixed H0C2 child also passes narrowly: different-scene hard Top-1
measured-spectrum retrieval at D65 Delta E76 <=1 covers 36.13% of CAVE queries,
wins 76.60%, and lowers selected synthetic-witness error to .339/2.975
median/p95 from smooth .894/3.939. Thresholds 2 and 3 fail group-stable median
reduction, so smooth OOD fallback remains mandatory. This is an empirical
canonicalizer mechanism, not film evidence; an independent spectrum source
must replicate the frozen T=1 policy before any broader branch.

That required `U5.R2H0C3` replication fails decisively on the independent
public-domain USGS library. The fixed CAVE bank selects 184/1,732 USGS AREF
queries at T=1 but wins only 6.52%; selected smooth error is .271/4.331 versus
hard 1.561/5.690 median/p95, every chapter direction favours smooth and the
complete fallback-policy p95 worsens. H0C2 is therefore retained only as
within-source mechanism evidence. Broad cross-source spectral canonicalization
closes without retuning, adding USGS to the bank, Top-K blending or neural
rescue; it never opens RGB spectral recovery, film fitting or stock truth.

The historical sibling remains unchanged: preserve the 258 already downloaded
FSA/OWI derivatives and defer further acquisition while named-stock P0
advances. Its output remains `historical-film/unknown-stock` and never
substitutes for a named-stock leaf.

Do not fit a colour expert on either branch until its own RF1 gate is frozen and
passed. FILM-R acquisition and its family/content stop are already complete.

### RF1.2 update, 2026-07-15

The four-roll 43-query BlueNeg nested LOO diagnosis is complete. Wrong-roll
source-content retrieval beats the correct physical-roll operator on all four
raw roll means; the cluster-equal correct-roll gain is -0.3856 Delta-E with 95%
interval `[-0.4956,-0.2071]`. Physical-roll information remains closed. The
result elevates similar-case retrieval of bounded explicit operators as an RF2
challenger, but only as archive-restoration mechanism evidence until it passes
the real-film RF1.1 gate.

## 9. Governance, scope and closure

### Non-goals

- do not reinterpret FilmSet, camera Film Simulations, LUTs or filename hints
  as physical stock truth;
- do not count historical/unknown archives as named stocks;
- do not reopen Roll2Film or train a larger model merely because its current
  physical-roll hypothesis failed;
- do not claim calibration without controlled paired stock/process/scan data;
- do not modify frozen experiment results, manifests or source bytes as part of
  this planning correction.
- do not cluster raw appearance, content, source or scanner proxies and call
  them stock modes; do not interpret a latent mode physically without
  independent metadata;
- do not use RF2.S0, current community pixels or 53/55/56 as latent-mode teacher
  truth. RF2.S0 and SF1.0B remain frozen negative evidence, while 53/55/56 is a
  required strength-path negative control.

### Definition of done

A stock leaf is done only when its label evidence, rights, grouping, content
support, nuisance controls, split, baselines, metrics, claim ceiling, hashes and
per-stock decision are durable. Ultimate's first stock-first milestone requires
at least three independently validated `S2` experts plus one separately
reported `H` expert; severe-artifact and visible-style gates remain mandatory.

### Change propagation and bottom-up reintegration

Every stock/data decision propagates to the evidence registry, this plan, the
Ultimate tracker, task board, AGENTS current truth and agent log. It must also
check sibling historical, product, evaluator and calibration lanes. Reintegrate
bottom-up: validate the dataset/stock leaf first, then RF1, RF2, RF parent,
Ultimate success/coverage and the user-visible claim label.

### Rollback and recovery

Planning changes are one docs-only commit and can be reverted without changing
data, configs, code or results. Later stock acquisitions use new manifests and
policy IDs; a failed or invalid stock is downgraded/quarantined while previous
experts and historical evidence remain reproducible. Never rewrite a frozen
holdout or relabel old results to preserve a preferred conclusion.

`U5.R2AE0` adds a newer independent external simulator only as a synthetic
algorithm-control source. Pinned MIT `spectral_film_lut` runs headlessly and
contains 86 unique exported material objects, but current profiles have no
per-row document/page/uncertainty identity. The Git history retains 31 removed
datasheet blobs for only a subset and supplies no replacement mapping or
redistribution authority. One de-duplicated, source-bound synthetic structural
bank audit may proceed. It cannot supply real-film pixels, stock truth,
operator fitting, teachers, LSM eligibility or product integration. See
`docs/U5_R2AE0_SPECTRAL_FILM_LUT_SOURCE_AUDIT.md`.

`U5.R2AE1` closes that synthetic child. The eight document-associated chains
are all visibly strong in metric space and non-basic, with exact replay and
three distinct common-output families, but each contains `1.61–9.50%`
negative-Jacobian grid cells against a frozen `.5%` ceiling. This is external
algorithm evidence only; it opens no visual stage, stock evidence, fitting,
teacher bank, LSM or integration. Projection, smoothing, clamping and
post-result profile/threshold search are forbidden by the frozen branch. See
`docs/U5_R2AE1_SPECTRAL_FILM_LUT_STRUCTURAL_BANK_RESULTS.md`.

`U5.R2AF0` separately audits AceTone and rejects its published Qwen/GRPO
selector under the non-generative contract. The released tokenizer is a
compact explicit-LUT representation, but `[0,1]` sigmoid output is not a
topology certificate: its objective has no monotonicity, positive-Jacobian or
invertibility term, and its training LUT corpus is not lineage-released. One
synthetic AF1 stress may test whether the exact checkpoint preserves already
safe analytic LUTs. No photos, benchmark acquisition, training, film evidence
or AE1 repair opens. See `docs/U5_R2AF0_ACETONE_SOURCE_METHOD_AUDIT.md`.

AF1 then rejects that tokenizer before photographs. All controls and exact
repeat pass, but every decoded safe analytic LUT has `7.54–10.60%` negative
Jacobian cells and fails RGB/Delta-E fidelity; identity itself folds at
`10.60%`. The sigmoid range bound and zero interior clipping do not rescue
orientation. No decoder, checkpoint, projection, Qwen, visual, training or
product branch opens. See
`docs/U5_R2AF1_ACETONE_TOKENIZER_TOPOLOGY_RESULTS.md`.

`U5.R2AG0` derives a smaller analytic alternative rather than repairing AF1.
The gradient of a fixed log-sum-exp strongly convex potential is an in-cube,
positive-orientation explicit map by construction. AG1 may test its 65 fitted
scalars on O0's existing paired synthetic controls only. Optimal transport
still selects a canonical distribution map rather than identifying a
physical film transform; no real pixels, unpaired fit or stock claim opens.

AG1 closes that compact family without weakening the construction. Exact
repeat, range, identity, analytic SPD/determinant/norm, inverse and
serialization gates pass. The positive-film control passes at `.00945` RGB
RMSE, but the density control remains `.04047` against the frozen `.015`
ceiling; whole/partition execution also differs by 1–2 float64 ulps against
an exact gate. No anchor-count, affine-wrapper, temperature, optimizer or
tolerance rescue opens. O0 remains the retained synthetic-capable safe
representation. See
`docs/U5_R2AG1_BOUNDED_CONVEX_GRADIENT_RESULTS.md`.

`U5.R2AH0` opens a different development-only ML question rather than adding
AG1 capacity or reopening W1. New generated groups hold a known O0 operator
fixed while content and nuisance vary; a hierarchical permutation-invariant
predictor may learn only bounded O0 parameters under explicit invariance,
anti-collapse and content-leakage controls. Deep Sets, VICReg, DANN and
Meta-OT justify components but do not establish reference-only or film
identifiability. W1's reserved confirmation remains unread. No real pixel,
unpaired operator, stock/mode or product gate changes. See
`docs/U5_R2AH0_GROUP_INVARIANT_REFERENCE_OPERATOR_AUDIT.md`.

AH1D then closes that learned reference-only route under its frozen
development contract. Two local-CUDA reports are byte-identical. Median
operator error is `.04001`, but p90 is `.09723`; improvement over
identity/global is only `7.25%/4.89%`. Content remains decodable at `27.10%`
balanced accuracy, fixed-content look separation is only `12.5%`, and the
`53/55/56` strength ordering reverses at Spearman `-1`. O0 range, Jacobian,
inverse, replay and set invariance pass, so this is operator
recovery/shortcut failure rather than renderer corruption. No larger encoder,
loss/adversary search, post-hoc strength correction or AH1C confirmation
opens. See
`docs/U5_R2AH1_GROUP_INVARIANT_REFERENCE_OPERATOR_RESULTS.md`.
