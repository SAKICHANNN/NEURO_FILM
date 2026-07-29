# Stock-first real-film research programme

Date: 2026-07-15

Node: `ULT > RF0.4-RF5`

Status: active execution plan

## 1. Objective and non-goals

Build a stock-selectable bank of explicit colour operators learned from
verifiable physical-film evidence. A successful expert must be visibly
stylised, distinguishable from the other promoted stocks after matching gross
contrast/saturation, transferable to an unseen roll or source, and free of
confirmed severe artifacts on the frozen full-resolution gold set.

`film_stock_id` is the primary class. Roll, process, lab, scanner, source,
content, era, ageing and restoration are nested controls. Missing facts remain
`unknown`. A user chooses the output stock; content-aware selection may happen
inside that stock but never substitutes one stock for another.

After stock identifiability, the conditional `H-LSM-1` programme may test
whether one stock contains multiple stable latent modes. It does not presume
that it does: `K=1` remains a formal result, no current stock has proved
multiple transferable modes, and unpaired digital-to-film operator
identification remains unresolved. See
`docs/planning/LATENT_STOCK_MODE_PROGRAM_2026.md`.

This programme does not treat a Capture One recipe, camera simulation, LUT,
filename hint, generic old-film archive or learned average of all film as stock
truth. Direct generative RGB editing remains excluded.

## 2. Separate coverage accounts

- Named stocks use `S0 claimed`, `S1 research-eligible`, `S2 transferable` and
  `S3 calibrated` evidence grades.
- Physical historical film with unknown stock uses `H historical/unknown`.
- An `H` expert can be useful, but never increments named-stock coverage.
- First milestone: at least three distinguishable `S2` stock experts plus one
  independent `H` expert. Research continues beyond that milestone.

The machine-readable authority is `configs/film_stock_registry.json`; its
validator cross-checks every BlueNeg stock string, roll count and frame count
against the exact-revision frozen metadata.

## 3. Obtainable-source matrix

| Source | Label authority | Grouping | Positive/display lane | Nuisance ceiling | Rights/access | Decision |
|---|---|---|---|---|---|---|
| BlueNeg | dataset-declared `film_type` at frozen revision; not edge-code/manufacturer proof | 53 physical rolls / 491 frames | 491 negated previews; 247 printed-photo proxy alignments | old negative, restoration target, unknown process/lab/scanner | public academic/commercial use with mandatory credit; bounded lanes about 956 MB | only immediately executable multi-stock source; select four pilots but keep all below `S2` until controls pass |
| Xi Film marketplace | contributor/platform-verified stock, camera and scan per licensed frame | contributor/source; roll identity not advertised | high-resolution positive scans | lab/scanner/process fields require contract | paid per-image licensing and manual delivery | promising future source-diversification lane; no purchase or training until licence explicitly covers ML and grouping is verified |
| Apollo flight archive | documented mission/magazine/film metadata | magazine/frame | raw flight-film scans remain rights-separated and unacquired | extreme orbital content, exposure/filter association and archive processing | NASA raw scans public domain with credit; ASU processed products restricted | `SF2.0A` closes Apollo 7 as content-confounded after 63/63 valid pages; no pixels or learning |
| NASA/JSC astronaut Earth photography | official exact media codes and mission-roll-frame IDs | mission/film roll/frame | photographs remain unrequested | mission/date/roll/geography/focal length/camera and scan pipeline can dominate | NASA/JSC credit/non-endorsement/third-party caveats; keyless public HTML query | `SF2.0C0` closes the source expansion: 25 missions, but only failed STS098 joins Velvia50/Portra400NC |
| Openverse search index | weak exact title/tag text only; upstream label unverified | source plus creator URL | no pixels requested | relevance pagination repeats, indexed licence may be stale, Flickr/source/content dominance | anonymous official API; every upstream right still requires live verification | `SF2.1A` closes: Ektar repeats 19 identities and only UltraMax passes per-stock gate |
| Smithsonian Open Access metadata | institution-authored object/archive metadata, usually stock-family rather than exact product identity | owning unit, collection and record; no demonstrated connected exact-stock graph | no pixels requested | unit/collection/era/content/scanner can become stock proxies; media rights are separate | official CC0 metadata mirror; seven relevant units total 6.64GB | `SF2.2R` closes before a formal audit: deterministic shard probes split Kodachrome and Ektachrome evidence across units and expose no exact connected multi-stock design |
| Newgrain public application | controlled exact-stock catalogue, but post labels remain community claims | pseudonymous user plus optional lab/scanner/process fields | no pixels requested or retained | platform/source workflow and self-report remain nuisance; no rights-cleared manifest | public frontend is technically readable, but Terms prohibit automated queries/scraping/mining and user content remains uploader-owned | `SF2.4R` closes before a formal audit; written permission or an officially licensed export is required |
| PROV VPRS 17684/17690 | institutional digitised negative collection linked to a physical register described as tracking film stock | same agency and negative-number ranges could supply strong grouping if register contents become accessible | SF2.5R metadata-only result; pixels forbidden | two byte-identical audits find all 30 register items physical-only and no digital/IIIF field | official API is permitted for non-commercial research under CC BY-NC; image reuse is not inferred | source closes before pixels; reopen only on official transcription/digitisation/licensed export or separately approved bounded physical-copy workflow |
| ColorReference Set 3 | exact Velvia 100F target-set statement | same five physical slides across four scanner/software pipelines | 20 scaled scans + five recorder-space grids + IT8/CGATS measurements | deliberately exposes scanner hardware/software nuisance; one target set only | free testing/development use stated, no standard redistribution licence | SF2.7R source pass; internal nuisance-control lane only, never stock fitting |
| ColorReference Velvia 100F recorder-source six-set lane | exact material header on every measured member | six target-set/charge IDs x five slides; shared production date 2005:05, roll/process unknown | five common recorder-device grids plus exact 8,640 source/measurement rows | source is unknown-profile recorder RGB; no camera scene or scanner RGB; set/charge and measurement nuisance remain | free testing/development statement, no standard redistribution licence | AQ1 repeatability passes; AQ2 joint held-grid operator identifiability closes with no eligible bounded champion or capacity rescue |
| NTNU controlled reversal study | publication/thesis exact Ektachrome E100 and Velvia 50 | two mock-up paintings; independent roll/process grouping unreported | matching-geometry hyperspectral captures and ten-band developed-film scans described | six analysed frames do not supply same-illumination cross-stock control | article CC BY; NVA thesis files use general Copyright Act terms; no dataset release | SF2.8R method evidence only; public data unavailable and stock effect unidentified |
| ColorReference multi-family IT8 references | four exact material headers plus one Ektachrome family header | one manufactured batch-average target per family; no independent roll/process groups | 288 common target IDs with Lab/density and 380--780nm spectra | target aim/manufacturing adjustment is unobserved; no camera-scene or common recorder input | public download page, no standard reusable data licence | SF2.9R physical spectral and target-manufacturing nuisance evidence only; no operator fitting |
| Color Precision comparison metadata | 20 URL-derived stock-looking labels; not verified physical IDs | 471 filename condition keys and 456 apparent two-scanner pairs; physical roll/process unknown | no pixels requested; comparison page embeds 927 S3 image URLs | scanner auto-adjustment and minor exposure edits disclosed; pair registration and independent replication unverified | public HTML repeats; all-rights-reserved footer and no explicit comparison-pixel research reuse grant | SF2.10R preserves promising scanner-nuisance topology but stops before pixels/fitting; reopen only with permission, replication and verified manifest |
| FILM-R v2 | filename-family hints only | 44 sibling pairs, single contributor | damaged/restored positives | family-content structural zeros | local CC BY 4.0 freeze | `S0`/artifact stress only; stock learning closed |
| LOC FSA/OWI | stock unknown | creator/location/sequence leakage guards | positive archive scans | age, common archive scanner, borders, restoration | public domain; 258 bounded local Phase-C pixels | independent `H` lane only; sealed while named-stock work advances |
| FilmSet | recipe names, not physical film | exact digital identities | paired Capture One renders | digital recipe | local research source | method/software control only |
| SillyStill CineStill | paper-level stock claim | 41 raw / 38 processed pairs described | one illustrative pair is public | one-stock/small-study domain | SF2.6R: dataset links remain placeholders and no root data licence exists | blocked; one example is not a fitting corpus |
| Emulating Emulsion | controlled Velvia 100 in publication | one roll, 33 chart pairs / 3,168 unique patches | no measurements or fitted parameters released | one-roll/chart domain | SF2.6R: method/figures public, but no reusable dataset/code licence | method precedent; blocked as a corpus |
| Manufacturer data sheets | authoritative product/process prior | no scene groups | no RGB target | measurement/interpretation mismatch | document-specific terms | curve/sensitivity constraints only |

This matrix records attainable evidence, not aesthetic reputation. New sources
enter only after URL/revision/licence/credit/hash/group and allowed-use fields
are frozen.

Primary source checkpoints:

- BlueNeg dataset card and metadata/licence contract:
  https://huggingface.co/datasets/ttgroup/blueneg-release/blob/b038a1ae68f42067ff12b5e79ddbe62919b7af23/README.md
- BlueNeg ICCV 2025 paper:
  https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.pdf
- Xi Film marketplace provenance/licensing description:
  https://www.xifilm.art/
- Apollo scan metadata description:
  https://apollo.im-ldi.com/ABOUT_SCANS/index.html
- FILM-R public dataset record:
  https://figshare.com/articles/dataset/Authentically_damaged_film_scans/21803304

## 4. First four pilots

The bounded first pass uses four exact BlueNeg labels because their metadata
and pixels are obtainable now and together permit a real wrong-stock control:

1. `kodak_gold_100_gen5`: 19 rolls / 226 frames / 189 public aligned frames;
   strongest sample base, but the prior four-roll result is nuisance-negative.
2. `kodak_ga_100_5095`: 3 unsealed rolls / 16 frames / no aligned proxy;
   exact dataset-declared product/code string and the minimum independent-roll
   count for a post-negation-preview label/shortcut identifiability pilot.
3. `fujifilm_nph_400`: 4 rolls / 53 frames / no aligned proxy;
   post-negation-preview identifiability only until a valid proxy lane exists.
4. `konica_super_xg_100`: 8 rolls / 30 frames / 6 aligned frames; manufacturer
   diversity, but the aligned lane is too small for promotion without a
   whole-roll/source-safe split.

The initial paper selection included `kodak_gold_400_gen5`, but whole-test-roll
preflight reduced it from 3 nominal rolls / 38 frames to 1 unsealed roll / 1
frame. It was therefore removed before acquisition and replaced by GA 100 5095.
This failure is retained as evidence that nominal dataset counts cannot choose
pilots before split sealing.

Selection means “audit first”, not “stock expert established”. `Fuji 100` and
`Fuji 400` remain `S0` because the product line is ambiguous. Every pilot has
`market_status`, process, lab and scanner recorded as `unknown` unless primary
evidence is later frozen.

## 5. Dependency-ordered experiment DAG

### SF0 — truth and acquisition

1. Validate the registry and exact BlueNeg counts byte-identically.
2. Freeze exact preview/pseudo-GT paths and byte/hash totals for the four stocks.
3. Seal every official-test roll and any perceptual sibling across splits.
4. Download only the frozen bounded lanes; verify every LFS SHA-256 and decode.
5. Publish stock × roll × content × partition × target-availability matrices.

Stop a stock at `S0/S1-candidate` if its label is ambiguous, fewer than three
independent learnable groups remain, or stock and content have structural
zeros that no matched control can resolve.

### SF1 — identifiability before colour fitting

For each stock, preregister and run:

- label audit and shuffled-label negative control;
- stock-by-content support and comparable-stock clique;
- leave-one-roll-out and, if possible, leave-one-source/scanner-out;
- correct-stock versus pooled-all-stock, wrong-stock, content retrieval,
  generic historical, and WB/curve/saturation controls;
- post-negation 8-bit preview descriptors separately from display proxies.

Do not treat the post-negation preview as physical density or mix its
descriptors with display-proxy targets. If the
correct stock cannot beat matched nuisance/retrieval controls with a
roll-cluster confidence interval above zero, close that stock mechanism rather
than add model capacity.

### SF2 — CPU explicit-operator ladder

Run the smallest survivor in this order, independently per stock:

1. robust global monotone curves plus 3D colour operator;
2. hierarchical stock mean plus bounded roll residual;
3. hard Top-1 stock-internal medoid/case expert with OOD fallback;
4. sparse mixture inside the selected stock;
5. optional bounded bilateral-grid residual.

All candidates render through an inspectable operator. Regularise monotonicity,
LUT smoothness, gamut, local-residual amplitude and temporal/spatial stability.
The deterministic safe-rich renderer is the fallback, not the teacher truth.

### Conditional LSM — within-stock mode hypothesis

Before SF2 can expose more than a global stock champion, the stock must pass
label/rights, connectivity, stock-identifiability, pixel and leakage gates.
`LSM0` freezes ontology and the observed/latent split; `LSM1` produces a
feasibility decision without training or clustering. Only a complete LSM1 pass
may open residual appearance/operator identifiability and a preregistered
`K=1` versus group-aware mixture audit.

Mode space is separate from within-mode content retrieval. Clusters explained
by strength, content, source, scanner, geometry or basic EV/WB/contrast/
saturation/luma are merged or rejected. Physical labels remain `unknown` or
`hypothesis_only`. Even stable modes do not enter product routing unless a
fixed-bank Evaluator Oracle significantly beats the stock global champion and
the full-resolution severe veto passes.

### SF3 — bounded ML challenge

Only if SF2 leaves measured residual value may a small encoder predict curve,
LUT mixture, case id, local-grid or effect parameters. Train with group-balanced
sampling, stock-conditional heads and an explicit anti-collapse/distinctiveness
term. Prohibit final-RGB generators, cross-stock averaging, random frame
splits, and promotion on training-source aesthetics alone.

### SF4 — confirmation and product integration

- frozen unseen roll/source confirmation;
- style-strength-matched stock distinctiveness;
- blind visual preference and “recognisable stock-like character” review;
- original-resolution severe-artifact veto, then graded content diagnostics;
- deterministic replay, hashes, provenance and OOD fallback;
- per-stock `S1/S2/S3` claim and separate `H` reporting.

## 6. Promotion metrics and failure branches

Promotion order is severe-artifact veto, visible style/appeal, stock
distinctiveness/transfer, content retention, then product reliability.
Saturation, contrast and mean cast are nuisance-matched before attributing a
gain to stock identity.

| Observation | Decision |
|---|---|
| outputs collapse after matched-strength normalisation | reject pooled/averaged representation; retain independent experts or close unsupported stocks |
| wrong-stock or content retrieval wins | label content/source-confounded; do not scale |
| post-negation preview signal exists but proxy lane absent | label/shortcut research only; no physical-density or display-colour expert claim |
| train rolls win but unseen roll/source does not | remain `S1`/source-specific; no `S2` |
| local model adds seams, banding or unstable colour | reject local residual and fall back to global operator |
| simple global operator ties ML | promote the simpler explicit operator |
| no current public source can support three `S2` stocks | preserve honest partial result and design a separately approved licensed/controlled acquisition programme |

## 7. Immediate ready leaf and evidence bundle

`SF0.1`, `SF0.2` and `SF0.3` are complete. Download verification and decode
integrity both rerun byte-identically. Integrity report SHA-256
`a93257ee45e14ac0519dc1ce76a840ebe4f517a8d5c413e33b17c915160d1caa`
at commit `3c52a31a...` shows 189 RGB PNG files, zero exact duplicates and
zero cross-frame dHash≤4 pairs after fail-closed manifest/metadata cross-checks.
All eight contact sheets were reviewed and expose substantial shortcut risk.
See `docs/REAL_FILM_STOCK_PILOT_INTEGRITY_RESULTS.md`.

`RF1.4A` closes preview-only stock learning. GA100/Konica fail structural
support; NPH400/Gold100 gives primary roll accuracy 0.727, permutation p=0.191
and a stronger 0.909 nuisance shortcut. `RF1.4B0` passes all 47 official
bbox/proxy pairs across six rolls. `RF1.4B1` then passes every frozen metric
and visual gate for the archive preview-to-display chain, but bounded 3x3
affine (5.307 Delta E76) slightly beats SepLUT17+3x3 (5.380). `RF2.S0` closes
direct transplant: archive-only OOD coverage passes, but full-strength style is
6.25 below the 7.0 floor, matched-basic residual is 1.94 below 4.9 and worst
gold clipping is 8.70%; weaker strengths get blander.

The community-source sequence is now much further advanced. Commons retains
clean provisional `S0` Ektar100 (26 files/eight authors) and UltraMax400
(37/eight), while YFCC retains Velvia50 (25/14 UIDs) and a diagnostic Ektar100
bridge (16/five). Every retained pixel passed bounded rights, decode,
duplicate, content and repeated visual severe-artifact review. These facts
establish usable aesthetic/failure-analysis pools, not stock response.

`SF1.0B` closes all current pools for stock learning. In the Commons edge,
global RGB reaches only 59.82% balanced accuracy with permutation p=0.280,
while 4x4 scene colour reaches 80.36%/p=0.024. In the YFCC edge, RGB reaches
65.71%/p=0.188, while luma/HOG/geometry already reach 72--76% and 4x4 scene
colour reaches 89.29%/p=0.002. Same-stock Ektar source geometry is
92.86%/p=0.016. Larger models and colour-operator fitting are therefore
forbidden on these cells.

`SF1.1` is complete. The exact 65,644,027,904-byte public YFCC100M SQLite
passes SHA-256 and its 7,826-part S3 ETag; two full scans are byte-identical.
Ektar100/Velvia50 passes the frozen metadata edge at 780/240 rows, 122/62 UIDs
and 16 shared UIDs. Ektar/UltraMax fails because UltraMax has only 27 UIDs
versus the frozen 30. No threshold is changed.

`SF1.2` passes: 61 bounded HTML requests confirm eight shared authors with a
current CC BY 2.0 page for both stocks; no image URL was requested. `SF1.3A`
retains 37 clean derivatives across all eight bilateral UIDs and passes
integrity, duplicate and autonomous full-resolution visual gates.

SF1.3A remains an `S0` acquisition/diagnostic pilot. It opens only the frozen
`SF1.3B` shared-author leave-one-UID-out stock-identifiability diagnostic. It
fails because global RGB is 56.25%/p=.464 and weaker than 68.75% nuisance
controls. The pool is closed for stock learning; no training, operator fitting,
LSM or GPU work is justified.

`SF2.0A` is complete and closes rather than becoming a model fallback. All 63
deterministic NASA/JSC Apollo 7 HTML pages validate across two SO-368 and five
SO-121 magazines. Support, the filter-free cross-stock bridge and three
reported exposure states pass, but only `spacecraft_hardware` spans the
required rows and two magazines per stock. The frozen requirement was two
shared content tags, so the result is `content_confounded`. Do not expand to
more Apollo 7 pages or pixels, and keep training, fitting and LSM forbidden.

`SF2.0B0` is complete and closes before a nuisance audit. The keyless snapshot
validates 168 Velvia 50, 329 Portra 400NC and 10 Portra 400VC STS098 rows with
zero cross-code ID overlap. Portra 400NC spans 12 rolls, but Velvia spans only
rolls 701 and 720A, below the frozen four-roll primary minimum. Do not open
SF2.0B1, lower the gate, substitute a post-result code or request photographs.

`SF2.0C0` is also complete and closes the broader exact-code source expansion.
Its six-request, aggregate-only census covers 25 missions and 13,255/397/122
rows for `VELVI`/`5775`/`5776`, but only STS098 contains both primary stocks.
Velvia therefore remains at two supported rolls below the frozen four-roll
gate. No alternative candidate mission exists; do not change codes or gates,
request photographs, fit/train, or open LSM from this result.

`SF2.1A` is complete and closes the Openverse relevance-search route. Its 48
bounded API requests return 240 rows per stock with no image access, but Ektar
contains 19 repeated Openverse IDs/landing URLs and violates the preregistered
stable-census contract. Only UltraMax passes the strict per-stock gate; Velvia
has one strict row, while Ektar and Portra fail creator dominance. Do not apply
post-result deduplication, relax licence/creator gates, open live pages or use
this index for fitting, training or LSM.

`SF2.2R` completes a bounded institutional-source reconnaissance without
opening a formal experiment. Smithsonian Open Access is an authoritative,
machine-readable metadata source, but fixed 8/256-shard probes across seven
photography-relevant units find family-level Kodachrome concentrated in
NMAH/EEPA and sparse Ektachrome in SIA, with no exact connected multi-stock
design. The seven complete unit indexes total 6,642,017,079 bytes. Do not fetch
that corpus merely to union disconnected unit/source/content signatures; no
pixel, fitting, training or LSM branch opens. See
`docs/REAL_FILM_INSTITUTIONAL_SOURCE_RECONNAISSANCE_RESULTS.md`.

`SF2.3` is complete and closes before live verification. Its byte-identical
offline audits retain 294 strict rows across 19 exact stocks and 30 normalized
author strings, with zero identity conflicts. Only Ektar100 and UltraMax400
pass the per-stock support gate, and they share one author rather than the
frozen two. There are zero retained edges and no component. Do not merge
unverified aliases or relax the graph after this result; no live pages, pixels,
fitting, training or LSM open. See
`docs/REAL_FILM_COMMONS_UNION_CONNECTIVITY_RESULTS.md`.

`SF2.4R` closes a Newgrain source-design reconnaissance before any formal
audit. The public application exposes a 385-entry approved stock catalogue and
the desired pseudonymous-user/lab/scanner/process schema, but its published
Terms prohibit automated searches, requests, queries, scraping and mining.
The project therefore retains no snapshot, implements no client and requests
no images. Written platform permission or an officially licensed export would
be a new external-authority branch; technical accessibility alone is not DoR.
See `docs/REAL_FILM_NEWGRAIN_SOURCE_RECONNAISSANCE_RESULTS.md`.

`SF2.5R` closes the PROV register linkage at machine accessibility. VPRS 17684
has 6,716 catalogued digital items and VPRS 17690 is described as its
film-stock register, but two normalized API passes find all 30 register items
physical-only with no digital/IIIF field. Downloading the image collection
without the register would not create stock labels, so no pixels, fitting,
training or LSM open. See
`docs/REAL_FILM_PROV_NEGATIVE_REGISTER_RECONNAISSANCE_RESULTS.md`.

`SF2.6R` refreshes the two controlled paired-source claims most relevant to an
explicit film operator. The pinned SillyStill repository still has placeholder
dataset links, one illustrative pair and no root licence. The Emulating
Emulsion project publishes its two-matrix/three-sigmoid method and experiment
design but not its measurements, fitted parameters, source code or reusable
data licence. Both remain method/metadata precedent only. The functional form
may be tested separately with original synthetic parameters, but no stock fit,
training, calibration or latent-mode claim opens. See
`docs/REAL_FILM_PAIRED_SOURCE_AVAILABILITY_RECONNAISSANCE_RESULTS.md`.

`SF2.7R` adds a strictly nuisance-focused same-physical-slide control. The
exact 71,068,957-byte lane contains four complete scanner/software pipelines
over the same five Velvia 100F Set 3 slides plus recorder-space grids and
measured references. Integrity and connectivity pass, but the source TIFFs are
not neutral digital captures and one target set is not independent-roll stock
evidence. Only a separately frozen scanner-nuisance quantification opens; no
fit, training, LSM or redistribution opens. See
`docs/REAL_FILM_COLORREFERENCE_SCANNER_NUISANCE_RESULTS.md`.

`SF2.7A` quantifies that control without treating untagged scanner RGB as
colourimetric data. Across all directed leave-one-slide-out comparisons, raw
median device-RGB distance is `.06055`; a bounded full affine reduces it
`76.11%` to `.01446`. Pair medians range from `.01464` for two
LS50/NikonScan devices to `.09865` for LS50/VueScan versus LS9000/NikonScan,
so the preregistered every-pair material gate fails. Future stock evidence must
report both raw and canonicalized results and must never relabel scanner
clusters as stock modes. See
`docs/REAL_FILM_SCANNER_NUISANCE_QUANTIFICATION_RESULTS.md`.

`SF2.8R` audits the strongest newly published controlled-film design found in
the current search. Fresh Ektachrome E100 and Velvia 50 were captured on one
Rolleiflex across two paintings, two illuminants and exposure variants, with
matching hyperspectral references and ten-band film scans. The exact public NVA
record contains a thesis and submission archive but no raw cube/TIFF,
measurements, code, manifest or dataset licence. The six frames selected for
analysis also lack a same-illumination cross-stock comparison, so stock and
illuminant remain confounded in the reported visual contrast. No fitting,
training or LSM opens. See
`docs/REAL_FILM_NTNU_CONTROLLED_REVERSAL_SOURCE_RESULTS.md`.

`SF2.9R` removes scene content and scanner RGB by auditing five direct
ColorReference IT8/ISO 12641 transmission-reference archives. All five pass
exact byte/CRC/parse gates twice, share 288 target IDs and provide 41-point
380--780 nm spectra. Pairwise common-patch target Lab medians span
`1.52--4.22` Delta E76. These numbers are not stock-look effect sizes: the
targets are batch-average scanner-calibration products manufactured toward
common IT8 aims, and no common uncalibrated recorder input, camera scene or
independent roll/process replication is lineaged. Retain only internal
physical-spectral and target-manufacturing nuisance evidence. No fitting,
training, LSM, redistribution or production integration opens. See
`docs/REAL_FILM_COLORREFERENCE_MULTIFAMILY_IT8_RESULTS.md`.

`RF2.C0` is also complete as a strictly separate external-control sibling. At
the pinned spektrafilm revision, Ektar100/fixed-e0 reaches 8.034 style and 7.343
matched-basic residual Delta E76 with zero new hard clipping and no confirmed
severe failure in three blind rounds plus 27 full-resolution external reviews.
It is retained only as a future Look Approximation comparison. Center-auto
colour-negative profiles span about 0.2x--4.2x scene mean luma and are retained
as negative adaptation evidence; they do not establish stock distinction.
This result uses no eligible real-film pixels and does not change any SF gate,
open operator fitting, provide teachers or raise an evidence grade. See
`docs/REAL_FILM_SPEKTRAFILM_EXTERNAL_CONTROL_RESULTS.md`.

The `U5.R2AE0` spectral_film_lut source audit is an algorithm-control sibling,
not new stock evidence. Although its current MIT source exposes named material
objects and a headless LUT path, the embedded profile rows lack immutable
document/page/uncertainty identities and historical datasheets cover only a
subset. AE1 may test reproducible structural diversity on a fixed synthetic
cube, but none of its names count toward `S0`, connectivity, identifiability,
pixel eligibility, fitting, training or latent-mode DoR.

AE1 subsequently rejects the external bank before photographs: despite strong
non-basic differences in every common-output family, all eight sampled LUTs
exceed the frozen local-orientation reversal ceiling. This cannot update any
stock evidence grade or become a teacher. It is retained only as motivation
for a separately defined orientation-preserving explicit representation.

AF0 is another algorithm-control sibling, not a stock-data node. AceTone's
published Qwen/GRPO reference selector is generative and excluded; its
training LUT corpus is not an evidence-backed film corpus. The exact released
VQ-VAE may be tested only on synthetic, analytic safe LUTs to determine
whether compression introduces colour-space folds. Even a pass would be
representation evidence only and cannot change any SF/RF/LSM data gate.

AF1 closes the tokenizer representation with all controls intact: each of
eight safe analytic LUTs reconstructs with substantial orientation reversal
and fidelity error. This is generic learned-representation negative evidence,
not a stock-data result, and changes no SF/RF evidence grade or LSM gate.

AG0/AG1 are likewise representation controls outside the stock evidence DAG.
The bounded convex-gradient map may be safe and compact, but fitting it to
unpaired stock histograms remains forbidden and would not identify a film
operator. No outcome changes current stock, pixel, connectivity or LSM gates.
AG1 now closes on insufficient density-control capacity: all analytic safety
properties hold, while the frozen density fidelity gate fails. This does not
alter any stock evidence grade or authorize a larger unpaired model.

AH0/AH1 are also outside the stock evidence DAG. They may test whether
same-known-operator/different-generated-content supervision can learn a
content-resistant reference representation that predicts only bounded O0
parameters. Synthetic labels do not exist for current scans; any pass remains
computational mechanism evidence and leaves stock, pixel, connectivity,
operator-identification and LSM gates unchanged.
