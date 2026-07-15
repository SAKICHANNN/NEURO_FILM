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
| Apollo flight archive | documented mission/magazine/film metadata | magazine/frame | high-resolution scans | extreme lunar/space content and archive processing | public archive access; derivative terms require a source-specific freeze | metadata/sample audit only; unsuitable as a general stock pilot without cross-content evidence |
| FILM-R v2 | filename-family hints only | 44 sibling pairs, single contributor | damaged/restored positives | family-content structural zeros | local CC BY 4.0 freeze | `S0`/artifact stress only; stock learning closed |
| LOC FSA/OWI | stock unknown | creator/location/sequence leakage guards | positive archive scans | age, common archive scanner, borders, restoration | public domain; 258 bounded local Phase-C pixels | independent `H` lane only; sealed while named-stock work advances |
| FilmSet | recipe names, not physical film | exact digital identities | paired Capture One renders | digital recipe | local research source | method/software control only |
| SillyStill CineStill | paper-level stock claim | small paired study described | claimed same-scene pair | availability unknown | data unavailable; repository licence insufficient | blocked |
| Emulating Emulsion | controlled Velvia 100 in publication | one roll and chart patches | controlled chart scans | one-roll/chart domain | public reusable data/licence not verified | method precedent; blocked as a corpus |
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
   count for a density-domain identifiability pilot.
3. `fujifilm_nph_400`: 4 rolls / 53 frames / no aligned proxy; density/preview
   identifiability only until a valid display-positive lane exists.
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
- raw-negative/density descriptors separately from positive/display proxies.

Do not mix negative density and display-positive targets into one loss. If the
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
| preview signal exists but positive lane absent | density-domain research only; no display-colour expert claim |
| train rolls win but unseen roll/source does not | remain `S1`/source-specific; no `S2` |
| local model adds seams, banding or unstable colour | reject local residual and fall back to global operator |
| simple global operator ties ML | promote the simpler explicit operator |
| no current public source can support three `S2` stocks | preserve honest partial result and design a separately approved licensed/controlled acquisition programme |

## 7. Immediate ready leaf and evidence bundle

`SF0.1` is complete. The committed contract cross-checks 13 strings / 53 rolls /
491 frames, seals all official-test rolls, and freezes 189 exact objects /
227,287,697 bytes. Gold 400-5 failed preflight and was replaced by GA 100 5095.
The evidence reruns byte-identically; see
`docs/REAL_FILM_STOCK_PILOT_ACQUISITION_FREEZE.md`.

The next leaf is `SF0.2`: download only the frozen manifest into the isolated
ignored root, verify 189/189 sizes and LFS SHA-256 hashes, reject external lane
files, and keep pixels undecoded. After commit/push, `SF0.3` performs decode,
duplicate/border/content/support audits before RF1.4. No GPU job is justified at
SF0/SF1.
