# Ultimate real-film mainline reopening report

Date: 2026-07-15

Node: `ULT > RF`

Status: active planning authority for Ultimate evidence

## 1. Corrected success condition

Ultimate succeeds only when a bounded explicit colour system learns from
verifiable scans of physical photographic film, remains visibly stylised on
unseen real rolls or independent sources, beats matched-strength pooled and
nuisance controls, and passes original-resolution severe-artifact review.

FilmSet, camera Film Simulations, Capture One recipes, hand LUTs and
pseudo-teachers are controls or pretraining material. They are never final
film truth. Unknown process/scanner data can support only
`real-film-derived/unknown-look`; grouped archive scans can support only an
archive/scanner-specific `roll-look`. `calibrated-reference` requires controlled
stock, process, scanner and whole-roll holdout evidence.

## 2. Evidence ledger

| Evidence | Physical film? | Group/control quality | Current valid conclusion |
|---|---:|---|---|
| FilmSet train/internal/final | no; Capture One digital recipes | exact digital pairs and frozen identities | explicit operators can reproduce some strong digital recipes safely; auxiliary method control only |
| CT8 final-628 | no | 628 untouched digital identities | Cinema Lab and ClassNeg pooled-L2 beat the fixed basic; Velvia pooled-L2 loses fidelity despite stronger style; no real-film conclusion |
| BlueNeg | yes, archive negatives/scans | 491 frames, 53 physical rolls, but usable matched pilot reduced to four Kodak Gold 100-5 rolls and restoration/scanner nuisance | archive/scanner-specific roll information is ambiguous; not digital-to-film or stock truth |
| local Flickr-derived files | claimed film, not sufficiently verified | 4,210-row audit has zero eligible rows and no reliable roll/scanner lineage | quarantined; no training or truth use |
| manufacturer sensitometry | physical prior, not image target | stock/process curves and partial spectral information | initialise/constrain a model only |
| FILM-R candidate | yes; 44 4K 35mm colour scans | stock-like filenames, one contributor; roll/process/scanner unreported | rights-clear real-film target/style and artifact evidence after local acquisition; not stock truth |
| Apollo candidate | yes; original NASA flight film | physical magazines and documented film types; scene domain is extremely narrow | possible roll-identifiability stress source; raw scans are huge and processed products have stricter rights |
| DOCUMERICA candidate | yes; federal documentary slides/negatives | photographer and archival metadata, but no reliable roll/scanner/process fields | diverse public-domain real-film archive look; uploader/era/scan nuisance dominates stock claims |
| SillyStill candidate | yes, paired digital/Cinestill claimed | paper reports 41 raw / 38 processed pairs | full dataset is not present at the official repo and no root data license exists; one example pair is insufficient |
| Emulating Emulsion candidate | yes, controlled Velvia 100 | one 36-exposure roll, 33 chart pairs / 3,168 unique patch correspondences | scientifically strong single-roll calibration precedent; public dataset/license was not found |

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
| FILM-R, Figshare 21803304 v2 | 88 files; 437,570,872 bytes; 44 damaged + 44 restored real scans | CC BY 4.0 | filenames identify 11 stock/format families; one contributor | download now; bounded RF0.1 ready leaf |
| FILM-AA, Figshare 21803292 | 20 files; 122,164,960 bytes; 10 empty damaged frames + annotations | CC BY 4.0 | authentic damage, no scene/roll colour truth | optional artifact-control sibling, not colour training |
| BlueNeg exact revision | 491 previews, 53 rolls, 13 film strings; current bounded lane 118,929,719 bytes | custom attribution license already snapshotted | physical roll/date/location/film string, alignment metadata | already acquired bounded subset; expand only under a preregistered whole-roll gate |
| NASA Apollo flight-film scans | almost 25,000 catalog images; raw examples about 1.2 GB each | NASA raw scans public domain with credit; ASU processed products restrict derivatives/commercial use | magazine, mission, film type, frame; scanner provenance | metadata/sample audit first; do not bulk-download raw |
| DOCUMERICA | about 15,981 online public-domain scans | US federal public domain | photographer, place/date/series; Kodachrome/Ektachrome collection-level history | metadata-only audit, then small photographer/source-stratified sample |
| SillyStill | paper: 41 raw pairs, 38 processed | official repo has no root license and full dataset links remain unavailable | paired tripod scenes, one claimed stock | blocked for training; metadata/reference only |
| Emulating Emulsion | 33 chart image pairs / 3,168 unique patches, one Velvia 100 roll | publication accessible; public data/license not found | controlled illuminant, exposure, camera, D50 scan | method precedent only until data becomes verifiably available |
| archival motion collections | 44 to 81,576+ frames depending source | source-specific and often unclear | reel/frame continuity, authentic damage | artifact research only until stock/scanner/license audit passes |

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
| RF-G1 grouping | roll/uploader/scanner/process fields support leakage-free holdouts | label unknown-look; no roll/stock claim |
| RF-G2 identifiability | correct group beats pooled, shuffled, nuisance and retrieval controls with roll-cluster CI above zero | close group claim; do not scale model |
| RF-G3 style | visible style exceeds safe bland baseline and matched simple enhancement | reject saturation/contrast-only gain |
| RF-G4 transfer | unseen real rolls/sources retain the gain after style matching | source-specific label or stop |
| RF-G5 artifacts | no confirmed severe issue on frozen full-resolution gold; stress rate reported with CI | deterministic fallback or candidate rejection |
| RF-G6 content | face/text/object/geometry retained; intended grain/halation judged separately | reject spatial residual/model |
| RF-G7 claims | label matches evidence: film-inspired, real-film-derived roll-look, or calibrated-reference | fail closed to lower claim |
| RF-G8 reproducibility | configs, seeds, hashes, split groups, software and evidence reports frozen | no promotion |

## 7. Dependency-ordered execution plan

```text
RF0 truth/data gate
  RF0.1 acquire and hash FILM-R under CC BY 4.0
  RF0.2 build source/rights/group manifest; unknown fields remain explicit
  RF0.3 metadata-only Apollo/DOCUMERICA/SillyStill/Emulsion audit
    -> RF1 signal audit
       RF1.1 real-scan distribution and nuisance separability
       RF1.2 BlueNeg nested leave-one-frame-out diagnostic
       RF1.3 freeze any larger whole-roll holdout before pixels
         -> RF2 CPU explicit expert ladder
            global -> hierarchical -> retrieval -> conditional
              -> RF3 GPU bounded challengers only if RF2 leaves residual value
                 -> RF4 unseen source/roll style and artifact confirmation
                    -> RF5 product integration with OOD fallback
```

Each leaf freezes a hypothesis, controls, metric, gate and failure branch before
pixel access. FilmSet remains a software regression and pretraining/control
lane; it cannot promote RF nodes.

## 8. Nearest ready leaf

`RF0.1` is ready: acquire Figshare FILM-R v2 exactly, verify every supplied MD5
and local SHA-256, snapshot the API metadata and CC BY 4.0 license, create a
manifest that pairs damaged/restored scans without inventing roll/scanner
fields, and run a visual/source integrity audit. The 417.3 MB bounded download
is scientifically justified and cannot by itself promote a model.

After RF0.1, the next leaf is `RF0.2`: determine whether filename families are
stock labels, physical rolls or merely contributor naming; until proven, set
`roll_id`, `process_id` and `scanner_id` to unknown and enforce source-level
holdout only.

### RF1.2 update, 2026-07-15

The four-roll 43-query BlueNeg nested LOO diagnosis is complete. Wrong-roll
source-content retrieval beats the correct physical-roll operator on all four
raw roll means; the cluster-equal correct-roll gain is -0.3856 Delta-E with 95%
interval `[-0.4956,-0.2071]`. Physical-roll information remains closed. The
result elevates similar-case retrieval of bounded explicit operators as an RF2
challenger, but only as archive-restoration mechanism evidence until it passes
the real-film RF1.1 gate.
