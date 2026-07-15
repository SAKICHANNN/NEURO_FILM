# Roll2Film Research Program — Colour-Transfer Algorithms from Unpaired Roll Sets

> **Superseded priority notice, 2026-07-15:** Ultimate is now a real-film-first
> program governed by `docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md`.
> FilmSet E1/CT5/CT8 is only a digital Capture One recipe control. Roll2Film is
> one falsifiable challenger among global, hierarchical, retrieval and bounded
> conditional explicit-operator methods; BlueNeg did not establish roll
> information. The stock-first programme in
> `docs/planning/STOCK_FIRST_REAL_FILM_PROGRAM_2026.md` is authoritative over
> every data/priority statement below. Nothing below may promote a method
> without per-stock real-film RF gates.

**Status:** conditional publication-oriented research hypothesis, amended 2026-07-15

**Paper type:** colour-transfer method, not a benchmark paper

**Current execution state:** fixed-budget affine and L2 method controls plus the local FilmSet pair-blind freeze pass; roll information remains unestablished and matched-strength baselines are next

**Download/training state:** FilmSet is local and its pair-blind/access freeze passed; the owner pre-authorized necessary downloads, but BlueNeg still waits on metadata/whole-roll scientific gates and costly GPU, participant, release or external-contact actions remain separately gated

> **2026-07-15 integration correction:** the current E0 is a restricted
> estimator unit test, not evidence that roll labels add information, because
> frame count and total target pixels changed together. The product route is
> independently winnable by fixed explicit experts. Roll2Film becomes the paper
> core only after fixed-total-sample grouping/nuisance controls pass. See
> `ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_INTEGRATION_20260715.md` for the evidence
> adjudication and revised order.

This plan corrects the previous benchmark-first emphasis in
`FARO_RESEARCH_PROGRAM_2026.md`. ChromaticTail/FilmStyleSafe remains useful as
an evaluation instrument and FARO remains useful as a product safety wrapper,
but neither is the primary paper. The primary paper must learn and execute a
colour transformation. If that algorithmic hypothesis fails, this research
route closes; it does not fall back to a standalone benchmark paper.

---

## 1. Decision and research contract

### 1.1 User objective

The desired system maps a digital photograph to a strong, recognisable
film-inspired colour rendering while preserving the photographed content. It
must avoid failures such as a smooth sky acquiring a large unexplained purple
region, but it must also avoid the opposite failure: averaging every training
sample into a weak generic enhancement.

The publication question is therefore:

> Can a reusable, strong colour operator be identified from **unpaired images
> grouped by physical film roll**, when scene content, exposure and scan
> nuisance are not paired to the digital source domain?

Working paper name:

> **Roll2Film: Identifying Reusable Film Colour Operators from Unpaired Roll
> Sets**

The strongest currently defensible novelty hypothesis is the supervision unit,
not a new name for a LUT, optimal transport, a flow or a benchmark:

> Multiple content-diverse frames from one roll are repeated observations of a
> shared roll-level imaging operator. Joint inference across that group can
> reduce the content/style ambiguity that makes single-image unpaired colour
> transfer under-identified.

This is a hypothesis, not yet a `first` claim. A dedicated grouped-observation
and imaging-system-identification search remains a gate before submission.

### 1.2 Required output

Given a digital image `x` and an inferred roll/look operator `T_r`, the method
must return

```text
y = T_r(x)
```

where `T_r` is replayable and inspectable. The first paper permits matrices,
monotone tone splines and a smooth invertible colour-volume residual. It does
not permit a decoder, diffusion denoiser, pixel generator or spatial resampling
to produce the final RGB image.

Grain, halation, bloom, dust and geometric texture are outside the colour-paper
method. They remain independent product effects.

### 1.3 Claim ladder

| Evidence level | Allowed claim | Forbidden claim |
|---|---|---|
| A — FilmSet hidden-pair evaluation | unpaired **film-recipe** or film-inspired colour transfer | real-film or named-stock fidelity |
| B — grouped real-film observations | roll-conditioned film-inspired transfer and group-level operator evidence | digital-to-film ground truth or stock/process separation |
| C — controlled paired rolls and scans | calibrated stock/process/scan profile within the measured capture contract | universal stock truth across unmeasured labs, scanners or cameras |

The primary paper can be publishable at A+B if it clearly says
`film-inspired` and demonstrates an algorithmic advance. A named Portra,
Velvia or Vision3 authenticity claim requires C.

### 1.4 Definition of done

The method-paper route is complete only if all of the following hold:

1. the frozen method transforms source images rather than merely scoring them;
2. group-conditioned inference beats matched single-reference and pooled
   unpaired operator baselines on hidden transfer targets;
3. increasing the number and diversity of roll frames improves operator
   recovery under preregistered controls;
4. output style remains visibly strong rather than collapsing toward identity;
5. severe chromatic artifacts do not increase relative to the strongest
   eligible explicit-operator baseline;
6. claims stay within the data level above;
7. code, splits, hashes, operator recipes and failure cases are reproducible.

---

## 2. Data reality: local bytes and verified remote access

The prior plan recognized claim boundaries but did not make current physical
availability and scientific eligibility central enough. File quantity is not
the same as usable supervision.

For this project, `available data` includes both local files and public remote
files that can actually be fetched. The audit uses four distinct states:

| State | Meaning |
|---|---|
| local-ready | bytes are on this machine and pass provenance/claim gates |
| remote-verified | a current file listing and an actual sample/range download succeeded; the full corpus is not yet local |
| published-not-verified | a paper/page mentions data, but no working file endpoint was demonstrated |
| unavailable | request-only, gated without access, placeholder, deleted or broken |

A dataset can be `remote-verified` yet still be scientifically quarantined if
its image rights, grouping or labels are inadequate.

### 2.1 Local audit, amended 2026-07-15

| Lane | Physical state | Scientific state | Valid use now |
|---|---|---|---|
| `data/film_domain` | 4,212 JPEGs on the current Windows host | **quarantined by default**. The legacy audit covered 4,210 manifest rows and found 0 eligible rows, no durable source/uploader/roll/scanner grouping, unknown Flickr-use scope and 140 near-duplicate pairs. Its verdict must not be silently extended to the two additional files, which require lineage propagation. | internal failure exploration only; not paper training, released weights or stock truth |
| `data/filmcase_references/velvia_50` | 26 unique files / about 50 MiB; CSV has 28 rows because two sources are duplicated; 8 owners, with 15/28 rows from one owner; 21 CC BY-SA 2.0, 5 CC BY 2.0, 2 CC0 | rights are traceable, but roll/process/scanner identity and source diversity are insufficient | interface and few-shot smoke only; not the main training or generalisation evidence |
| `data/ip2p_train` | 100 pairs at 256 px, one Portra-400 prompt | synthetic pseudo-pairs: each “digital” input was generated by WB/gamma corruption of its film target | dataloader and training smoke only |
| FiveK source and `freeze_v1` | partial 903-file freeze present on the current Windows host; complete source absent | FiveK is neutral retouching, not film truth | generic operator/canonicalisation auxiliary only after a separate manifest/rights decision |
| FilmSet decompressed image tree | local: 21,140 files / 11,262,805,356 bytes; 4,657 train and 628 test identities per domain | Capture One recipe pairs, not physical film; final 628 must be an access-controlled one-shot lockbox | internal pair-blind recipe transfer and paired upper bound after manifest freeze |
| `data/physics` | 11 of 13 recorded manufacturer PDFs are present | physical priors, not RGB mapping targets | tone/response constraints and hypothesis design |
| CIE and camera spectra | local calibration tables and 40 recorded camera sensitivities are present | usable academic priors subject to their individual terms | colour management, sensor simulation and synthetic operators |

Two consequences are non-negotiable:

- there is no real same-scene digital/film pair on this computer;
- the 8.4 GiB Flickr directory cannot rescue the paper, because it lacks the
  very grouping and provenance needed by the hypothesis.

### 2.2 Public, recoverable lanes

| Resource | Reachability checked 2026-07-12 | Data value for this paper | Decision |
|---|---|---|---|
| FilmSet | **local on current Windows host**; remote source remains verified | 5,285 digital originals + three paired film-recipe targets; hidden-pair transfer evaluation | primary Level-A corpus; freeze locally without another download |
| BlueNeg | **remote-verified** by file tree and actual range read | 491 physical-film frames, 53 roll groups, 13 film-type strings | primary Level-B mechanism pilot |
| LOC FSA/OWI colour archive | **metadata/pilot/group verified**; public domain | 558 unique LOC scans; 258 bounded derivatives currently retained; 457 known-creator records; 43 conservative creator/location groups | sealed `H historical/unknown` auxiliary lane; never named-stock coverage |
| DigitalFilm_dataset | **remote-verified** by file tree and actual ZIP range read | about 3.35GB of unpaired film-labelled images | quarantined: Internet-image lineage is insufficient |
| PhotoGAN C200 | paper description found; no first-party corpus endpoint verified | unpaired Fuji-C200-labelled/digital images | not counted as available |
| SillyStill CineStill pairs | repository found; dataset links still say “Not yet available” | tiny real same-scene paired set if ever released | unavailable |
| Emulating Emulsion chart pairs | paper found; public data endpoint not verified | controlled physical calibration | baseline only; data unavailable |
| FilmGrainStyle740k | request-by-email lane | grain, not colour mapping | not currently downloadable and not a colour-paper dependency |

#### FilmSet — primary algorithm-development lane

The [FilmSet paper](https://www.ijcai.org/proceedings/2023/0129.pdf)
describes 5,285 RAW originals, each rendered into Cinema, Classic Negative and
Velvia styles, with 4,657 training samples and a paper-reported 638 test samples. The targets
were created as film simulations rather than measured physical film scans.
The [official project](https://cxh-research.github.io/FilmNet/) links the
[Kaggle dataset](https://www.kaggle.com/datasets/xuhangc/filmset). Kaggle API
metadata checked on 2026-07-12 reports 11,262,805,356 bytes and labels the
dataset MIT.

The complete distributed archive is now local. Its decompressed tree has 628
files in every test branch, not 638, and 4,657 in every train branch:
`4 × (4,657 + 628) = 21,140`. Runtime manifests and evaluation therefore use
`archive_observed_test_n=628`, while retaining `paper_reported_test_n=638` as
publication provenance. The earlier official Kaggle CLI/member-download probe
continues to establish remote provenance.

Use:

- train as unpaired by destroying pair access inside the training loader;
- form content-diverse pseudo-roll sets from targets sharing one recipe;
- keep all 628 archive-observed aligned test pairs hidden for one final transfer evaluation;
- retain paired supervised FilmNet/LUT models only as upper bounds.

Before weights or examples are released, archive-level terms and attribution
must still be snapshotted; a website metadata label is not a substitute for a
release audit.

#### BlueNeg — primary grouped-real-film pilot

The [BlueNeg dataset card](https://huggingface.co/datasets/ttgroup/blueneg-release/blob/main/README.md)
records 491 35mm negatives, explicit `roll_id`, frame number, date, location,
film type and paired print/pseudo-ground-truth fields where available. Its
current `meta.json` contains 53 roll IDs and 13 recorded film-type strings.
The custom license permits academic and commercial use with the prescribed
image credit.

A metadata-only live audit on 2026-07-12 found that the 8-bit negative-preview
lane is about 688 MB and the 8-bit pseudo-ground-truth lane about 268 MB. These
small lanes are enough for an initial group-identification probe; the hundreds
of gigabytes of 16-bit DNG/TIFF files are not an initial dependency.

This is also **remote-verified**: the Hugging Face API returned the concrete
file tree, and a range request successfully read the first 1,024 bytes of
`negative-preview-8bit/blue-intact/19880215B-26-tama-safari-bogor.preview.png`;
the server reports a 1,707,444-byte object.

BlueNeg is a negative restoration/archive dataset, not a digital-to-film
capture dataset. It can test whether roll grouping contains stable operator
information, but it cannot alone validate the final digital-to-film mapping.

#### Unavailable or insufficient lanes

- The [SillyStill repository](https://github.com/mikasenghaas/sillystill)
  describes same-scene CineStill 800T pairs, but its Zenodo/Hugging Face
  downloads remain marked “Not yet available”; it is not a dependency.
- [Emulating Emulsion](https://musicofmusix.github.io/assets/misc/siggraph_abstract.pdf) fits a compact
  physical model from 3,168 paired colour-chart patches captured over one roll.
  It is an essential nearest baseline and evidence that controlled chart data
  is valuable, but its training data has not been verified as public.
- DPED and FiveK can support generic enhancement/canonicalisation comparisons,
  not film identity.
- [DigitalFilm_dataset](https://huggingface.co/datasets/Richards-Sheehy-sudo/DigitalFilm_dataset)
  is remote-verified and ungated: its five ZIP files total about 3.35GB, a
  1,024-byte range read from the 849,129,648-byte Kodak Gold ZIP succeeded, and
  the card labels the collection MIT. However, its own description says film
  samples were collected from the Internet and does not provide the per-image
  rights/roll/scanner lineage needed here. It is downloadable but remains
  quarantined from confirmatory paper training.
- The PhotoGAN C200 paper describes 1,964 Fuji-C200-labelled and 2,158 digital
  Unsplash images, but no current first-party corpus endpoint was verified. It
  is `published-not-verified`, not available data.

### 2.3 Minimum data gate by claim

| Goal | Minimum evidence |
|---|---|
| no-data method feasibility | synthetic known invertible operators, exact seeds and group-size scaling |
| film-recipe algorithm paper | FilmSet training targets used unpaired; 628 archive-observed aligned pairs hidden for final evaluation; paper-reported 638 retained as provenance |
| grouped real-film mechanism | multiple rights-clear roll groups with frame-level metadata; BlueNeg is the first available pilot |
| named-stock calibrated paper | controlled same-scene digital/film charts and natural scenes; multiple rolls and at least two recorded process/scan sessions; whole-roll holdout |

No current local dataset satisfies the final row.

---

## 3. Nearest work and the surviving gap

The paper must compare against colour-transfer methods, not merely against
artifact metrics.

| Existing work | Occupied claim | Consequence for Roll2Film |
|---|---|---|
| [FilmNet / FilmSet, IJCAI 2023](https://www.ijcai.org/proceedings/2023/0129.pdf) | supervised film-recipe enhancement with multi-frequency processing and LUT refinement | FilmNet is a paired upper bound; FilmSet is pseudo-film, not measured film |
| [Parameterized Color Enhancement](https://arxiv.org/abs/2001.05843) | paired and unpaired global parameterized colour transforms | “unpaired explicit operator” is not novel |
| [D-LUT, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Li_D-LUT_Photorealistic_Style_Transfer_via_Diffusion_Process_WACV_2025_paper.html) and [StatLUT](https://arxiv.org/abs/2607.08227) | reference-conditioned fine-grained or generated LUTs | LUT prediction/generation is not novel |
| [SA-LUT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Gong_SA-LUT_Spatial_Adaptive_4D_Look-Up_Table_for_Photorealistic_Style_Transfer_ICCV_2025_paper.html) | content/style-conditioned spatially adaptive LUT | local/context-aware LUT is not novel and is outside the first method |
| [CanonCGT, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.html) | style-neutral canonical pivot plus supervised and unpaired refinement | canonicalise-then-reapply is not a primary novelty |
| [ChameleonTuner, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Tan_ChameleonTuner_Automatic_ISP_Color_Tuning_in_Subjective_Scenarios_WACV_2026_paper.html) | region correspondences plus multi-objective LUT search for misaligned pairs | region-aware LUT optimisation is occupied |
| [ColorFM, ECCV 2026](https://arxiv.org/abs/2607.07119) | semantic hierarchical colour coupling, instance optimisation, pseudo-pairs and learned flow transfer | semantic coupling, pseudo-pairing and flow matching are fatal novelty threats |
| [Semantic Pseudo-Pairing ISP, 2026](https://arxiv.org/abs/2605.07495) | DINOv2/fused-GW semantic pseudo-pairs for unpaired ISP transfer | semantic OT/pseudo-pairs cannot be the headline |
| [User-Controllable Color Transfer, 2010](https://diglib.eg.org/items/27ff4cfc-1927-4888-875f-fa62246085f2) | constrained parametric transfer designed to avoid visual artifacts | “constrained transfer” alone is old |
| [Emulating Emulsion, SIGGRAPH Posters 2025](https://musicofmusix.github.io/assets/misc/siggraph_abstract.pdf) | 30-parameter physical digital-RAW-to-film-scan model fitted from paired chart patches on one roll | compact physical modelling and “one roll” alone are occupied; the distinction is **unpaired multi-scene repeated observation** |

Also occupied: retrieval/ranking, best-of-`K`, style distributions, diffusion
editing, generic risk control and benchmark construction. They may be baselines
or support tools but not the method claim.

### Surviving provisional novelty statement

> Roll2Film treats a physical roll as a repeated-measure weak-supervision unit
> and jointly identifies a shared, reusable colour operator from multiple
> unpaired, content-diverse frames while explicitly separating restricted
> per-frame nuisance.

The following words are deliberately absent: `first`, `guaranteed safe`,
`accurate Velvia`, and `learns film physics`. They become available only after
the corresponding evidence exists.

---

## 4. Proposed colour-transfer method

### 4.1 Observation model

Let `D = {x_j}` be ordinary digital images in a declared working space. Let
`Y_r = {y_ri}` be frames grouped by physical roll `r`. There is no
`x_j <-> y_ri` correspondence.

The working hierarchical model is

```text
y_ri = A_ri o T_base o delta_r(x_ri)
```

where:

- `T_base` is the shared population/operator prior and `delta_r` is a strongly
  shrunk roll-specific offset; together they define `T_r`;
- `A_ri` is a deliberately small per-frame nuisance family, initially scalar
  exposure plus diagonal white-balance gains;
- `x_ri` is unobserved neutral scene colour.

The factorisation needs a frozen gauge: mean within-roll log exposure is zero,
the geometric mean WB gain is one, and neutral-axis/overall-gain anchors belong
to `T_base`. `A_ri` capacity is preregistered and may not be enlarged after a
failed result. Otherwise nuisance and roll style can absorb one another and the
operator is not identifiable.

Without process/scanner metadata, `T_r` includes stock, development, scan and
batch appearance. It must be called a **roll-look**, not a stock response.

An optional later hierarchy is

```text
T_r = T_process_scan,r o T_stock
```

but it is opened only when multiple rolls per stock and process/scan metadata
make the factors estimable.

### 4.2 Explicit operator family

Use a capacity ladder:

1. `3x3 matrix + three monotone tone splines`;
2. the above plus a low-rank smooth 3D residual;
3. only if necessary, an invertible colour-space coupling/spline operator.

Every survivor must support forward rendering, numerical inversion, Jacobian
evaluation, identity initialisation and export to a dense LUT. Regularisation
covers neutral-axis behaviour, positive/conditioned Jacobian, gamut headroom,
smoothness and bounded local chroma amplification.

These constraints do not constitute the novelty claim. They make the inferred
mapping replayable and reduce interpolation/discontinuity failures.

### 4.3 Neutral conditional density

Learn a frozen neutral-photo density

```text
p0(chroma | luminance, achromatic structure, semantic context)
```

from digital originals. Conditioning is used to avoid interpreting a
sky-heavy target roll and a portrait-heavy source set as a colour-style
difference. Semantic features are frozen and stop-gradient; they never render
pixels.

FilmSet originals are the first appropriate source. Synthetic known operators
can expand colour coverage without pretending to add film truth.

### 4.4 Roll-level inverse identification

For each roll, infer a posterior over the shared operator and restricted
nuisance:

```text
q(theta_r, eta_ri | Y_r)
  proportional to
  p(theta_r) * product_i [
    p0(T_theta_r^-1(A_eta_ri^-1(y_ri)) | u_ri)
    * |det J_(T^-1 o A^-1)|
    * p(eta_ri)
  ]
```

The essential comparison is joint roll inference versus:

- one target image;
- the same number of images with shuffled roll labels;
- a pooled target domain with no groups;
- wrong-roll and same-stock/wrong-roll operators;
- matched semantic-OT and canonical-pivot baselines.

Only after optimisation demonstrates a repeatable group advantage should a
permutation-invariant set encoder amortise inference for speed.

### 4.5 Common-support control

Each roll provides incomplete colour/semantic/exposure coverage. Operator
regions with insufficient shared evidence shrink toward the global prior or
identity rather than extrapolating freely. A held-out query records which
colour cells are supported by the inferred roll.

This control is important for the user's purple-sky failure, but it is not a
claim of semantic safety. A globally smooth operator can still make an entire
sky unattractive. FilmStyleSafe and the product fallback remain external
audits of that residual risk.

### 4.6 Strong style without mean collapse

The primary model estimates a posterior, not a single unconstrained domain
mean. In the first paper this posterior expresses operator uncertainty. A
multimodal style extension is retained only if the data show distinct stable
modes after roll/scanner/exposure nuisance is controlled.

Promotion requires a style floor measured against target roll statistics and
human pairwise judgements. Identity fallback is acceptable for the product,
but high fallback coverage cannot be presented as successful colour transfer.

### 4.7 Why spatial purple blocks are structurally disfavoured

The final image is rendered by one smooth global colour function. It does not
decode or resample spatial features, so it cannot invent a new connected region
independently of input colour. Smoothness, invertibility and gamut constraints
also remove discontinuous LUT cells and banding as degrees of freedom.

This is narrower than a universal artifact guarantee. If later evidence proves
that a spatial residual is necessary, that residual is a separate extension
with a new novelty and safety audit; it is not silently added to Paper 1.

---

## 5. Falsifiable experiment program

### E0 — no-data identifiability simulator

Generate known invertible colour operators and apply them to ordinary images
grouped into pseudo-rolls. Independently vary:

- group size `n in {1, 2, 4, 8, 16, 32}`;
- scene diversity and colour-volume coverage;
- per-frame exposure and WB nuisance;
- source/target content shift;
- operator strength and smoothness;
- missing-support cells.

Primary endpoints are operator-grid error, hidden-image `Delta E`, inverse
likelihood on unseen frames and failure at unsupported colours. The decisive
result is a group-size/coverage curve, not a pretty gallery.

**Stop:** if joint inference does not improve with independent scene coverage,
or nuisance absorbs the true operator, close the grouped-identification
hypothesis before downloading or training a large model.

### E1 — FilmSet paired-blind colour transfer

1. Freeze the official train/test identities and hashes.
2. In the training loader, destroy all original-target pair access and expose
   only domain/style membership.
3. Form content-diverse pseudo-rolls from each recipe target.
4. Keep the 628 archive-observed paired test targets inaccessible until the method and
   hyperparameters are frozen.
5. Evaluate the inferred operator on the original input and compare with the
   hidden recipe target.

Because every FilmSet image of one recipe already shares the same synthetic
look, this experiment tests multi-frame colour-volume coverage, not physical
roll identity. Compare equal-size/equal-semantic-coverage random groups,
pooled groups, shuffled groups and repeated copies of one frame. Physical-roll
information is tested only by E0 known per-roll operators and E2 BlueNeg.

Baselines include Reinhard/Pitie statistics, unpaired parameterised colour
enhancement, D-LUT/reference LUT methods, CanonCGT, ColorFM or its closest
available implementation, semantic pseudo-pairing, and paired FilmNet/LUT upper
bounds.

The primary transfer endpoint is target match after applying the frozen
inferred operator to all 628 hidden digital originals, together with matched
human preference/naturalness. Inverse likelihood and roll classification are
mechanism diagnostics only. Report `Delta E00`/`Delta EITP`, PSNR/SSIM as secondary fidelity measures,
operator smoothness/Jacobian/gamut failures, style strength and matched human
preference. No single metric substitutes for visual transfer quality.

**Stop:** if the method cannot close a preregistered portion of the gap between
strong unpaired baselines and the paired upper bound, make no algorithmic gain
claim.

### E2 — BlueNeg roll-information pilot

Use metadata and the sub-1-GB preview/pseudo-ground-truth lanes first. Perform
leave-one-frame-out group inference and test whether the correct roll explains
its hidden frame better than:

- a random roll;
- a same-film-type wrong roll;
- a date/location/scene-matched wrong roll;
- shuffled roll IDs.

Pre-freeze eligible rolls and match wrong-roll controls on film type,
blue-corrupted/intact partition, date/location, luminance and semantic
composition. Add grayscale-only, metadata-only and ICC/profile shortcut
controls. BlueNeg has small and uneven roll groups; if a matched comparison is
underpowered or unavailable, record `inconclusive/fail`, never weak support.

Separate blue-corrupted and blue-intact partitions and do not reinterpret a
restoration target as a clean digital/film pair.

**Stop:** if the roll advantage disappears after scene/date/film-type controls,
the public real-film lane does not support the central mechanism.

### E3 — real film-inspired transfer

Only after E0-E2 pass, obtain a rights-clean target collection with explicit
roll, owner, process, scanner and license fields. Freeze entire rolls as
holdouts. Evaluate transfer outputs on new digital images with colour-style
matching and naturalness judgements.

**Stop:** if performance is driven by uploader/scanner identity or cannot
generalise across held-out rolls, label the result source-specific and do not
claim film-style transfer.

### E4 — controlled calibration, optional later paper/section

Use colour charts, exposure brackets and natural scenes captured on digital
and film, with at least three rolls and at least two process/scan sessions per
claimed profile. Keep whole rolls hidden.

This stage distinguishes real stock/process fidelity from an attractive
film-inspired look. It is not required to start E0 or E1, but it is required to
use calibrated named-stock language.

---

## 6. Ablations and hostile controls

Required ablations:

1. single frame versus `2/4/8/16/32` frames;
2. correct versus shuffled roll labels;
3. shared operator only versus shared operator plus restricted nuisance;
4. unconditional density versus achromatically conditioned density;
5. no support shrinkage versus support shrinkage;
6. point estimate versus posterior uncertainty;
7. matrix/curves versus smooth residual capacity;
8. correct roll versus same-stock/wrong-roll;
9. style floor off versus on;
10. explicit operator versus a matched pixel generator, with geometry/content
    evidence reported rather than assumed.

Shortcut checks:

- scanner/uploader/date/location prediction from learned operator;
- semantic composition imbalance;
- ICC/profile leakage;
- duplicate and near-duplicate leakage across rolls/splits;
- exposure/WB explaining the claimed style;
- unsupported colour cells receiving large displacement;
- user preference driven only by saturation, contrast, grain or halation.

---

## 7. Paper failure conditions

Any one of the following blocks the primary claim:

| Gate | Failure |
|---|---|
| G0 data | no rights-clear grouped target data and no FilmSet hidden-pair lane |
| G1 identifiability | group size/diversity does not improve known-operator recovery |
| G2 roll information | correct-roll advantage vanishes after matched controls |
| G3 transfer | no meaningful hidden-target gain over strong unpaired colour-transfer baselines |
| G4 style | gain comes from weakening the transform or collapsing to identity |
| G5 artifacts | strong style increases severe chromatic failures versus the eligible explicit-operator baseline |
| G6 factorisation | the method mostly identifies exposure, WB, scanner or uploader rather than a reusable roll look |
| G7 claims | named-stock/calibrated language lacks controlled paired evidence |
| G8 novelty | dedicated search finds prior work already using unpaired roll-set repeated measures for equivalent operator identification |

Failure does **not** reopen a benchmark-paper fallback. Engineering may adopt
the strongest existing method; the research claim closes or is reformulated
around a genuinely new algorithmic question.

---

## 8. Engineering and research are separate lanes

### Engineering product lane

The software may use the best rights-compatible existing method—FilmNet,
CanonCGT, SA-LUT, ColorFM, StatLUT, a supervised LUT, or the current analytic
renderer—after empirical comparison. FARO-style full-resolution audit,
fallback, colour management and deterministic effects remain valuable product
components. Engineering success does not require the method to be novel.

### Research lane

The research lane asks whether roll-group weak supervision identifies a better
reusable transfer operator. Its contribution order is:

1. new supervision/formulation;
2. operator-identification method;
3. identifiability evidence and hostile controls;
4. actual colour-transfer results;
5. artifact evaluation as supporting evidence.

Benchmark construction is not a paper headline.

---

## 9. DRPT execution tree

Parent: `ULT > U5.CT — algorithmic colour transfer`.

| Node | State | Deliverable | Exit evidence |
|---|---|---|---|
| U5.CT0 | complete, re-gated 2026-07-15 | corrected problem, current data audit, nearest-work boundary and claim ladder | paper core remains colour transfer, but Roll2Film requires fixed-budget promotion evidence |
| U5.CT1 | in progress; L2 spline core passes | L0–L2 invertible operator contract and known-operator simulator | explicit L0/gauge/shaper closure remains; L2 inverse/Jacobian/bake and fixed-budget recovery pass |
| U5.CT2 | complete | FilmSet/BlueNeg manifest adapters, licence snapshots, group/split schema | FilmSet passed; BlueNeg exact-revision metadata/whole-roll gate supports only a four-roll Kodak Gold 100-5 matched core |
| U5.CT3 | in progress; affine and L2 fixed-budget controls pass | fixed-total-sample identifiability plus stronger nuisance/prior/scanner factorials | physical-roll information remains unestablished until matched real-roll controls |
| U5.CT4 | complete | FilmSet paired-blind corpus, 4,657 internal identities and frozen 628 lockbox | archive hashes, role isolation, zero cross-pool leakage and no final-payload decoding pass |
| U5.CT5 | complete on internal confirmatory | matched-strength baselines and Roll2Film solver | fixed Lab/pooled-L2 recipe bank passes sampled fidelity/style and all-238 full-resolution severe veto; official 628 remains sealed |
| U5.CT6 | complete; ambiguous | BlueNeg preview roll pilot | one held-out roll loses and one wins; physical-roll information is not established |
| U5.CT7 | stopped by CT6 | optional amortised set inference and complete ablation | not run; no replicated group-information signal to amortise |
| U5.CT8 | ready for non-human one-shot recipe confirmation | one-shot 628 transfer and artifact/style study | frozen fixed recipe bank, confidence intervals and no post-hoc retuning |
| U5.CT9 | future external-data gate | controlled named-stock calibration | whole-roll/process/scan holdout; Level-C claim only |

ChromaticTail/FilmStyleSafe nodes become evaluation dependencies of CT5-CT8.
FARO and FilmCase remain engineering/system baselines. They are not parallel
primary publication routes.

### Change propagation, reintegration and execution mode

This reframe propagates upward to `ULT`, sideways to U4/FARO/FilmCase and
downward to every U5.CT data, method and evaluation leaf. Interface consumers
must use the same claim ladder and the same distinction between local-ready,
remote-verified, published-not-verified and unavailable data. If a data or
novelty gate changes, update affected child experiments, sibling baselines,
README/AGENTS/task board, the reproduction manifest and validation contracts.

Re-integrate bottom-up: verify the changed leaf evidence, then U5.CT, then
supporting U4/FARO/FilmCase contracts, then the `ULT` parent and public-facing
docs. The project agent log records the propagation evidence.

Default write execution is **Mode A**, one integration owner and one writer.
**Mode B** is allowed only for disjoint read-only data/literature/hostile-review
leaves. The current Mode-B evidence bundle has three read-only claims: local
data eligibility, algorithm formulation and novelty threats. Allowed files for
those reviewers are none; all project files are forbidden files for subagent
writes. The root agent is the integration owner. Any conflict or overlapping
claim is resolved against direct file/API evidence before integration. Mode C
requires a separate coordination artifact and is not active here.

### Immediate no-data order

1. freeze `T_theta` API, working space, inversion and Jacobian tests;
2. implement the known-operator pseudo-roll simulator;
3. run the group-size/nuisance identifiability curve;
4. write FilmSet pair-blinding and BlueNeg grouping manifests without images;
5. perform the dedicated grouped-observation novelty audit;
6. request approval for the smallest data acquisition only if E0 passes.

---

## 10. Final decision

The project is not pursuing “a benchmark about film colour.” It is pursuing a
colour-transfer algorithm whose central scientific bet is that roll-level
repeated observations make a reusable film-look operator more identifiable
than isolated unpaired images.

The current computer does not yet contain enough eligible data to train that
paper honestly. It does contain enough code, physical priors and metadata
context to complete the operator contract, simulator, identifiability study,
manifest schemas and novelty gate before any large download. FilmSet and the
sub-1-GB BlueNeg lanes form a concrete, bounded acquisition path if those
no-data gates pass.
