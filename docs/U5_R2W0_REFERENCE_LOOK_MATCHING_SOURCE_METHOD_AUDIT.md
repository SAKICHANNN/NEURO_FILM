# U5.R2W0 Reference-Look Matching Source and Method Audit

Date: 2026-07-27

Decision: **retain a provisional Canonical Factorized Safe Match (CFSM)
research branch; reject NFRM as an established final algorithm or a
film-authenticity claim**

## Executive adjudication

The submitted NFRM/CFSM proposal is useful because it introduces a supervision
structure that the failed marginal-distribution experiments did not have:

> apply one known explicit look to several unrelated contents, then require a
> reference representation or parameter predictor to recover the same look
> across those contents.

That is a legitimate synthetic identifiability experiment and a practical
reference-matching product direction. It does not solve the project's harder
unpaired digital-to-film problem. A single final photograph still entangles
scene content, capture response, exposure, illuminant, prior editing, process
and output interpretation. Without additional observations or assumptions,
those factors cannot be uniquely separated into a true film operator.

The working name is therefore `CFSM`, and only provisionally. `NFRM` suggests
that film identity has already been established, which the current evidence
does not support. Valid output labels remain `reference-look` or
`film-inspired/look-approximation`.

## What the proposal adds

The following ideas are accepted as research requirements:

1. learn or fit from repeated **same-look/different-content** groups rather
   than treating every edited photograph as an independent style;
2. keep source normalization, shared look and optional local adaptation as
   distinct factors;
3. predict or retrieve only an explicit bounded operator or its parameters;
4. render final RGB deterministically through the existing colour engine;
5. require album-level consistency, identity behaviour, OOD abstention and
   severe-artifact gates;
6. compare a single reference with multiple same-look references and an
   explicitly labelled paired synthetic upper bound;
7. open spatially varying residuals only if a global operator is proven
   insufficient.

These points complement the stock-first program. They do not replace
`film_stock_id`, the current source/connectivity gates or the conditional LSM
program.

## Disposition of the remaining submitted architecture

| Submitted component | Disposition in this project |
|---|---|
| `WorkingImageV2`, RAW/HDR/SDR state | Directionally valid but not new W work. Existing U1.2-U1.5 already own ingress, alpha/frame, wide-colour and HDR fail-closed boundaries. Do not create a parallel image contract. |
| CanonCGT-style reference canonicalization | Retain only as a hypothesis. U5.R2F1 already found the raw learned LUTs unsafe; W1 first measures whether simple output-only evidence identifies anything before opening another canonicalizer. |
| ColorFM/HCC semantic correspondences | Already isolated under U5.R2U0/U1. HCC creates pseudo-pairs rather than truth and executes only after S4; semantics may not enter W1's look descriptor. |
| Hist2Style/SA-LUT bilateral or 4D local layer | Deferred. A global explicit operator must first pass and show a localized residual. Direct RGB or unconstrained grid output remains forbidden. |
| StatLUT-style statistic-to-LUT predictor | Prior only. Its random known-LUT supervision supports W1's synthetic design, but unavailable assets and content-sensitive Lab statistics do not justify execution. |
| Multiple candidate generation plus aesthetic ranker | Product-stage possibility only after an identified operator bank exists. Automatic metrics may reject candidates; current A0 evidence cannot certify a preference champion. |
| Album-level shared look | Accepted as an evaluation contract. Shared look parameters are fixed across the album; only bounded normalization/adaptation may vary by image. |
| Grain, halation, dust and scratches | Correctly remain separate FilmFX layers. They may not contaminate colour-operator identification. |
| Mobile/desktop application work | Deferred until an algorithm survives synthetic, real-raster, Style-safe and product gates. It is not evidence for the algorithm. |
| Thirty-person professional blind study | Valid future U8 evidence, but external recruitment and publication are separately authorized actions and not current inputs. |

## The central identifiability correction

The submitted equation canonicalizes each source image but treats the
reference image as if it were already a pure look observation. That is
asymmetric:

```text
submitted:
  final reference R -> look signature z_R
  source X -> source canonicalizer -> shared look

required research model:
  latent content C_r
    -> reference capture/exposure/WB/prior-grade nuisance N_r
    -> unknown shared look F
    -> final reference R
```

From `R` alone, `N_r` and `F` have gauge freedom: many combinations produce
the same final RGB distribution. A source canonicalizer can also absorb the
style it is supposed to preserve. The experiment must therefore keep three
information regimes separate:

- **output-only single reference**: the desired practical problem and the
  weakest information regime;
- **output-only multiple same-look references**: tests whether independent
  contents average away enough nuisance to identify a stable look;
- **paired synthetic reference upper bound**: sees the exact pre-look
  reference observation and measures representation/optimizer headroom; it
  is never presented as an available real-world pair.

If only the paired upper bound succeeds, the practical reference-matching
claim fails. If multiple references succeed but a single reference fails, the
product contract must require multiple references rather than hiding the
information gap with a larger model.

## Style space and content space must remain separate

The proposed CILS list mixes two different tasks. Tone curves, hue-by-luma
responses and bounded operator parameters can describe a look. Skin, sky,
foliage, water, subject/background, mood and intent describe content or
application context.

CFSM therefore uses:

- a **look/operator space** for retrieval or prediction of curves, matrices,
  bounded flow/LUT parameters and confidence;
- a separate **content/applicability space** for common-support checks and,
  only after a look is selected, bounded per-image adaptation.

The same embedding may not simultaneously decide the look and semantic
application. Raw RGB histograms, low-frequency scene colour, CLIP features,
source identity, geometry and scanner fingerprints are forbidden look labels.
This follows directly from SF1.0/SF1.3 and U5.R2S1/S2: content and source
nuisance can be more predictive than the desired signal.

## Why group supervision changes the ML question

The output-only problem is not made identifiable merely by using a larger
encoder. However,
[weakly supervised disentanglement](https://proceedings.mlr.press/v119/locatello20a.html)
shows that observations known to share factors provide qualitatively
different information from an unlabelled image pool, and
[multi-view identifiability](https://proceedings.iclr.cc/paper_files/paper/2024/hash/956e5427549a82a7472e02adc88360e9-Abstract-Conference.html)
formalizes how shared latent information can become recoverable across views
under explicit assumptions.

These results do not prove that a photographic look, still less a film stock,
is identifiable here. They do justify the W1 ordering:

1. generate exact same-look/different-content groups;
2. verify that a fixed descriptor recovers the known operator and rejects
   nuisance;
3. only if the information gate passes but simple prediction is insufficient,
   test a small group-supervised parameter encoder;
4. never treat unlabelled final film scans as if they supplied those groups.

Thus ML is not rejected in principle. The required supervision structure,
not network capacity, is the scarce resource.

## Prior art and retained lessons

### Deep Preset

[Deep Preset (WACV 2021)](https://openaccess.thecvf.com/content/WACV2021/html/Ho_Deep_Preset_Blending_and_Retouching_Photos_With_Color_Style_Transfer_WACV_2021_paper.html)
is direct prior art for learning that unrelated photographs carrying the same
Lightroom preset should share a representation. It used 500 presets, 1,200
Flickr2K images and a positive pair-wise loss. Its explicit 69-setting preset
head is useful evidence for factorized supervision.

The current official repository does not release the generated 600,000-sample
corpus: it estimates the full training data at roughly 500 GB, offers only the
natural-colour inputs, and instructs users to reproduce the remaining styles
through Lightroom scripting. Code and trained models are non-commercial
research only. Deep Preset is therefore method precedent, not a bounded
rights-ready data acquisition or reusable teacher.

Its own supplementary evidence also exposes the trade-off: enforcing the
shared preset representation improves stability but can hurt exact preset
prediction, while the unrestricted RGB generator gives stronger visual
results. The direct generator is forbidden here. The published training set
is not a complete, rights-cleared project asset and the official repository
states non-commercial research use.

### NTIRE 2026 retouching-transfer challenge

The official
[NTIRE 2026 challenge report](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Elezabi_Photography_Retouching_Transfer_NTIRE_2026_Challenge_Report_CVPRW_2026_paper.html)
tests a stronger information regime than the submitted output-only proposal:
the reference contains the same photograph both before and after editing.
Its hidden development and automatic test sets use 18 and 12 new presets,
respectively. This is valuable evidence that preset-held-out, content-held-out
retouch transfer is experimentally measurable.

The strongest directly relevant submission, SIREN-VA, computes a
57-dimensional descriptor from the aligned before/after RGB residual,
retrieves preset-specific initializations and then fine-tunes on the query
pair. This closely matches the user's intuition of retrieving a previously
seen edit, but it also demonstrates why the input regime must be stated:
aligned residuals expose the edit far more directly than a styled image alone.
The submission averages the top three neural checkpoints and produces RGB
through an INR; both choices are forbidden as a Style-safe design here.
Retain instead:

- paired residual descriptors as an explicit information upper bound;
- held-out-preset and held-out-content evaluation;
- hard Top-1 bounded-operator retrieval before any sparse mixture;
- test-time fitting only of an explicit safe operator.

All seven final systems are INR/test-time-optimization variants, often using
coordinates, RGB residuals or pixel-space ensembles. Their leaderboard is
useful product evidence but does not validate output-only CFSM, a film
operator, or this project's topology contract.

### InstantRetouch

[InstantRetouch (2026)](https://arxiv.org/abs/2602.17044)
is the closest published architecture to the desired
look-space/content-space separation. It encodes style from aligned
before/after pairs, then retrieves references by query-content similarity and
aggregates only the retrieved style latents. Its ablation reports that using
all references is worse than a small relevant subset. This supports:

- infer or select the look before content-aware application;
- keep content retrieval inside the already selected look/user/stock;
- prefer hard or sparse retrieval over a dense average that washes out style;
- hold out both content and preset families.

It does not solve the project's output-only problem. Its normal input is a
paired edit; its unpaired-style mode creates a pseudo pair from semantically
different images. The decoder is a sigmoid-ended per-pixel conditional MLP
that directly predicts RGB without a cube, positive-Jacobian, replay or
identity-strength guarantee. Its SigLIP content retrieval may be useful only
in the separate application space after look selection, never as look or
stock evidence. The LAION/preset VCIRB construction also requires an
independent asset and rights audit before any use.

### Neural Preset

[Neural Preset (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Ke_Neural_Preset_for_Color_Style_Transfer_CVPR_2023_paper.html)
provides a closer structural precedent: normalization and stylization are
separated, and a small network predicts parameters for a deterministic neural
colour mapping. Its self-supervision uses synthetic LUT/filter
perturbations, which supports the proposed generated-operator pilot.

It does not provide this project's structural guarantees. The reported DNCM
parameterization is not a positive-Jacobian cube-preserving flow, and the
paper reports compression-artifact amplification and colour-gamut/content
mismatch limitations. The project may reuse the supervision idea, not the
unsafe operator or its claims.

### Hist2Style

[Hist2Style (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Galor_Hist2Style_Histogram-Guided_Stylization_with_Bilateral_Grids_CVPR_2026_paper.html)
is strong product evidence that a small model can compile a global colour
descriptor into a low-resolution bilateral-grid render. The paper reports
1.5M parameters, a `8 x 16 x 16` bilateral grid carrying `3 x 4` affine
coefficients plus alpha, and separate monotone curves. Its 31-expert,
3,000-valid-trial study reports Hist2Style preference rates of 82.57% against
SA-LUT, 73.75% against IDT, 73.24% against Xia, 72.58% against WCT2, 70.59%
against D-LUT and 61.62% against PhotoWCT2. The paper-scale examples are
visibly stylized without apparent geometry corruption, although several make
aggressive semantic hue changes. This is product evidence, not a clean teacher
for this project:

- six or seven styles per Unsplash Lite image produced about 1.7M FLUX edits,
  filtered to 1.1M examples; prompts also have generated lineage. This violates
  the current non-generative training-lineage boundary;
- its marginal histogram conditioning remains content-ambiguous;
- the supplement says that using a different same-style image's histogram
  struggles to match colours accurately;
- its high-resolution path can show grid artifacts.

The pinned public repository contains only README/static material at commit
`e59808d49cce4f2fc31ed505dc190923251904d1`, says code is awaiting approval
and has no root licence. Do not download weights or reproduce the generative
dataset. Retain only the global-first/local-later structure, explicit
affine-grid/curve decomposition, strong-versus-bland visual-evaluation lesson
and requirement to test same-look references on unrelated content.

### Emulating Emulsion

The public full paper for [Emulating Emulsion (SIGGRAPH 2025)](https://dl.acm.org/doi/10.1145/3721238.3730707)
is the strongest current **paired physical operator and acquisition-design
prior**, but not an output-only reference solution. It fits
`R_scan = M2 f(M1 R_digital)`: two `3 x 3` matrices around three independent
four-parameter sigmoid curves in log-exposure, for 30 parameters total.

The published experiment uses one 36-exposure Velvia 100 slide roll, three
illuminants near 3300/5500/7700 K, 11 one-EV-spaced exposures per illuminant,
synchronised Fujifilm X-Pro3 and Nikon F2AS captures, an X-Rite Digital SG
chart and a D50 1000 cd/m2 scan backlight. Excluding repeated chart-border
patches yields 3,168 unique digital/scan patch correspondences. Five-fold
reported RMSE is `.0491` for no model, `.0111` for a LUT, `.0133` for
matrix-plus-curves, `.0116` for the full two-matrix model and `.0123` for the
inverse-matrix variant. Paper-scale scenes show strong, coherent green,
warmth and contrast changes without apparent geometry corruption.

The evidence ceiling remains strict:

- one roll, one slide stock and only two general-scene film exposures provide
  no independent roll/process/source/scan replication;
- the paper publishes aggregate measurements and figures, but no raw patch
  table, fitted parameters, executable code or reusable data licence;
- the matrices and sigmoids are fitted by unconstrained SciPy least squares
  and do not establish this project's range, Jacobian, inverse or replay
  guarantees;
- the positive-film formulation does not identify a colour-negative orange
  base, inversion or print chain;
- this paired chart protocol cannot identify an operator from unrelated
  output-only photographs.

The existing U5.R2J0/J1 clean-room experiments tested only the publication's
functional family with original synthetic witnesses. They did not use the
publication's fitted parameters or measurements and therefore do not refute
the real fitted Velvia model. J1 remains closed because its synthetic
automatic style did not create visual product value over R2E1/safe-rich.
Retain the paper as a future owner-cleared calibration family and as evidence
for factorized explicit operators, not as current executable/training data or
stock truth.

### Color Transfer with Modulated Flows

[Color Transfer with Modulated Flows (AAAI 2025)](https://arxiv.org/abs/2503.19062)
uses an EfficientNet colour encoder to modulate separate content and style
Neural ODE flows through a uniform latent distribution. Its pointwise RGB
transport is a useful distribution-matching challenger, but marginal
distribution correspondence is not same-known-operator truth. The paper
explicitly reports artifacts and unintended colour replacement at stronger
transfer/fewer integration steps; the released output conversion clips to
`[0,1]`, which can conceal out-of-range behaviour rather than proving an
in-range map.

The pinned implementation at
`e1884a0208160a00e0d48da7cc25e804b92772be` has no root licence and provides
neither this project's cube/Jacobian/inverse gates nor stock evidence. No
checkpoint run opens. A later challenger may distil a learned flow into a
validated explicit grid only if the simpler W1 descriptor/retrieval route
fails and a separate rights/lineage contract passes.

### PPR10K and group consistency

[PPR10K (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html)
contains 11,161 RAW portraits in 1,681 groups, with three independently
retouched targets per photograph and explicit group-level-consistency
evaluation. It is useful future evidence for album consistency, portrait
safety and content-adaptive application after a reference look has already
been identified.

It is not same-known-operator truth: each expert edits each group rather than
applying one recorded global recipe across unrelated content. The official
dataset is 406 GB, restricts images and derived data to non-commercial
research, and is therefore neither a bounded ready acquisition nor a source
of film/stock evidence. Its 130 MB XMP partitions are potentially valuable
metadata, but opening them still requires a separate access, licence,
lineage and byte-budget child. Do not use expert identity or group membership
as a surrogate look label.

### MMArt-PPR10K explicit-parameter subset

The public
[MMArt-PPR10K dataset card](https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k)
exposes a smaller derivative of PPR10K with `before.jpg`, `processed.jpg`,
Lightroom XMP/Lua configurations and generated instruction/chain-of-thought
text. A metadata-only repository audit on 2026-07-27 found 4,055 XMP recipes
over 1,412 base content identifiers; the median content has three retained
recipes. One public configuration confirms that the target is factorized into
explicit white balance, exposure, tone, HSL, calibration, curve, texture and
sharpening parameters rather than an opaque final-RGB label.

This is useful evidence for a later **explicit parameter-manifold** control:

- separate basic normalization from hue/curve/detail decisions;
- measure within-group parameter consistency without interpreting a person,
  group or instruction suffix as a film mode;
- compare a bounded parameter predictor with a directly fitted raster
  operator;
- test whether sparse content retrieval improves parameter selection without
  averaging incompatible styles.

It is not a ready dataset leaf. The mirror declares Apache-2.0, while the
[official PPR10K repository](https://github.com/csjliang/PPR10K) restricts
images and derived data to non-commercial research and prohibits exploitation
of derived data. The derivative README does not reconcile that upstream
restriction, and its generated language is AI-produced supervision rather
than observed expert rationale. Therefore only public metadata and one
configuration sample were inspected: no image payload, bulk recipe download,
training, operator fitting, film claim or redistribution is allowed. A future
child would first need owner-approved lineage/rights adjudication and an exact
mapping from derivative rows to PPR10K expert/group identities.

### Gelatin Labs structured film-scan provenance

The current official
[Gelatin Labs Content Credentials page](https://techweek.gelatinlabs.com/)
describes unusually strong per-frame provenance: roll number, film stock,
processing, scanner, scan style (`GelForm`), bit depth, frame/format and
capture date are embedded with a signed C2PA chain. It also reports 11 scan
styles and Noritsu/Fujifilm Frontier scanning. In principle, a consented
export with repeated rolls, stocks, scanners and GelForms would have much
better nuisance connectivity than the project's current community pools.

This is not an available dataset. Customer scans are private, no research
export/API/licence is offered, and the same official page explicitly marks
the files against model training, inference and data mining. The public
community page must therefore not be crawled, sampled or treated as a source
manifest. Retain this only as a future owner-consented or written-permission
source design; external outreach is outside the current Goal, and no pixel,
metadata acquisition, stock learning or operator fitting opens.

### Results WILL Vary controlled-source design

The public Results WILL Vary description is valuable only as a future
acquisition pattern: repeated same-scene colour-checker captures, exposure
brackets and recorded developer/scan metadata across many stocks would create
the connectivity that community output photographs lack. The current site
also states that displayed images are edited and are not a scientific
comparison; its technical companion and raw controlled data are not publicly
available, and no reusable training licence is supplied. The film-box
placeholders are explicitly AI-generated. Do not download images, infer stock
operators or initiate external outreach. Retain the proposed capture design,
not its displayed pixels or claims.

### Rejected opaque preset derivative

The `Phitran21/adaptive-photo-retouching-6style` dataset card is not an
acceptable shortcut. It is a 4.54 GB gated-auto derivative whose licence field
requires separate disclosure/contact, and the card describes MiniMax M3
generated per-image recipes rather than six fixed recorded presets. No payload
was requested. It cannot provide same-known-look truth, film evidence or a
rights-clear benchmark.

### SA-LUT and PST50

[SA-LUT (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Gong_SA-LUT_Spatial_Adaptive_4D_Look-Up_Table_for_Photorealistic_Style_Transfer_ICCV_2025_paper.html)
is useful engineering prior for a content-dependent context coordinate plus a
4D LUT and for full-resolution photorealistic-style evaluation. Its PST50
benchmark supplies reference/content judgments, not repeated known-operator
labels. The cross-attention context generator also does not provide this
project's content-invariance, topology or deterministic safety guarantees.
Retain it only as a later local-residual/product challenger after a global
operator leaves a stable spatial residual; it cannot validate W1
identifiability.

### CanonCGT

[CanonCGT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.html)
is the closest published architecture to the submitted factorization. It
predicts one image-adaptive `17^3` LUT to map an input into a canonical pivot,
then a second image-adaptive `17^3` LUT conditioned on the reference grade.
This is strong evidence that canonicalization and reference-driven grading
should be tested as separate stages rather than collapsed into one opaque
mapping.

It does not resolve the project's identification problem:

- the canonical target is MIT-Adobe FiveK Expert C, a selected editing
  convention rather than a measured neutral scene or film-independent truth;
- supervised training uses FiveK plus 56 Lightroom preset styles, and the
  later self-supervised protocol applies known colour perturbations to
  different crops of the **same** photograph. Neither setting identifies an
  operator from unrelated final photographs;
- the reference embedding and both LUT generators use MobileNet/spatial image
  features. The resulting transform is global per image but explicitly
  content-conditioned, so content substitution and same-look/unrelated-content
  invariance remain mandatory;
- the released implementation returns `identity + residual` LUT values
  directly. It has no exact output-range, positive-Jacobian, spectral-norm,
  inverse, coefficient or replay constraint. Evaluation-side clamping is not
  a renderer safety proof;
- its paper evaluates reconstructed synthetic perturbations and reference
  statistics. It does not provide film-stock evidence, a severe-artifact
  veto, unpaired operator truth or a calibrated colour-state contract.

This source and its deployment branch are already frozen project evidence,
not a newly opened candidate. U5.R2F0 pinned the official Apache-2.0 repository
at commit `229a7d3ec24af19ea5d97bb136f0317762bb9261`, its 20.9 MB E2E checkpoint
and the unresolved two-byte `SSL.pth` pointer. U5.R2F1 then ran the E2E model
over the fixed nine-reference by nine-gold bank:

- reference-conditioned pairwise output Delta E76 was `8.5322`, proving that
  the model did not merely average every reference;
- six of nine policies passed both style and non-basic floors;
- zero of nine passed clipping/range, with raw-final out-of-range fractions
  reaching `12.2397%`.

U5.R2F2 found that node clipping remained structurally unsafe, while uniform
safe contractions passed structure but reduced every policy below the style
and non-basic floors. U5.R2G0/G1 retained much of the reference-conditioned
style in a constrained explicit operator but still produced no complete
severe-first survivor; the CanonCGT deployment/distillation branch is closed.
These results and their frozen gates must not be reopened or rewritten.

CanonCGT remains useful only as prior for a **scientifically different**
question introduced by W1: whether same-known-look observations over
unrelated contents expose operator information under explicit content and
nuisance controls. If W1 passes, a new child may study source-only
canonicalization or compare fixed statistical descriptors with a frozen grade
descriptor, but it must use new preregistered groups, reserve headroom beyond
the evaluator epsilon by construction, and output an already admissible
bounded operator. It may not rerun the F1 reference bank, rescue F2/G1, tune
their thresholds or imply film authenticity.

Any later positive result could establish reference-look transfer or a useful
canonicalization prior. It still could not be called a real-film operator,
calibrated stock response or solution to unpaired digital-to-film
identification.

### StatLUT

[StatLUT (arXiv 2026)](https://arxiv.org/abs/2607.08227) is unusually close
to the admissible ML boundary. Its image-reference path discards spatial
features and represents both content and style with a one-dimensional Lab
luminance histogram, a two-dimensional `ab` histogram and a colour-conditioned
mean-luminance map. A compact attention mapper predicts a `16^3` residual 3D
LUT rather than final image pixels. The paper's patch-shuffling control is a
useful semantic-leakage test, and its `L`, `ab`, `L|ab` factorization is a
stronger fixed-descriptor candidate than raw marginal RGB histograms.

This does not establish the submitted output-only algorithm. StatLUT's mapper
uses **both** the content-image statistics and the reference-image statistics;
its self-supervision applies random LUTs to content images and derives style
references from those transformed images. It therefore has known synthetic
operator truth during training, while a lone final film scan still does not.
The published random-LUT corpus is not film evidence. Its clamped residual LUT
with monotonicity and total-variation penalties is also not an exact
cube-preserving guarantee: penalties can fail, and clamping can conceal folds
or boundary saturation.

Retain only the following conditional challenger:

- after W1, compare the frozen factorized descriptor with a non-learned
  `L`/`ab`/`L|ab` descriptor under the same unseen-operator, unseen-content and
  nuisance probes;
- if fixed descriptors expose sufficient information but a linear head is
  inadequate, allow a small parameter-only mapper to predict the existing
  bounded O0 flow or a separately projected safe LUT;
- add patch-shuffling, content substitution, identity-reference and
  source/reference-statistic ablations;
- keep the paper's text-conditioned H-Diffuser outside FilmCase and all
  Style-safe training;
- do not promote the candidate without exact range, Jacobian, inverse, replay,
  held-out operator and severe-artifact gates.

No code, weights or dataset release is linked from the current v1 preprint, so
StatLUT is method evidence rather than a ready external asset.

### Distribution-conditioned transport

[Distribution-Conditioned Transport (2026)](https://arxiv.org/abs/2603.04736)
is relevant architecture prior for conditioning a transport model on
distribution embeddings. The paper explicitly distinguishes distribution
matching from within-distribution coupling: a target distribution may be
matched while source correspondence is ignored. This reinforces rather than
removes the S3/S4 operator-identification gates.

### Capture One and other submitted methods

[Capture One Match Look](https://support.captureone.com/hc/en-us/articles/22188770298269-Match-Look-Tool)
supports the product decomposition into normalization, light/contrast, colour
adjustment and colour tone, but is a black-box baseline rather than a
scientific teacher.

ColorFM/HCC remains the separately audited U5.R2U branch. CanonCGT remains a
normalization prior with unresolved asset/rights and operator-safety issues.
StatLUT remains unavailable as an eligible asset. D-LUT and diffusion-derived
video-LUT teachers are incompatible with the no-generative lineage boundary.

## Existing project components to reuse

No new `src/color_match/` package is justified yet. The first pilot should
reuse:

- U5.R2O0's cube-preserving stationary velocity flow and exact renderer;
- generated content/style utilities from U5.R2S2-S4;
- the current range, Jacobian, inverse, replay and style-retention gates;
- WorkingImage and current fail-closed colour-state boundaries;
- existing full/tile and severe-artifact infrastructure if a pixel candidate
  is eventually opened.

Creating a parallel renderer, `WorkingImageV2`, recipe schema or safety stack
before the information experiment passes would duplicate mature project
boundaries and make the result harder to audit.

## First admissible experiment

U5.R2W1D is a generated, rights-free reference-look identifiability pilot. It
uses known bounded O0 operators applied across independent generated content
groups. It compares:

1. identity/global-mean operator;
2. raw histogram nearest-neighbour negative control;
3. fixed look descriptors plus ridge regression;
4. seen-look hard retrieval;
5. output-only single-reference inference;
6. output-only multi-reference inference;
7. paired-reference oracle upper bound.

All methods must output one bounded explicit operator. The hidden Hald grid or
operator coefficients are used only for labels/evaluation in this synthetic
pilot and never made available to an output-only query.

The `53/55/56` owner anchors form a required strength-path negative control:
they are one Velvia direction at strengths `.72`, `.50` and `.58`, not three
independent looks. A valid representation must preserve their strength order
while treating them as one operator direction. `09` remains a weaker
same-family control, and the externally observed red-speckle/posterization
case remains in future pixel-level regression.

## Branches

- only seen-look retrieval succeeds: retain a bounded case-bank retriever;
- output-only single reference fails but multi-reference succeeds: require
  multiple references and reject single-reference claims;
- unseen-look parameter regression succeeds: open a small parameter-only
  predictor challenger;
- only paired-reference upper bound succeeds: close practical inference as
  unidentified;
- content or nuisance predicts the look signature: close without capacity
  rescue;
- a simple deterministic descriptor/retriever wins: do not add a neural
  model;
- any structural or severe-artifact gate fails: reject the candidate;
- synthetic success: retain mechanism evidence only; real professional or
  film data still needs an independent rights, lineage and claim gate.

## Evidence retained locally

The following ignored source-recon artifacts are research records, not
redistributable project assets:

| Source | Local evidence | SHA-256 |
|---|---|---|
| Hist2Style arXiv v1 paper | `outputs/source_recon/hist2style_cvpr2026/paper_arxiv_v1.pdf` | `4DB6C2DE06FBD42129057623B257B3459A649A435DE0D47FE2B823E28AF2DAB3` |
| Hist2Style supplement | `outputs/source_recon/hist2style_cvpr2026/supplemental.zip` | `531AD04268332801186F3114237F199E7E4032FEAD5648F5537D3A8A109205DD` |
| Hist2Style public source | `outputs/source_recon/hist2style_cvpr2026/source` at commit `e59808d49cce4f2fc31ed505dc190923251904d1` | no root licence; README/static release only |
| Emulating Emulsion abstract | `outputs/source_recon/emulating_emulsion_siggraph2025/siggraph_abstract.pdf` | `2A1689C9ABFC610D16BBD164228E68D7BCAB4AC6C2E9BE0C5636442DFEAA81FF` |
| Emulating Emulsion full paper | `outputs/source_recon/emulating_emulsion_siggraph2025/full_paper.pdf` | `02CE6345ED08CB61B70A7089CC488B5F7D7B6809DB2E51CE0C4410F0606A5D47` |
| Modulated Flows paper | `outputs/source_recon/modflows_aaai2025/paper_aaai2025.pdf` | `22E8A643505B5599B71D4AB2E43BFCC3158FFC7A32D811BD85606C5494CC330C` |
| Modulated Flows public source | `outputs/source_recon/modflows_aaai2025` at commit `e1884a0208160a00e0d48da7cc25e804b92772be` | no root licence |
| Deep Preset paper | `outputs/source_recon/deep_preset_wacv2021/paper.pdf` | `ED69E1863C69EA340B1B7C82E5811DE4991F506BA2983067CDEE3A439AF08A35` |
| Deep Preset supplement | `outputs/source_recon/deep_preset_wacv2021/supplemental.pdf` | `21844C5BBEFDE82707145C9DDF86C97A8696C36691F8C1BE6B0D1F9A533353A6` |
| NTIRE 2026 retouch-transfer report | `outputs/source_recon/ntire2026_retouch_transfer/paper.pdf` | `9C0E31B36E77A6CC4EFDCCE12DD1DEF2ADAEFFA377D6D12B226314EA600BDAF6` |
| InstantRetouch paper | `outputs/source_recon/instantretouch_2026/paper.pdf` | `32DBC46190C3F67E9DC3C25F442FD21BF6ED8A7238B18A500388DF5AABEC394B` |
| Neural Preset paper | `outputs/source_recon/neural_preset_cvpr2023/paper.pdf` | `5A6AD5F16A308AF055EE1CE8497E124F710C1A81967CE945D3FB72B2AA2ED376` |
| Neural Preset supplement | `outputs/source_recon/neural_preset_cvpr2023/supplemental.pdf` | `CFB5553DB153C3E5433ACC74807D175D44ACAC06DAFEBD24A130B64D53FAC9B7` |
| CanonCGT CVPR 2026 paper | `outputs/source_recon/canoncgt_cvpr2026/paper_cvpr2026.pdf` | `5DE5A83241F5EC4251248A7B5E565F616CF15D40DE6FB6B1C842F15DDDEF9811` |
| CanonCGT official source | `outputs/source_recon/canoncgt_cvpr2026` at commit `229a7d3ec24af19ea5d97bb136f0317762bb9261` | licence `1EB85FC97224598DAD1852B5D6483BBCF0AA8608790DCC657A5A2A761AE9C8C6`; E2E weight `916F7AD5028D3FEF51CF915BB51FEBC9508E7A378C0BABE9FF3D3992B8C594F7` |
| StatLUT arXiv v1 paper | `outputs/source_recon/statlut_arxiv2026/paper_v1.pdf` | `A722F9869AB99824855A5A8465DA4730A7FEAC701DDDBA5E98198EA3C8AED9DB` |
| Distribution-Conditioned Transport paper | `outputs/source_recon/distribution_conditioned_transport_2026/paper.pdf` | `66613EEE32EBA7F19741E2F97443BC60136CB5A99CA50C4F134431D0418C7D19` |
| MMArt-PPR10K public README | `outputs/source_recon/mmart_ppr10k_readme.md` | `00E81B37E54F3A0D818CA0E49B192399C7589323398A42F7D19A024E3C8F703E` |
| MMArt-PPR10K one public XMP sample | `outputs/source_recon/mmart_ppr10k_public_metadata/config.xmp` | `78EC5860C5F458793EDA51F44D9EC4594909D74310C94DC72017DCA6C7077D75` |
| MMArt-PPR10K one public Lua sample | `outputs/source_recon/mmart_ppr10k_public_metadata/config.lua` | `E479B4A420D72484A65FA2D4726B7BCFE940444F559964B78DA1D43F6DFF30BD` |

## Conditional real-pixel bridge: INRetouch RTD

The WACV 2026 INRetouch release supplies the first discovered external dataset
whose topology directly matches the accepted same-known-look/different-content
question. Its official
[dataset card](https://huggingface.co/datasets/omaralezaby/Retouch_Transfer_Dataset)
states that 167 Lightroom presets were applied to 569 selected MIT-Adobe FiveK
images. It separates 508/61 contents and 145/22 presets for development and
benchmark use. Each preset directory repeats one named preset across many
different contents, while a `natural` directory preserves the corresponding
pre-preset image. This is substantially stronger supervision than final
reference photographs and can test:

- output-only single- and multi-reference identification;
- paired-before/after explicit-operator fitting as a real-raster control;
- held-out content and held-out preset generalization;
- whether a recovered operator is only a preset/content shortcut.

It is not real film, stock evidence or professional per-image grading. It is a
batch-rendered Lightroom-preset control derived from MIT-Adobe FiveK.

`preset_id` is also not automatically a global-operator label. The INRetouch
paper explicitly reports that the same preset can modify different images in
different ways and motivates coordinate/local-context conditioning on that
basis. Although geometry- and portrait-specific presets were excluded, the
remaining recipes may still contain adaptive or spatial adjustments. RTD must
therefore begin with a within-preset global-explainability audit:

1. fit each natural/styled pair independently with the bounded global O0
   operator;
2. compare per-image operators inside a preset;
3. fit one shared operator on development contents and evaluate held-out
   contents;
4. measure aligned spatial residual and test whether a global operator is an
   adequate truth class.

Only globally coherent presets may test W1's global reference mechanism.
Adaptive/local presets form a separate stress/output-profile class. They
cannot label one global prediction as wrong or open a local network before
the global residual is measured.

A read-only 2026-07-27 API audit found repository revision
`3e100e1fa896d9ed023cd1545890400edd67f949`, 96,164 listed files and the
following topology:

| Partition | Listed files |
|---|---:|
| Train | 74,168 |
| Validation | 8,906 |
| Benchmark/Test | 1,403 |
| Benchmark/Test_References | 11,684 |

An exact public-filename join against the local 128-identity/903-file FiveK
`freeze_v1` found 15 unique shared source identities. They occupy paired RTD
roles: 13 Train and the same 13 Test_References, plus two Validation and the
same two Benchmark/Test. This confirms the source-ID namespace but does not
provide RTD pixels. Existing local RAW, Expert-C and project-derived outputs
are different representations and cannot substitute for the preset renders.

The ignored public API snapshot is 6,293,126 bytes at SHA-256
`3ACFAC6F57EC469E04670DEE500E64050D51FF401D3E43998FEDA7AA1A407E2F`.
It contains paths and repository metadata only, not gated image payloads.

The hosting page reports approximately 18 GB. Access is gated and requires
sharing account contact information and accepting CC BY-NC-SA 4.0 terms.
Consequently:

- no pixel or gated file was accessed in this audit;
- no automatic acceptance, contact disclosure or download is authorized by
  the read-only source review;
- project use remains research-only and conditional on a separate human
  click-through decision, exact manifest/hash audit, MIT5K lineage
  reconciliation and non-commercial/share-alike isolation;
- it cannot train or ship production weights under the current unresolved
  project-release licence;
- if admitted later, start with the bounded benchmark partition rather than
  the 18 GB corpus, and freeze content/preset partitions before viewing
  results.

This source is therefore the highest-value conditional W2 real-raster bridge,
not an immediately ready data leaf.

The public method code was separately acquired for source inspection at exact
commit `cbf0db19487222c21d63116c320357764438360d`; its licence file SHA-256 is
`0FAFE8C0D5DF2A65B16E99D27DD658E0626F5CCE346ACECE07F8969716591E97`.
The official WACV paper was pinned at SHA-256
`C4AC0F73B70F829EEE0683D78B40FDD7E67279C59286EE21F0ACA433EF9A2823`.
The released network takes absolute image coordinates plus source RGB and
directly predicts output RGB through sine-activated pointwise/depthwise
convolutions, trained per reference pair with an L1 pixel loss. This explains
its ability to model local edits, but it violates this project's final-RGB
rule and has no positive-Jacobian, cube/range, strength-zero or explicit
operator guarantee. The code may be retained as a research-only comparison;
its renderer and weights are not admissible CFSM components. The dataset
topology is useful independently of the method.

## Claim ceiling

CFSM is presently a **candidate reference-look mechanism**. It has not learned
a real film stock, recovered a true photographic grading procedure, solved
unpaired operator identification, passed real-pixel Style-safe evaluation or
become the project's final algorithm.
