# U5.R2H0 film-specific spectral/sensitometric source audit

Date: 2026-07-24  
Node: `ULT > U5 > U5.R2 > U5.R2H0`  
Decision: **one Velvia 50 datasheet-only witness is feasible; a calibrated or
identified stock operator is not**

## Question and result

This audit asks whether a clean-room, non-generative film simulation can be
built from inputs that this project can actually obtain and reproduce. It
does not ask whether an attractive transform can be fitted to the current
unpaired film pixels; those pixels remain closed for fitting, training and
latent-mode discovery.

The narrow answer is positive for one experiment. The official Fujichrome
Velvia 50 bulletin publishes all three graph families needed for an
approximate direct-positive spectral witness:

1. three layer characteristic curves;
2. three spectral-sensitivity curves;
3. separated yellow, magenta and cyan spectral dye-density curves.

The current Kodak Portra 400 and Ektar 100 sheets publish characteristic and
spectral-sensitivity curves, but their spectral-density panel contains only a
midscale-neutral total curve and a minimum-density curve. It does not expose
independent CMY dye bases. Those sheets therefore cannot identify a complete
negative spectral transmittance, still less a negative-to-print operator.

## Reproducible local source inventory

All manufacturer material is `prior_only`; no PDF is an image target or
permission to claim stock calibration.

| Source | Local evidence | What it supplies | What remains missing |
|---|---|---|---|
| [Fujichrome Velvia 50 bulletin](https://asset.fujifilm.com/master/emea/files/2020-10/a71dda63e2662f012b3b74110794918a/films_velvia-50_datasheet_01.pdf) | `data/physics/fujifilm_velvia_50/product_information_bulletin.pdf`; 524,708 bytes; SHA-256 `bebfb4305fd3edc5e3505a9a0f4a0a1078da1db7672aacb4aa1bac1de3d64cfd` | characteristic, sensitivity and separated Y/M/C dye-density graphs | numerical tables, batch/process state, scanner/projector response, scene spectrum |
| [Kodak Portra 400 E-4050](https://www.kodakprofessional.com/sites/default/files/wysiwyg/pro/resources/e4050_portra_400.pdf) | `data/physics/kodak_portra_400/technical_data.pdf`; 262,115 bytes; SHA-256 `e83ac6775d37832a4cb466892a3e1cf4c88917a6ee93384e59d6924b1cd97e3a` | characteristic and layer sensitivity; total midscale/D-min spectrum | separated CMY dye bases, mask/coupler response, print chain |
| [Kodak Ektar 100 E-4046](https://www.kodakprofessional.com/sites/default/files/wysiwyg/pro/resources/e4046_ektar_100.pdf) | `data/physics/kodak_ektar_100/technical_data.pdf`; 276,841 bytes; SHA-256 `e39971d16dc8679fb92742a229c237acde5b0dddb0428a708f8ad68d7a1d5590` | characteristic and layer sensitivity; total midscale/D-min spectrum | separated CMY dye bases, mask/coupler response, print chain |
| CIE 1931 2-degree CMFs and D50/D65/A illuminants | official CIE CSVs under `data/calibration/cie` with companion metadata | reproducible observer and viewing/exposure illuminants | actual scene spectrum and capture-camera spectral state |

The local `data/physics` tree contains all 13 PDFs in the scripted
manufacturer scope. Camera spectral-sensitivity databases also exist, but a
display-referred sRGB photograph does not identify its original camera RAW,
illuminant or scene reflectance. Those camera curves are therefore forbidden
as per-image truth in this experiment.

## Primary method evidence

Fujifilm's published print-film simulation describes the physically relevant
factorization: integrate negative transmittance against print-film spectral
sensitivity, map printing exposure to analytical dye amount, then compose CMY
dye spectra and base transmittance. The paper reports an internal average
colour difference below 2 for its measured system, but it does not publish
the complete numerical sensitivity, ADA, dye and base profiles needed to
reproduce that result. It validates the factorization, not a downloadable
calibrated profile: [Fujifilm R&D No. 56, print-film simulation](https://asset.fujifilm.com/www/jp/files/2019-12/c89517b7f58934196d65288f0c4b86cc/ff_rd056_009_en.pdf).

The 2025 paper [From Negatives to Positives: Modeling the Photochemical
Printing Process](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/archiving/22/1/33)
is useful as an independent warning: missing enlarger filtration, paper
spectral response and nonlinear exposure-to-density behaviour require
approximations. A datasheet-only negative-to-print chain is consequently not
identified.

Input RGB is another independent ambiguity. A tristimulus value has many
spectral metamers, and a film layer can distinguish spectra that match under
the CIE observer. Two primary methods demonstrate valid but non-unique
canonicalizations:

- [Mallett and Yuksel 2019](https://diglib.eg.org/items/bbffa865-e99c-4c1f-bd33-70102dc8af78)
  constructs three sRGB reflectance primaries with exact D65 reproduction;
- [Jakob and Hanika 2019](https://rgl.epfl.ch/publications/Jakob2019Spectral)
  uses a compact smooth, energy-conserving function space; its official
  [rgb2spec repository](https://github.com/mitsuba-renderer/rgb2spec) is
  BSD-3-Clause.

Neither method recovers the original scene spectrum. Agreement between
multiple canonicalizers is therefore a necessary robustness diagnostic, not
evidence that one reconstruction is physically true.

## External-control boundary

The current spektrafilm repository is an informative external control only.
Its current project page states GPLv3 for software, CC BY-SA 4.0 for profiles,
and explicitly says the Portra CMY diffuse densities are generic because they
are normally unpublished. No current code, profile, LUT, graph trace or
method-specific implementation is copied or used as a teacher. The existing
RF2.C0 isolated comparison remains unchanged. This clean-room branch is
derived from manufacturer sources, CIE data and primary papers.

## Candidate comparison

| Candidate | Public inputs sufficient? | Main expected value | Fatal or claim-limiting gap | Decision |
|---|---:|---|---|---|
| Velvia 50 direct-positive datasheet witness | partially | stock-specific nonlinear layer and dye interaction; no print ambiguity | graph digitization, RGB metamerism, no scan/projector/process calibration | open H0A as `film-inspired/datasheet-prior` |
| Portra/Ektar direct negative scan | no | could model negative formation | no independent CMY basis or scanner interpretation | close until new measured/public data |
| Portra/Ektar negative-to-print | no | most plausible intended colour interpretation | negative dye bases, paper sensitivity/tone/dyes, filtration and process all missing | close; do not synthesize missing truth |
| Fujifilm internal PFS reproduction | no | strongest published physical factorization | numerical internal profiles are not published | method prior only |
| black-and-white development simulation | partially | density/grain/development research | not the current colour-style priority and still lacks calibrated input/exposure | defer, not reject |
| fit current unpaired stock scans | forbidden | might look attractive | stock/source/content identifiability failed | closed by SF1.3B |

## H0A: allowed next leaf

H0A may implement an isolated **Velvia 50 datasheet spectral witness** under a
separate frozen contract. Its first job is identifiability and numerical
feasibility, not visual promotion.

Required stages:

1. version the page/image hashes, graph axes and manually or automatically
   digitized knots;
2. render an overlay and quantify pixel-space trace error so graph extraction
   cannot silently become hand-authored colour grading;
3. reconstruct bounded reflectance spectra from linear display-sRGB using a
   clean-room smooth constrained solution;
4. generate at least two colour-matched metamer alternatives in the null
   space of the D65 CIE observer;
5. expose digitized Velvia sensitivity layers, map through digitized reversal
   characteristic curves, compose the separated dye-density spectra, and
   integrate the resulting transmittance under explicit D50 and D65 viewing
   illuminants;
6. report input colourimetric reconstruction error, output metamer spread,
   illuminant spread, neutral-axis behaviour, raw range, clipping and sampled
   operator regularity on a synthetic grid;
7. stop as `unidentified` if the apparent stock effect is dominated by the
   choice of RGB-to-spectrum canonicalizer.

The first implementation must not use real-film images, owner anchors or
existing style outputs to choose curve knots, exposure, white balance,
strength, illuminant or thresholds. Only after H0A's numerical report is
frozen may a separately preregistered H0B evaluate a fixed operator on the A0
gold/stress images.

## DoR, DoD and stop conditions

H0A DoR:

- exact manufacturer/CIE inputs and hashes are available;
- digitization and canonicalization assumptions are explicit;
- no GPL/profile incorporation and no current-pixel fitting;
- output label and claim ceiling are frozen.

H0A DoD:

- deterministic code, curve data, overlay evidence and tests;
- exact config hash, software commit, seed and synthetic manifest;
- at least one smooth canonical spectrum and two metamer stress spectra;
- numerical report with no post-result threshold changes;
- branch decision: `stable_enough_for_fixed_visual_pilot`,
  `canonicalizer_sensitive_unidentified`, or `numerically_invalid`.

Stop H0A immediately for invalid/ambiguous graph axes, non-finite output,
unbounded spectra, non-monotone neutral-axis failure that cannot be explained
by the published reversal curve, or unreproducible traces. Do not add a
learned model, fit stock pixels, borrow external profiles, or tune against the
visual set to rescue it.

## Claim ceiling

At best H0A can establish a deterministic, data-sheet-constrained,
film-inspired Look Approximation and measure its sensitivity to missing
spectral information. It cannot establish a real digital-to-Velvia operator,
a calibrated stock response, an original scene spectrum, a process/scanner
profile, authenticity, product preference, or latent stock modes. Ultimate
remains stock-first and ACTIVE regardless of this branch result.
