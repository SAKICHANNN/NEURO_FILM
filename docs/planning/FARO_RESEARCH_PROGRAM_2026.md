# FARO Research Program — Strong Film-Inspired Colour Under Controlled Policy Risk

> **Publication-priority update, 2026-07-12:** this document is no longer the
> primary paper route. The algorithm-first colour-transfer program is
> [`ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md`](ROLL2FILM_COLOR_TRANSFER_RESEARCH_2026.md).
> ChromaticTail/FilmStyleSafe is supporting evaluation, and FARO is a product
> safety/system wrapper. A standalone benchmark paper is not the fallback if
> Roll2Film fails. The historical FARO method and experiment design below are
> retained as baselines, evaluation assets and system-engineering context.

> Status: research source of truth from 2026-07-11. This document defines a
> falsifiable scientific program, not an implemented-model claim.
>
> Product goal: transform digital photographs into visibly strong,
> film-inspired colour while materially reducing transformation-induced
> failures such as a large purple sky region, neon chroma islands, skin
> contamination, highlight blotches, banding and unstable local colour.
>
> Research goal: advance photographic colour rendering beyond average
> reconstruction metrics by optimizing strong style subject to an explicit,
> policy-level severe-artifact budget on a declared evaluation distribution.
>
> Scope boundary: without controlled film/digital measurements, outputs are
> `film-inspired`, not calibrated reproductions of a named stock.

---

## 1. Executive decision

The previous FilmCase plan remains a useful baseline and ablation, but it is
not sufficiently novel as the primary paper contribution. Content-aware style
retrieval, diverse tone distributions, adaptive LUTs, bilateral-grid
retouching, physical film response models and preference learning all have
strong prior art.

The revised research hypothesis is **FARO**:

> **FARO = Film-Aesthetic Risk-controlled Operator policy.** A frozen policy
> proposes a fixed-budget set of explicit colour operators, renders each one on
> the query, ranks the observed outcomes, audits the finalist through the exact
> full-resolution export path, and otherwise returns identity. Calibration and
> evaluation apply to the **complete proposal → render → select → fallback
> policy**, never to candidates in isolation.

The key scientific object is no longer an RGB target or a similarity between
two source photographs. It is the cross-application outcome:

```text
(query image, explicit transform) -> rendered candidate
                                  -> style utility
                                  -> local severe-artifact risk
```

The central claim is deliberately conditional and narrower than a generic
"safe LUT" claim:

> Selection-aware, end-to-end control of a fixed best-of-`K` colour policy can
> move the deployed style–artifact–coverage frontier beyond a single global
> operator, a source-similarity retriever and candidate-wise rejection.

If the fixed-bank, cross-rater empirical ceiling cannot beat the best global
operator, this claim is false and the adaptive branch stops. If the frozen
policy does not satisfy its pre-registered marginal empirical risk gates on
hidden scenes, the project must not make a risk-control claim.

The historical FARO priority was:

1. **Paper A first (superseded 2026-07-12):** establish that local severe chromatic failures are a
   real, measurable evaluation gap;
2. **FARO system paper only after four gates:** a frozen operator bank has
   useful empirical-ceiling headroom, complete-policy calibration works,
   target-look salience is non-trivial, and all-scene preference improves;
3. **new FARO method paper is blocked:** LTT, two-stage risk control and the
   2026 joint selective certificate already cover the generic statistical
   shape. A separate theory audit must identify and prove a narrower gap before
   any algorithmic novelty claim.

---

## 2. Two deliverables, two standards

### 2.1 Engineering product

The product may reuse established methods. Its standard is reliability and
visual quality, not novelty.

Recommended product stack:

```text
high-precision WorkingImage / colour-state contract
  -> neutral/canonical input transform
  -> curated global film-look operator bank
  -> optional bounded local affine/bilateral residual
  -> analytic + learned artifact monitor
  -> preference selector
  -> deterministic identity/minimal-transform fallback
  -> grain / halation / bloom as separately inspectable effects
  -> colour-managed export + replayable recipe
```

The initial product can use hand-designed or offline-fitted curves/LUTs and the
known preferred anchors. It does not have to wait for the FARO generator or a
paper result.

### 2.2 Scientific program

The benchmark paper must establish the problem and evaluation protocol. A
separate method paper must earn one irreducible contribution rather than claim
novelty for the whole engineering stack:

1. strong photographic stylization under a non-compensable severe-artifact
   constraint;
2. a local artifact benchmark and risk–coverage evaluation that exposes
   failures hidden by mean PSNR/SSIM/LPIPS;
3. a fixed-budget, cross-rater empirical-ceiling protocol that can falsify the
   need for routing;
4. for the FARO system, adopt and validate selection-aware calibration of the
   complete adaptive best-of-`K` policy, including fallback;
5. for a method paper only, solve a pre-audited statistical problem not already
   covered by LTT, two-stage risk control, joint selective certificates,
   noisy-label robustness or non-exchangeable CRC.

The paper must not claim novelty for LUTs, bilateral grids, hard Top-1 routing,
physical response curves, diverse style sampling or pairwise preference alone.

---

## 3. Closest prior art and novelty threats

### 3.1 Global, white-box and bounded colour operators

| Work | What already exists | Boundary relevant to FARO |
|---|---|---|
| [Deep Bilateral Learning, TOG 2017](https://groups.csail.mit.edu/graphics/hdrnet/) | A low-resolution network predicts a bilateral grid of local affine transforms applied at full resolution. | Structural fidelity does not guarantee semantic colour safety. |
| [Exposure, TOG 2018](https://arxiv.org/abs/1709.09602) | Reinforcement learning chooses interpretable global editing operations from unpaired high-quality targets. | No explicit local severe-artifact budget or calibrated rejection. |
| [DeepLPF, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Moran_DeepLPF_Deep_Local_Parametric_Filters_for_Image_Enhancement_CVPR_2020_paper.html) | Networks predict local parametric filters. | Average reconstruction evaluation does not characterize rare colour disasters. |
| [Image-Adaptive 3D LUT, TPAMI 2020](https://arxiv.org/abs/2009.14468) | Image-adaptive fusion of learned basis LUTs. | Soft fusion can average styles; no semantic tail-risk gate. |
| [Spatial-Aware 3D LUT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.html) | Image and pixel-level adaptive LUT fusion. | Local freedom can create the exact failures FARO must measure. |
| [AdaInt, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_AdaInt_Learning_Adaptive_Intervals_for_3D_Lookup_Tables_on_Real-Time_CVPR_2022_paper.html) | Learnable non-uniform LUT sampling. | Better approximation is not a severe-artifact guarantee. |
| [SepLUT, ECCV 2022](https://arxiv.org/abs/2207.08351) | Separated 1D tone and 3D colour transforms. | Strong engineering representation and required baseline, not a new thesis. |
| [Neural Preset, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ke_Neural_Preset_for_Color_Style_Transfer_CVPR_2023_paper.html) | Self-supervised deterministic neural colour mapping with normalization then stylization. | Strong content fidelity, but no calibrated local disaster risk. |
| [RSFNet, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ouyang_RSFNet_A_White-Box_Image_Retouching_Approach_using_Region-Specific_Color_Filters_ICCV_2023_paper.html) | Region-specific white-box colour filters and masks. | Region masks can themselves introduce semantic colour errors. |
| [SA-LUT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Gong_SA-LUT_Spatial_Adaptive_4D_Look-Up_Table_for_Photorealistic_Style_Transfer_ICCV_2025_paper.html) | Style-guided 4D LUT plus content–style context map and PST50. | Extreme exposure and semantic mismatch remain failure modes; no abstention. |
| [SVDLUT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Lightweight_and_Fast_Real-time_Image_Enhancement_via_Decomposition_of_the_ICCV_2025_paper.html) | Efficient decomposition of spatial-aware LUTs. | Efficiency contribution, not tail-risk control. |

### 3.2 Strong, diverse and multimodal photographic style

| Work | What already exists | Boundary relevant to FARO |
|---|---|---|
| [Automatic Content-Aware Color and Tone Stylization, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Lee_Automatic_Content-Aware_Color_CVPR_2016_paper.html) | Content-aware style ranking, diverse exemplar selection, regularized global transfer and human evaluation. | Directly defeats novelty claims based only on content-aware style selection. |
| [CanonCGT, CVPR 2026/arXiv](https://arxiv.org/abs/2606.01638) | Reference grading through a canonical pivot representation. | A strong neutral-pivot/reference-transfer baseline, not evidence for FARO novelty. |
| [Context-Based Automatic Local Image Enhancement, ECCV 2012](https://www.microsoft.com/en-us/research/publication/context-based-automatic-local-image-enhancement/) | Candidate-image retrieval and local context-dependent transformation search. | Retrieval alone is old; FARO must condition on the actual transform outcome and risk. |
| [Deep Preset, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Ho_Deep_Preset_Blending_and_Retouching_Photos_With_Color_Style_Transfer_WACV_2021_paper.html) | Predicts parameters of low-level colour transformations from references. | Parameter prediction is not sufficient novelty. |
| [TSFlow, 2022/2024](https://arxiv.org/abs/2207.05430) | A normalizing flow models diverse tone styles in style space and explicitly addresses average-style collapse. | Diversity alone is already occupied; no explicit severe-risk budget. |
| [DiffRetouch, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32288) | Diffusion models diverse expert retouching and predicts affine bilateral transforms. | Strong direct baseline for multimodality; no fail-closed deployment policy. |
| [D-LUT, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Li_D-LUT_Photorealistic_Style_Transfer_via_Diffusion_Process_WACV_2025_paper.html) | Score matching and Langevin dynamics derive reusable 3D LUTs from a style image. | Diffusion/score-based LUT generation is not new. |
| [Video Color Grading via LUT Generation, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Shin_Video_Color_Grading_via_Look-Up_Table_Generation_ICCV_2025_paper.pdf) | Diffusion directly generates a global LUT conditioned on reference style. | Operator-space generation is occupied without FARO's risk formulation. |
| [InstantRetouch, personalized retouching, 2025/2026](https://arxiv.org/abs/2602.17044) | Content-similar paired examples are retrieved and their style latents aggregated. | Source similarity and latent aggregation are required baselines. |
| [InstantRetouch, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_InstantRetouch_Efficient_and_High-Fidelity_Instruction-Guided_Image_Retouching_with_Bilateral_Space_CVPR_2026_paper.html) | A diffusion teacher is distilled into a one-step bilateral-grid renderer. | Distilling generative priors into bounded operators is already occupied. |
| [Hist2Style, 2026](https://arxiv.org/abs/2606.01819) | Strong reference style is distilled into locally affine bilateral transforms and evaluated with photographers. | A strong engineering and paper baseline; filtering is not calibrated tail-risk control. |
| [Personalized Photographic Style / PPSD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Learning_Personalized_Photographic_Style_from_Pairwise_User_Preferences_CVPR_2026_paper.html) | Pairwise preferences from 767 users and comparative personalized-style evaluation. | Preference learning itself is not new. |
| [RetouchIQ, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_RetouchIQ_MLLM_Agents_for_Instruction-Based_Image_Retouching_with_Generalist_Reward_CVPR_2026_paper.html) | Tool-using retouch agents and a generalist reward model. | Required reward/evaluator baseline; a scalar reward does not establish local severe-risk control. |
| [StatLUT, arXiv 2026](https://arxiv.org/abs/2607.08227) | Spatially agnostic statistics, topologically smooth LUT generation and multimodal text control. | A very recent, non-peer-reviewed threat to topology-safe multimodal LUT claims. |

### 3.3 Pixel-generative and unpaired translation alternatives

| Work | What already exists | Boundary relevant to this project |
|---|---|---|
| [CUT, ECCV 2020](https://arxiv.org/abs/2007.15651) | Unpaired translation with patchwise contrastive content preservation. | A required historical baseline; pixel generation can alter local content and does not solve multimodal style choice or chromatic tail risk. |
| [SDEdit, ICLR 2022](https://arxiv.org/abs/2108.01073) | Noise-and-denoise editing trades input fidelity against realism through a pretrained diffusion prior. | Useful product/prototype option, but stochastic pixel synthesis can change geometry, text and texture. |
| [InstructPix2Pix, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html) | Instruction-conditioned image editing learned from generated edit pairs. | The current V3 engineering baseline; prompts/strength do not provide film identification or local chromatic-risk control. |
| [ControlNet, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | Adds spatial conditioning to pretrained diffusion models. | Can reduce structural drift, but cannot certify colour validity and adds compute. |
| [IP-Adapter, 2023](https://arxiv.org/abs/2308.06721) | Decoupled image-prompt conditioning for diffusion. | Useful reference/style conditioning; reference strength and content preservation remain empirical. |

These models remain engineering comparators. FARO deliberately evaluates
colour-operator policies first because exact geometry preservation and
full-resolution rendering are easier to audit; this is an engineering choice,
not proof that operator outputs are aesthetically safe.

### 3.4 Film-specific work

| Work | What already exists | Claim boundary |
|---|---|---|
| [FilmSet / FilmNet, IJCAI 2023](https://www.ijcai.org/proceedings/2023/0129.pdf) | 5,285 high-resolution images and a multi-frequency film-style network. Targets are Capture One recipes, not measured film scans. | Useful pseudo-preset baseline; insufficient for named-stock truth. |
| [CNNs for Digital-to-Film, 2024](https://arxiv.org/abs/2411.15967) | A small aligned digital/Cinestill 800T dataset and direct translation baselines. | Useful paired small-data baseline; data scale and process variability are severe limits. |
| [Emulating Emulsion, SIGGRAPH Posters 2025](https://musicofmusix.github.io/assets/misc/siggraph_abstract.pdf) | About 30 analytic parameters model digital RAW → film exposure → film response → scan RAW, fitted from one Velvia 100 roll. | Physical compact film colour is already occupied; negative/print/process uncertainty remains open. |
| [BlueNeg, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.html) | A high-quality 35mm negative restoration dataset and proxy-GT protocol. | Restoration evidence, not digital-to-film aesthetic supervision. |

### 3.5 Evaluation and risk

| Work | Useful precedent | Missing piece |
|---|---|---|
| [PPR10K, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html) | Human-region priority and group consistency. | Not strong film style or local chromatic tail risk. |
| [PIPAL, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1421_ECCV_2020_paper.php) | Large-scale human judgments show standard IQA failures. | Restoration distortions, not photographic colour catastrophes. |
| [Debiased Subjective Assessment, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cao_Debiased_Subjective_Assessment_of_Real-World_Image_Enhancement_CVPR_2021_paper.html) | Selects diagnostic samples by algorithm disagreement and image diversity. | No domain-specific severe-veto ontology. |
| [BAND-2k](https://arxiv.org/abs/2311.17752), [FS-BAND](https://arxiv.org/abs/2311.18216), [CAMBI](https://arxiv.org/abs/2102.00079) | Banding detection, maps and subjective labels. | Does not cover purple regions, chroma islands or skin/highlight contamination. |
| [DiffIQA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Toward_Generalized_Image_Quality_Assessment_Relaxing_the_Perfect_Reference_Quality_CVPR_2025_paper.html) | Separates fidelity and naturalness when reference quality is imperfect. | No non-compensable style-safety gate. |
| [HP-Edit, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_HP-Edit_A_Human-Preference_Post-Training_Framework_for_Image_Editing_CVPR_2026_paper.html) | Human-preference-aligned editing evaluator and benchmark. | Scalar scores can hide rare catastrophic failures. |
| [Risk-Controlling Prediction Sets, JACM 2021](https://arxiv.org/abs/2101.02703) | Distribution-free calibration of set-valued decisions for a declared expected loss. | A direct statistical baseline; applying it unchanged is not FARO novelty. |
| [Learn then Test, 2021](https://arxiv.org/abs/2110.01052) | Recasts risk-control calibration as multiple hypothesis testing. | A direct baseline for selecting among policies with valid risk bounds. |
| [Two-stage Risk Control, 2024/2025](https://arxiv.org/abs/2404.17769) | LTT/CRC guarantees tailored to retrieval followed by ranking. | Directly threatens novelty based on wrapping proposal and ranking into a complete policy. |
| [Label Noise Robustness of Conformal Prediction, JMLR 2024](https://www.jmlr.org/papers/v25/23-1549.html) | Characterizes and corrects risk control under several noisy-label regimes. | Human-label error is not an untouched statistical problem; any extension needs a sharper multirater/cluster gap. |
| [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | Calibrates nested decisions to control expected loss under assumptions. | Does not provide per-image or arbitrary-shift zero-defect guarantees. |
| [Non-Exchangeable Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html) | Extends CRC beyond exchangeable samples using relevance weights. | Shift/cluster extensions require comparison, not a generic novelty claim. |
| [Joint Finite-Sample Certificate for Adaptive Selective CRC, arXiv 2026](https://arxiv.org/abs/2606.08517) | Jointly upper-bounds selected risk and lower-bounds acceptance and deployment utility after adaptive finite-grid selection. | Highly isomorphic to FARO's generic risk–coverage–utility gates; currently a non-peer-reviewed but fatal novelty threat. |

The colour literature search found no verified benchmark that combines strong
film-inspired colour with localized severe chromatic-artifact labels and
all-scene deployed-policy evaluation. However, the statistical literature
**does** already cover finite-grid policy selection, two-stage retrieval/ranking
and joint risk–acceptance–utility certificates. Therefore complete-policy
calibration is an adopted reliability tool, not automatically a new algorithm.

---

## 4. Research insights after novelty pressure-testing

These are concise decision summaries, not hidden reasoning traces.

### Insight 1 — Average-style collapse is a target problem

Multiple visually valid outputs make RGB regression converge toward a bland
conditional mean. Larger networks do not repair an objective that rewards
averaging. TSFlow and DiffRetouch already establish this diagnosis, so FARO
must go beyond merely adding a stochastic latent.

### Insight 2 — Generating operators is safer, but not safe enough

Curves, LUTs and bilateral grids prevent generative geometry rewrite. They do
not prevent a smooth but semantically absurd purple sky or green face.
Structural fidelity and chromatic validity are different axes.

### Insight 3 — Local adaptation is both capability and hazard

Global LUTs are stable but cannot distinguish identical input colours used by
different semantic regions. Context maps solve this limitation while opening a
new path to local colour blocks, seams and instability. Local capacity must be
conditionally enabled, bounded and retractable.

### Insight 4 — This is a selective-decision problem

The desired system should not always emit its strongest candidate. It should
maximize style only among candidates supported by the current input and risk
model. Abstention and fallback are product capabilities, not evaluation
failures.

### Insight 5 — Mean quality metrics answer the wrong question

A severe defect covering 2–3% of an image may barely change PSNR, SSIM or
LPIPS, yet dominate human rejection. Safety and preference must be
lexicographic at the **deployed-policy** level: first require the fixed policy
to satisfy its severe-risk and coverage gates, then compare its actual output
on every scene, including fallback. Method-specific survivor subsets are
diagnostic only.

### Insight 6 — Physical accuracy and aesthetic preference are separate papers

A physical film process model requires controlled capture, process and scanner
evidence. Aesthetic film-inspired rendering can use unpaired references and
preferences. Mixing them produces unsupported authenticity claims and unclear
scientific conclusions.

### Insight 7 — Candidate calibration breaks after best-of-K selection

Even if every candidate score is marginally calibrated, proposing many
candidates and taking the highest predicted utility among those that pass a
threshold creates selection bias and winner's curse. The statistical object
must be the complete frozen policy, with candidate budget and fallback inside
the calibration loop.

### Insight 8 — The benchmark is the lower-risk scientific breakthrough

The strongest near-term novelty is not another LUT generator. It is a
prospective benchmark that makes rare local chromatic disasters
non-compensable and evaluates the deployed output on every scene. Paper B
exists only if that benchmark reveals headroom and a genuinely new
selection-aware control method closes it.

---

## 5. FARO method

### 5.1 Operational task and primary estimand

The primary task is **film-inspired look selection**, not named-stock
reproduction. A style `s` is a frozen look ID with:

- a method-independent, rights-cleared reference board spanning declared
  scene/exposure strata;
- a source/rights manifest and frozen board-construction rule;
- a written palette/tone specification, look-adherence rubric and
  forbidden-failure list.

The operator bank is part of the method and never part of the definition of
`s`. If the board lacks traceable film origin, the task is renamed
`photographic-look stylization`; named-film or physical-emulsion language is
not allowed.

The primary evaluator population is experienced photographers/colourists. The
confirmatory preference score is
`W = P(FARO win) + 0.5 * P(tie)` for the **deployed FARO output** versus the
strongest eligible global policy on the same scene. The primary style measure
`S` is a pre-registered binary/ordinal judgement of adherence to the independent
look specification by raters qualified on that look domain. Personal owner
preference and exact film-stock matching are secondary estimands and must not
be mixed into this claim.

For a scene `x`, style `s`, fixed candidate budget `K`, frozen proposer/bank
`B_s`, policy setting `lambda`, exact final renderer `F`, explicit transforms
`T_i` and fallback `I` (identity), define:

```text
candidates       B_s(x; K, seed=hash(scene_id, style_id, version)) = {T_1 ... T_K}
final outputs    y_i = F(T_i(x))
deployed output  pi_lambda(x, s) in {y_1 ... y_K, F(I(x))}
```

This makes confirmatory deployment deterministic; risk is over the declared
scene distribution. Any genuinely random/interactive sampling mode is a
separate estimand over scenes and internal randomness and is excluded from the
primary claim until independently calibrated.

`L_sev(x, pi_lambda(x,s))` is a scene-level binary event: the transform
introduced or materially amplified a pre-defined severe chromatic artifact.
`A(x, pi_lambda)` indicates that the policy deployed an adaptive candidate
rather than identity. `S(x,pi_lambda,s)` indicates that the deployed output
meets the frozen target-look threshold. Candidate risk scores are only ranking
features; they do **not** carry a statistical safety guarantee.

The complete frozen policy is eligible only if, on an independent hidden
sample:

```text
UCB_95[P(L_sev=1)]                         <= epsilon_policy
UCB_95[P(L_sev=1 | A=1)]                   <= epsilon_selected
LCB_95[P(S=1)]                             >= s_min
LCB_95[P(A=1 and S=1)]                     >= c_min
LCB_95[W = win + 0.5*tie]                  > 0.50 + delta
```

Planning defaults are `epsilon_policy = epsilon_selected = 1%`,
`s_min = 70%`, `c_min = 50%` and `delta = 5 percentage points`; all values,
the confidence procedure, `K`, deterministic seed rule, comparator and sample
size must be frozen by preregistration before hidden evaluation. These are
design targets, not achieved results. The selected-output risk gate prevents
identity fallback from diluting failures among strong adaptive outputs. Severe
outputs are never compensated by style: missing any gate makes the policy
ineligible.

### 5.2 Adopted selection-aware policy calibration

Construct a finite, pre-declared family of complete policies
`Pi = {pi_lambda}`. Each member includes proposal, candidate count, seeds,
preview, finalist selection, full-resolution rendering/audit and fallback.
Evaluate the **actual final output** of every `pi_lambda` on calibration scenes
and choose a policy using a pre-selected existing framework: Learn then Test,
two-stage risk control, or the joint finite-sample selective certificate. The
procedure must jointly account for policy-level and selected-output risk,
style-qualified coverage and utility; cross-fitting is required when learned
scores share labels with policy choice. Policy selection and tie-breaking are
deterministic: certified utility first, then lower certified risk, then higher
style-qualified coverage, then manifest ID.

The generic guarantee target is:

```text
Pr{ R_policy(pi_hat) <= epsilon_policy,
    R_selected(pi_hat) <= epsilon_selected,
    S(pi_hat) >= s_min,
    C_style(pi_hat) >= c_min } >= 1 - alpha
```

Its exact assumptions, confidence allocation and utility statement come from
the adopted procedure and must be reproduced, not renamed as a FARO theorem.
The statement remains marginal and distribution-specific, not a per-image,
OOD or universal zero-defect certificate.

#### R4T theory extension — blocked / method target unresolved

A possible theory study would combine clustered scene sampling, correlated
best-of-`K` outcomes and fallible multirater severe labels. But each ingredient
already has adjacent work in noisy-label conformal prediction,
non-exchangeable CRC and joint selective certification. Before any derivation,
R4T requires a dedicated statistics novelty audit and a written theorem target
specifying the policy class, cluster/noise model, confidence allocation,
eligibility set, utility rule and deterministic tie-breaking.

R4T passes only if it proves a valid guarantee not implied by the closest
methods and yields a materially tighter risk–style–coverage frontier against
LTT, two-stage risk control and the 2026 joint certificate. Otherwise FARO
remains a computer-vision system/application paper and Paper A carries the
primary scientific novelty.

### 5.3 Numerically constrained operator family

FARO uses a capacity ladder rather than one unconstrained network. These
constraints preserve geometry and numerical regularity; they do not certify
semantic colour validity.

#### Layer A — physical/photographic global core

- camera/input normalization in a declared scene/display colour state;
- optional 3×3 digital-to-film mixing matrix;
- monotone per-layer log-exposure response curves;
- optional film-to-scan/output matrix;
- explicit negative/direct-positive/print interpretation metadata.

This optional layer is inspired by, and must compare against, Emulating
Emulsion. A universal 3x3 mixing matrix is underdetermined under metamerism and
camera/process variation; no physical-authenticity claim is allowed without
controlled data. It is not part of Paper B's core novelty.

#### Layer B — numerically constrained global aesthetic residual

- identity-plus-residual 3D LUT or compact neural colour field;
- monotone luminance response;
- bounded first/second derivatives;
- tetrahedral or validated interpolation;
- gamut projection and output-headroom contract;
- neutral-axis and saturated-source guard bands;
- no spatial resampling.

The parameterization should enforce as many properties by construction as
possible. Soft regularization alone is not a certificate, and even a smooth,
invertible, in-gamut LUT can turn an entire sky purple. This layer is a strong
renderer/baseline, not semantic safety.

#### Layer C — optional bounded local residual

- low-resolution bilateral grid of affine colour transforms;
- fixed guidance/slicing semantics;
- amplitude, total-variation and halo/seam bounds;
- unsupported-output-edge penalty;
- disabled for low-confidence semantic/state inputs;
- removal required if it does not independently improve a pre-registered local
  failure class.

Coefficient bounds do not imply end-to-end output bounds: the guide gradient,
slicing, input gradient and gamut mapping all contribute to the final
derivative. A low-amplitude smooth purple sky can also pass amplitude/TV/seam
checks. Therefore local constraints are diagnostic regularizers only; final
outputs require the same policy-level human risk evaluation.

### 5.4 Fixed-bank core and optional proposals

The first Paper B system uses a small frozen discrete bank with fixed `K`, cost
and seeds. More complex proposers are optional extensions, in increasing
complexity:

1. curated discrete operator bank;
2. medoid retrieval from a transform library;
3. conditional mixture over discrete operator atoms;
4. normalizing flow over constrained operator parameters;
5. diffusion/flow generation in operator space;
6. generative teacher distilled only into the constrained parameter space.

The proposal mechanism never directly emits the final full-resolution RGB
image. A stochastic generator must prove value against the discrete bank and
TSFlow/DiffRetouch/StatLUT-style baselines at identical candidate/render
budget. Diversity without higher eligible policy utility is not progress and
operator generation is not claimed as novelty.

### 5.5 Counterfactual outcome scorer

scorer consumes:
Every proposed operator is first rendered on a colour-managed preview. The
scorer consumes:
scorer consumes:

- input photometric and semantic descriptors;
- operator descriptor and constraint margins;
- preview/output diagnostics;
- difference maps and local connected-component statistics.

It produces separate outputs:

- style salience;
- pairwise preference utility;
- severe-artifact probability by category;
- uncertainty/OOD support;
- explanation record naming the risky region/category.

This differs from source-only retrieval: the score conditions on what the
specific transform actually did to the query. Preview scores may only
pre-screen/rank. Every finalist must be re-rendered and audited at native
resolution through the same bit depth, gamut mapping, tiling and codec path as
the exported result; the preview can miss small islands, banding and seams.

### 5.6 Artifact evidence stack

The monitor combines three complementary layers. None is a certificate.

#### Analytic checks

- output gamut/headroom and clipping components;
- LUT smoothness, monotonicity and Jacobian fold checks;
- local Lipschitz/amplification estimates;
- unsupported chroma edges in input-smooth regions;
- banding detectors;
- perturbation stability under lossless re-encode, small exposure/WB shifts and
  benign resize/crop.

#### Semantic/local checks

- sky, skin, neutral wall/snow, foliage, saturated red/yellow, neon and
  highlight regions;
- large unexpected hue components;
- chroma islands and false neon edges;
- high-chroma highlight quantization/blotching;
- local mask seams or inconsistent same-material colour.

#### Learned severe-artifact model

- trained on real failures from all baseline families;
- augmented with parameterized synthetic failures;
- predicts spatial masks and category-specific severity;
- validated across held-out algorithm families and unseen transform
  parameterizations, not only held-out images;
- tested with and without operator descriptors to expose family shortcuts.

No learned IQA score may clear a candidate by itself.

### 5.7 Counterexample-guided refinement

FARO actively searches for failures rather than waiting for random test images.

Counterexample families include:

- smooth sky/skin/neutral gradients across exposure and white balance;
- saturated red metal and specular highlights;
- low-light sensor noise amplified into colour islands;
- narrow-gamut JPEG and 8/10/16-bit ramps;
- mixed illuminants and near-clipped channels;
- semantic pairs with similar RGB but different valid mappings;
- small input perturbations intended to expose route or context-map jumps.

Optimize input nuisance parameters and operator parameters to maximize monitor
risk inside realistic bounds. Verified failures enter an error table and the
next training/evaluation cycle. Counterexamples never enter final test after
being used for model selection.

A mandatory analytic counterexample is a broad, smooth, in-gamut purple-sky
mapping that satisfies monotonicity, smoothness, Lipschitz and local-TV checks.
It demonstrates why structural/numerical constraints cannot be advertised as
semantic safety.

### 5.8 Selective deployment and fallback

The formal paper fallback is identity, because the benchmark labels
transformation-induced or amplified chromatic harm. A product may additionally
offer a very weak validated global transform, but its risk must be reported and
it must never be called "known-safe." Report:

- raw adaptive and style-qualified adaptive coverage;
- selected-output severe risk;
- overall deployed-policy risk including fallback;
- style gain over fallback;
- coverage at a fixed upper risk bound;
- calibration error and OOD fallback rate.

The selected `pi_lambda`, not its accepted candidates, is calibrated. The
confirmatory policy uses the frozen scene/style-hash seed rule, so it is
deterministic for a versioned input. The adopted LTT/CRC/joint-certificate
procedure may be used only when its stated exchangeability, loss, policy-family
and selection assumptions hold. It provides a marginal finite-sample statement
on the declared distribution, not universal per-image safety. Worst-group,
unseen-family and shifted-domain results remain separate empirical reports.

---

## 6. FilmStyleSafe benchmark

### 6.1 Evaluation principle

Use two labels in a fixed hierarchical protocol; do not evaluate preference on
a method-dependent survivor subset.

#### Stage 1 — severe-artifact adjudication

Reviewers see input/output at full resolution and 100% crops. Each result is
labelled `safe`, `minor`, `severe` or `uncertain`, with category and mask.
`Severe` means that the transform introduced or materially amplified an
objectionable chromatic failure relative to the input. Uncertain cases require
adjudication. Every confirmed event enters the scene-level numerator and is
retained in all later analysis. The policy fails only when the pre-registered
upper risk bound exceeds its budget; this protocol is not a zero-event veto.

#### Stage 2 — end-to-end deployed-output preference

Every scene enters blinded comparison using the output the policy actually
deployed, including identity fallback:

1. with the independent reference board visible, does each output meet the
   frozen palette/tone/look-adherence rubric;
2. which has the clearer and stronger target-look character;
3. which would the reviewer keep as the final photograph, with tie allowed.

Conditional preference among non-severe adaptive outputs is a secondary
diagnostic only. Nuisance-matched controls with approximately matched mean
luminance, contrast, temperature and total chroma test simple explanations,
but residual confounding means they cannot prove a causal palette mechanism.

### 6.2 Severe-artifact ontology

- purple/green/other implausible large smooth-region cast;
- skin or neutral-region contamination;
- neon chroma island or high-chroma highlight speckle;
- banding, posterization or hard colour shelves;
- clipping/gamut components with visible hard boundaries;
- local hue jumps, grid seams or halo;
- chromatic instability under benign exposure/WB/encode perturbation.

Geometry, face, text, object and texture corruption are a separate structural
ontology used only when a generative RGB baseline is included. They are not
evidence for or against a colour-operator safety claim.

### 6.3 Semantic Chroma-Island Score

Develop and validate a local metric, provisionally `SCIS`:

1. estimate the smooth global relation between input and output;
2. compute residual chroma in a perceptual opponent space;
3. restrict detection to input-smooth or semantically sensitive regions;
4. threshold at multiple scales and find connected components;
5. report maximum component area, top-k area, q95/q99 residual, boundary
   mismatch and worst semantic slice.

SCIS can be structurally biased toward global LUTs, so metric development must
include legitimate local edits as hard negatives: local exposure/dodge-burn,
skin protection, sky densification, mixed-illuminant correction and intentional
local colour accents. Human labels must distinguish intended local edits from
objectionable islands. The detector and threshold are frozen before testing on
unseen algorithms/transform families; the primary metric test is sensitivity
at a pre-registered low false-positive rate, with AP/IoU secondary. SCIS is a
candidate contribution, not an assumed-valid label generator or safety gate.

### 6.4 Required scorecards

#### Artifact

- overall deployed-policy and adaptive-selected severe rates with one-sided
  confidence bounds;
- per-category and worst-group rate;
- SCIS and spatial mask AP/IoU;
- banding, clipping/gamut component and unsupported-edge diagnostics;
- median, q95, q99 and worst case for a separately frozen continuous severity
  score; CVaR is not reported for the binary severe event.

The confidence unit is the independent parent scene, never crops, candidates
or perturbations. Exact binomial bounds are used only when the confirmatory
sample contains one independently sampled scene per provenance cluster; if
clusters contribute multiple scenes, a pre-validated cluster-level bound
replaces it. Worst-group confidence claims require their own powered scene
counts; otherwise they remain descriptive. Adversarial stress results are
reported separately from marginal deployment risk.

#### Style and appeal

- pairwise style-salience and keep-preference rates;
- hierarchical Davidson or Thurstone-with-ties analysis;
- look-adherence rate and style-qualified adaptive coverage with lower bounds;
- nuisance-control win rate;
- inter-style separation and cross-scene identity;
- optional exposure-conditioned distance to unpaired film reference
  distributions, never used alone as truth.

#### Content

- architectural proof of no spatial resampling for global operators;
- edge/keypoint/OCR/face diagnostics only for generative RGB baselines;
- DISTS/LPIPS/SSIM only as secondary diagnostics.

#### Routing and selection

- fixed-bank, cross-rater empirical-ceiling gain over the strongest eligible
  global policy;
- regret to that empirical ceiling, not regret to a universal Oracle;
- expert utilization and collapse;
- risk–coverage and style-gain–coverage curves;
- OOD and unknown-state fallback behavior;
- route stability under benign perturbations.

#### Product

- 4K/24MP/100MP latency and memory;
- CPU, Windows GPU and Apple parity;
- deterministic replay and recipe hashes;
- tile/full-frame agreement and video consistency.

### 6.5 Human-study requirements

- blind randomized presentation;
- rater qualification on chromatic-failure and legitimate-local-edit examples;
- separate look-domain qualification against the frozen reference-board
  rubric;
- at least three independent severe judgements per deployed output, with the
  final number determined by reliability and power pilots;
- trained adjudication for uncertain/severe cases;
- pairwise preference with ties allowed;
- scene and reviewer treated as grouped/random effects;
- cluster bootstrap by scene;
- sentinel/repeat trials, inter-rater reliability, pre-registered rater
  exclusions and stopping rules;
- a frozen scene-label rule: three initial blind ratings; if all three are in
  `{safe, minor}`, set `L_sev=0`; any `severe` or `uncertain` triggers a
  separate three-senior-rater blind panel; panel majority `severe` yields
  `L_sev=1`, panel majority non-severe yields `0`, and unresolved/missing
  adjudication is conservatively counted severe in the primary analysis;
- sentinel sensitivity/specificity and a pre-registered conservative
  misclassification sensitivity analysis;
- sample-size/power analysis for both severe risk and minimum preference gain;
- grain, halation, bloom and sharpening disabled; colour state, export gamut,
  bit depth and viewing zoom fixed during the colour study;
- display/viewing protocol based on
  [ITU-R BT.500-15](https://www.itu.int/rec/R-REC-BT.500-15-202305-I/en).

AI/VLM judges may help triage or provide a secondary reproducibility channel;
they cannot replace external human evidence for the primary preference claim.
The default statistical conclusion concerns the **protocol-adjudicated severe
rate**. A latent true-severe claim requires an independently validated
misclassification model or a conservative correction that still passes.

### 6.6 Preregistration table for the confirmatory study

| Item | Planning value to freeze before hidden data |
|---|---|
| Target task | One method-independent look ID, rights/source manifest, board-construction rule and palette/tone rubric at a time; rename to photographic look if film origin is not traceable |
| Scene sampling frame | Named source pools, inclusion probabilities, strata, target weights and exclusions; one random parent scene per provenance cluster for exact-binomial analysis |
| Rater population | Recruitment channels, minimum photography/colour experience, look-domain qualification, geography/display eligibility and any analysis weights |
| Unit | Independent parent scene; source/uploader/camera/roll clustered before splitting; stress scenes analyzed separately |
| FARO policy | Entire proposer, `K`, scene-hash seed rule, scorer, threshold, full-resolution renderer/audit and identity fallback |
| Comparator | Strongest eligible global policy selected on independent pre-evaluation scenes using the same look, renderer/export, adjudication, `epsilon` and style-salience gates; comparator fallback frozen |
| Safety endpoints | Overall protocol-adjudicated scene severe rate and conditional severe rate among adaptive outputs |
| Safety gates | One-sided 95% upper bounds `<= epsilon_policy` and `<= epsilon_selected`; planning both at `1%` |
| Label rule | Frozen initial/adjudication voting, uncertain/missing handling, blind state, sentinel accuracy and misclassification sensitivity analysis |
| Style gate | One-sided 95% lower bound on frozen look-adherence pass rate `>= s_min`; planning `s_min=70%` |
| Anti-triviality gate | One-sided 95% lower bound on `P(A=1 and S=1) >= c_min`; planning `c_min=50%` |
| Primary efficacy endpoint | All-scene deployed-output tie score `W=P(win)+0.5*P(tie)` versus the eligible global policy |
| Minimum effect | Lower one-sided 95% bound on `W` `>0.55` (planning `delta=5pp`); this, not a point-estimate-only rule, supports a 5pp minimum-gain claim |
| Candidate budget | Planning `K=8`, identical render/compute budget for best-of-`K` baselines |
| Multiplicity | Adopt LTT/two-stage/joint-certificate control on calibration; test one frozen policy using the pre-registered conjunction of both safety, style, style-qualified coverage and efficacy gates |
| Randomness | Primary deployment deterministic by versioned scene/style hash; stochastic modes require a separate scene×seed estimand and calibration |
| Sample size | Cluster-aware risk plus tie-model simulation, frozen before test; about 300 independent **adaptive accepted scenes** are needed even for a zero-event one-sided 95% selected-risk upper bound near 1%, so 50% coverage can require roughly 600 total scenes before failures, grouping, label error or replication |
| Failure rule | Missing any gate yields no confirmatory FARO superiority/risk-control claim; thresholds are not relaxed after seeing hidden outcomes |

If this sample/annotation budget is infeasible, the honest alternatives are to
pre-register a larger `epsilon` or publish descriptive benchmark results. More
candidates, crops or correlated perturbations cannot replace independent
scenes.

The confirmatory deployment sample is drawn from a frozen rights-cleared
manifest rather than an informal image collection. The manifest must specify
scene-source probabilities and weights for portrait, sky/landscape,
architecture/neutral, foliage, night/neon, saturated-highlight, mixed-light
and low-light strata. Deliberately adversarial stress strata are reported in a
separate table and never reweighted to impersonate deployment risk.
Exact binomial inference is reserved for an IID draw from that frozen target
mixture with one scene per provenance cluster. Fixed stratified quotas,
unequal weights or repeated clusters require a pre-specified design/cluster
bound validated in simulation before B3.

### 6.7 Leakage barriers and empirical-ceiling protocol

Paper A and FARO use separate, provenance-grouped hidden partitions:

| Split | Purpose | Forbidden reuse |
|---|---|---|
| A0 benchmark development | Ontology, annotation guide, SCIS features/threshold and legitimate-local hard negatives | No tuning on A1 or any B-hidden scenes |
| A1 Paper-A hidden test | Confirm metric gap, low-FPR SCIS sensitivity and annotation reliability on unseen transform families | One evaluation before Paper A freeze; never used as the FARO method test |
| B0 model/global development | Operator bank, scorer/monitor training, synthetic counterexamples and an independently selected strongest eligible global policy | No B1-B4 outcomes for model/comparator choice |
| B1 empirical ceiling | Frozen bank and `K`; one panel selects the best adjudicated-nonsevere, look-qualified member or identity when none exists; a disjoint panel evaluates that complete all-scene policy against the eligible global comparator | Every scene stays in risk/preference denominators; not reused for calibration/final claims |
| B2 policy calibration | Run every complete `pi_lambda`; adopt LTT/two-stage/joint certification and select exactly one frozen policy | No threshold changes after B3 begins |
| B3 hidden FARO test | Both safety gates, look adherence, style-qualified coverage and all-scene tie-score endpoint | One evaluation only; no error inspection before lock |
| B4 external replication | New sources/clusters and an external rater pool | No development feedback before primary FARO report freeze |

Scene/source/uploader/camera/roll lineage and perceptual/near-duplicate hashes
are mandatory split keys across both paper programs. Artifact-monitor
evaluation also holds out complete failure/transform families and unseen
parameterizations. The B1 result is an **empirical fixed-bank ceiling**, not
proof of an optimal Oracle: bank size, render budget and
selection/evaluation rater panels are frozen. Its annotation
cost is budgeted explicitly as approximately `N*K*r_s` safety judgements plus
selection and independent comparison judgements.

---

## 7. Competing hypotheses

| ID | Hypothesis | Minimal discriminating test | Failure action |
|---|---|---|---|
| H-METRIC | Existing metrics fail to identify local severe chromatic artifacts at the required reliability. | Correlate PSNR/SSIM/LPIPS/IQA and SCIS candidates with dense human artifact labels across unseen algorithms. | If existing metrics suffice, remove the benchmark novelty claim. |
| H-STYLE | The independent look rubric identifies strong film-inspired character beyond generic preference or saturation. | Qualified-rater adherence study against traceable boards, nuisance controls and photographic-look controls. | If not, rename the task photographic retouching and remove film-style claims. |
| H-GLOBAL | A single numerically constrained global operator is sufficient. | Compare the strongest eligible global policy with the complete all-scene empirical-ceiling policy of a frozen fixed-budget bank. | If not rejected, stop adaptive routing and ship global. |
| H-MULTI | Strong target-look style is genuinely multimodal after matching saturation, contrast, luma and WB. | Cluster adjudicated-nonsevere, look-qualified operators using nuisance-matched controls and held-out scenes. | If modes disappear, stop stochastic/operator-distribution modeling. |
| H-PHYS | A physical global core improves data efficiency and cross-camera/exposure stability. | Matched-capacity physical+residual vs pure LUT on controlled and unpaired lanes. | If no independent gain, keep physical model only as metadata/optional calibrated lane. |
| H-LOCAL | Bounded local freedom improves a pre-registered failure class. | Global winner vs bilateral residual at matched style strength and both risk gates. | Worse risk eligibility or no independent gain deletes local branch. |
| H-SET | Set-valued proposals improve the eligible style frontier over one deterministic prediction. | Discrete bank/flow/diffusion candidates with equal compute and same selector. | If no gain, retain simplest deterministic proposer. |
| H-OUTCOME | Counterfactual outcome scoring beats source-only similarity. | Source retrieval vs transform descriptor vs preview-conditioned risk/utility ranker. | If no gain, remove the ranker. |
| H-GUARD | Counterexample training reduces real tail failures at matched style and coverage. | Train without/with counterexamples; test on unseen algorithms, parameterizations and real failure families, with descriptor-blind ablation. | If only synthetic gain or family shortcuts explain it, reject monitor claim. |
| H-POLICY | Existing selection-aware certification can control the complete best-of-`K` FARO system while retaining strong-look coverage. | Freeze `K`, scene-hash seeds, selector, full-resolution audit and identity fallback; adopt the closest valid certificate and test one policy on B3. | If either risk, style, style-qualified coverage or preference gate fails, make no FARO system risk-control claim. |
| H-THEORY | Clustered multirater labels and correlated colour candidates expose a certifiable gap not covered by current joint/noisy-label/non-exchangeable methods. | Dedicated statistics audit, explicit theorem and matched-valid simulation against LTT, two-stage and the 2026 joint certificate. | If no distinct theorem/tighter valid frontier exists, close the method-paper branch. |

---

## 8. Experiment DAG and gates

```text
R0 colour/evaluation literature + novelty freeze
  -> R1 FilmStyleSafe ontology and metric-failure study
       -> A0 development -> A1 hidden Paper-A test
       -> if no metric/style-definition gap: narrow/rename paper
  -> R2 global operator frontier
       -> strongest eligible global policy
       -> B1 complete-policy empirical-ceiling-over-global gate
            -> fail: stop adaptive research; productize global
            -> pass: R3 fixed-bank counterfactual scorer
  -> R4A adopt valid LTT/two-stage/joint complete-policy certificate
  -> R4T optional theory novelty audit; blocked/unresolved by default
  -> R5 B3 hidden end-to-end policy test
       -> both risks + look + style-qualified coverage + all-scene preference
  -> R6 optional proposer/local/physical/counterexample extensions
       -> each must improve the frozen frontier independently
  -> R7 external human replication and full ablation
       -> paper decision
```

### R1 — benchmark first

Deliverables:

- artifact ontology and annotation guide;
- independent reference-board/film-lineage manifest and look-adherence rubric;
- scene/rater sampling frames, cluster rule and label-error protocol;
- synthetic stress generator specifications;
- real baseline failure gallery;
- human mask pilot;
- conventional metric failure report;
- SCIS v0 with A0 development and A1 unseen-family confirmation.

Stop rule: do not tune FARO on final benchmark test images.

### R2 — strongest global frontier

Baselines:

- current safe_lab/#09/#56 family;
- identity and simple curves/manual LUT bank;
- Emulating-Emulsion-style physical core;
- Automatic Content-Aware Color and Tone Stylization;
- CanonCGT and source-similarity retrieval;
- image-adaptive 3D LUT, AdaInt, SepLUT and Neural Preset;
- SA-LUT/Hist2Style bounded local baselines;
- D-LUT/StatLUT-style operator generation where code/terms permit.

Gate: freeze the strongest global policy that passes the same full-resolution
renderer/export, both severe-risk gates and look-salience gate before testing
routing. With fixed `K`, render budget and bank, the B1 cross-rater complete
empirical-ceiling policy (best qualified nonsevere candidate, else identity)
must show statistically supported all-scene tie-score gain. Power and the
`N*K*r_s` annotation budget are approved before this matrix is built.

### R3 — fixed-bank counterfactual scoring

Baselines:

- random/global;
- content/style ranking from prior retrieval work;
- source-only photometric/semantic kNN;
- transform-only risk;
- InstantRetouch-style latent retrieval;
- CanonCGT-style canonical-pivot transfer;
- RetouchIQ/Hist2Style-style reward or style-quality scoring;
- cheap preview + transform-aware scorer.

Gate: significant empirical-ceiling-regret reduction across held-out
scene/source groups without source or operator-family shortcuts. The finalist
must survive the exact full-resolution export audit.

### R4A — adopted selection-aware complete-policy calibration

Required baselines are: simple best-of-`K`, best-of-`K` plus analytic veto,
candidate-wise calibrated rejection, a standard selective classifier and FARO
under LTT, two-stage risk control and the 2026 joint finite-sample certificate.
Reproduce the adopted guarantee and validate both overall/selected risk plus
style-qualified coverage. This is a reliability contribution to the system,
not claimed statistical novelty.

### R4T — optional theory branch, blocked on a fresh novelty audit

Only open this branch after mapping noisy-label, clustered/non-exchangeable and
joint selective-risk work. It requires a distinct theorem and a tighter
matched-valid frontier; otherwise close it and retain the system paper label.

### R5 — hidden end-to-end policy test

Freeze the one final policy before B3. Report identity fallback on every scene.
Overall risk, selected-output risk, look adherence, style-qualified adaptive
coverage and all-scene tie-score preference form one pre-registered
conjunction: missing any gate fails the FARO confirmatory claim.
Any claimed statistical risk control must name its loss, confidence level,
policy family, exchangeability assumption, multiplicity procedure and observed
shifts.

### R6 — optional extensions after the fixed-bank policy passes

Compare medoid retrieval, soft LUT mixing, TSFlow-like flow,
DiffRetouch/operator diffusion, physical cores, local bilateral residuals and
counterexample-trained monitors. Hold out complete failure/transform families
for monitor evaluation. Each extension must improve eligible policy utility or
style-qualified coverage at identical `K`/compute and preserve both risk
gates; raw diversity or synthetic AUROC is not a pass condition.

### R7 — external replication and paper ablation

Required ablations:

- eligible global vs complete empirical-ceiling policy vs selected;
- source similarity vs counterfactual outcome;
- deterministic vs set-valued proposer;
- soft averaging vs hard/selective choice;
- physical core on/off;
- global vs local residual;
- analytic monitor vs learned monitor vs both;
- counterexample training on/off;
- fallback and risk calibration on/off;
- adjudicated-label sensitivity/misclassification correction;
- SCIS vs established IQA metrics;
- nuisance-matched style controls.

---

## 9. Data strategy and claim discipline

### 9.1 Lanes

| Lane | Data role | Allowed claim |
|---|---|---|
| Digital query/stress | Diverse inputs, perturbation and deployment evaluation | Content/style robustness on declared distribution |
| Expert retouch/preference | Learn or validate visual preference | Photographic preference, not film authenticity |
| Unpaired film scans | Reference distributions and qualitative style evidence | `film-inspired`, unpaired evidence |
| Synthetic operator data | Coverage, counterexample and representation training | Operator behavior only |
| Controlled film/digital capture | Physical parameter fitting and held-out validation | Conditional calibrated stock/process/scanner claim |

### 9.2 Initial public baselines

Potential sources, subject to per-dataset terms and manifests:

- MIT-Adobe FiveK for expert retouching baselines;
- PPR10K for portraits, skin priority and group consistency;
- PPSD for pairwise personal photographic preference when released and
  licensed;
- FilmSet only as a Capture One pseudo-film baseline;
- PST50/iRetouch where terms permit method comparison;
- BAND-2k/FS-BAND/CAMBI for banding diagnostics;
- current rights-cleared project inputs and synthetic colour stress patterns.

Legacy Flickr rows remain quarantined until source, owner/group and license
lineage are complete. Quantity cannot override provenance.

### 9.3 Independence

- split by scene/source/uploader/camera/roll as applicable;
- no active operator/case derived from gold or test;
- hold out complete algorithm families for artifact-model evaluation;
- freeze calibration before final test;
- aggregate perturbations to their parent scene for confidence intervals;
- keep preference fitting, safety calibration and final human evaluation
  disjoint.

---

## 10. Paper plan

### Historical Paper A — benchmark/evaluation (primary path superseded 2026-07-12)

Working title:

> **ChromaticTail: Evaluating Strong Photographic Colour Under Local
> Severe-Artifact Constraints**

`FilmStyleSafe` is the film-inspired task suite inside the broader benchmark;
the title does not imply that unpaired references identify physical film.

Primary contributions:

1. non-compensable policy-eligibility gate plus all-scene deployed-output
   preference protocol;
2. local chromatic-artifact ontology and masks;
3. prospective, intent-aware SCIS validation and conventional-metric failure
   analysis across unseen transform families;
4. fixed full-resolution colour/export protocol and legitimate-local-edit hard
   negatives;
5. scene-level tail, worst-group, multirater-label sensitivity and
   risk–coverage reporting on an A1 hidden split.

Paper A is viable only if annotation evidence shows a real metric gap and SCIS
generalizes across unseen algorithms.

### Paper B-S — FARO system/application (conditional)

Working title:

> **FARO: Risk-Controlled Best-of-K Colour Operators for Strong Photographic
> Looks**

Primary contributions:

1. transform-outcome scoring and full-resolution auditing for explicit colour
   operators on the new chromatic-tail task;
2. a complete deterministic best-of-`K` system that adopts the closest valid
   LTT/two-stage/joint certificate for overall and selected-output risk;
3. end-to-end evidence of stronger target-look adherence and all-scene
   preference at pre-registered severe-risk and style-qualified-coverage gates.

The operator bank, physical core, local bilateral residual, stochastic
proposer, artifact monitor, preference model and statistical certificate are
components/baselines, not seven novelty claims. Paper B-S is viable only after
benchmark validity, target-look validity and the fixed-bank empirical-ceiling
gate. Its positioning is computer-vision system/application unless R4T
independently earns a new theorem.

### Paper B-T — statistical method (blocked / no contribution claimed)

LTT, two-stage risk control, noisy-label conformal work, non-exchangeable CRC
and the June 2026 joint risk–acceptance–utility certificate occupy the current
proposal. This branch has no working title or claimed method. It opens only if
R4T proves a non-redundant clustered/multirater/correlated-policy result and a
tighter matched-valid frontier; otherwise it is closed.

### Optional Paper C — calibrated film process

Extend the compact physical model to colour negative, print/scan
interpretations, process/scanner variation and uncertainty. This requires
controlled data and should not be conflated with aesthetic routing.

---

## 11. Engineering roadmap independent of paper success

1. Complete high-precision WorkingImage ingress/egress, ICC, 16-bit and declared
   colour state.
2. Implement a versioned recipe schema and deterministic global curve/LUT
   renderer.
3. Build a strong curated operator bank around preferred anchors.
4. Implement analytic artifact monitors and explicit identity/minimal fallback.
5. Add colour-only candidate comparison and diagnostics before effects.
6. Add bounded local residual only for demonstrated global failures.
7. Integrate grain/halation/bloom as separate layers after colour approval.
8. Add batch, preview/cache, cross-platform parity and non-destructive UI.

If every research hypothesis fails, this roadmap still produces useful
software.

---

## 12. Immediate ready leaves

| Leaf | Status | Output | Gate |
|---|---|---|---|
| FARO-R0 | complete by this document | Literature/novelty audit and new problem statement | Links and claims independently checked |
| FARO-R0T | ready, optional | Dedicated statistical novelty audit and theorem-or-close decision | Must distinguish LTT, two-stage, joint selective certificates, noisy-label and non-exchangeable risk control before Paper B-T opens |
| FARO-R1A | ready | Chromatic ontology + annotation schema + A0/A1 and B0-B4 split contract + sampling/preregistration/power worksheet | Covers current failures, legitimate-local hard negatives, rater error, style rubric, scene/rater frames and exact estimands |
| FARO-R1B | pending on R1A | Synthetic stress generator and baseline failure suite | No test leakage; parent-scene grouping |
| FARO-R1C | pending on R1B | Metric-failure pilot and SCIS v0 | A0 development; A1 hidden low-FPR validation on unseen transform families |
| FARO-R2A | ready after current renderer foundation | Versioned identity/curve/LUT operator contract | Property/golden tests |
| FARO-R2B | pending on R2A | Global operator frontier | Frozen strongest eligible champion |
| FARO-R2C | pending on R2B/R1 | B1 complete fixed-bank cross-rater empirical-ceiling policy and annotation budget | Identity on no-qualified-candidate scenes; all-scene risk/style/preference pass/fail/inconclusive |
| FARO-R3/R4A | blocked on empirical-ceiling pass | Fixed-bank scorer plus adopted complete-policy certificate | No adaptive training before bank value exists; both risk, style and style-qualified coverage gates required |
| FARO-R4T | blocked on R0T gap | Optional new statistical method | No method-paper claim without distinct theorem and matched-valid gain |

No costly model download, GPU training, public release or participant study is
authorized by this document. Those operations retain their existing approval
gates.

### 12.1 Definition of done for the research-program node

`FARO-R0` is done only when:

- the closest prior-art claims are linked to primary sources;
- novelty statements are framed as hypotheses rather than guaranteed facts;
- engineering and publication objectives are separated;
- every adaptive/model branch depends on the global-frontier and empirical-ceiling
  gates;
- evaluation includes overall and selected-output risk, protocol-adjudicated
  label uncertainty, target-look adherence, all-scene tie-score preference,
  tail and style-qualified coverage;
- active tracker, planning index, task board, project instructions and agent
  log all point to the same authority and status.

Downstream method nodes are not done by completion of this document.

### 12.2 Change propagation and bottom-up reintegration

This research reframe changes the U5 parent hypothesis and therefore propagates:

- upward to the `ULT` publication goal and product/research boundary;
- downward to FilmCase, which becomes a baseline/ablation;
- sideways to U4 evaluation, which supplies the FilmStyleSafe severe and
  preference evidence;
- into U1/U2, which provide the high-precision deterministic operator runtime;
- into U7/U8 only after method, legal and human-study gates pass.

Reintegration is bottom-up: validate R1 benchmark evidence, then R2 global and
empirical-ceiling evidence, then and only then admit R3–R6 adaptive work. A failed child
returns the parent to the simplest surviving global/product path; it does not
authorize a more complex sibling automatically.

### 12.3 Rollback and recovery

This node changes planning/docs only. Rollback is the scoped documentation
commit; existing renderer code, data, outputs and FilmCase artifacts remain
unchanged. If the novelty audit is later contradicted by closer prior art,
record the new source, narrow or retire the affected claim, update the tracker
and re-run the branch gate rather than rewriting historical evidence.

---

## 13. Final claim boundary

The strongest defensible near-term claim is:

> FARO studies whether a complete deterministic fixed-budget colour-operator
> policy can improve target-look adherence and all-scene tie-score preference
> over the strongest eligible global policy while satisfying pre-registered
> **marginal overall and selected-output protocol-adjudicated severe-risk**
> budgets and style-qualified coverage on a declared sampling frame.

It must not be written as:

- universally artifact-free;
- an exact named-film reproduction without controlled measurements;
- the first adaptive LUT, diverse retouching model, retrieval-based enhancer,
  physical film model or preference learner;
- the first complete-policy or joint risk–acceptance–utility certificate;
- safe merely because geometry is unchanged;
- safe because an individual candidate score passed a threshold;
- a per-image, OOD or universal guarantee derived from marginal calibration;
- superior because one average metric increased.

The project succeeds scientifically only if it advances the measurable
style–artifact–coverage frontier and survives the pre-registered stop rules. It
succeeds as engineering if it produces strong, reliable software even when the
adaptive research hypothesis is falsified.
