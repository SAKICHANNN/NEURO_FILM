# Roll2Film ultimate research audit — integration decision

> Date: 2026-07-15
>
> Scope: evidence integration, not a new experimental result
>
> Raw external response: [`docs/reference/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_20260715_CN.md`](../reference/ROLL2FILM_ULTIMATE_RESEARCH_AUDIT_20260715_CN.md)
>
> Preserved SHA-256: `C051CF747A00EB98F50B5654EEE4CEE6DBABAC5834378DB49A2525ABB292E27F`

## Decision

The audit is strong enough to correct the active research order, but not to promote its visual judgements or draft thresholds as project results.

Adopt this split immediately:

1. **Product mainline:** fixed strong explicit experts → strength/basic-adjusted diversity gate → optional transform-aware hard selector → image-level validated support/strength path → explicit full-resolution render → severe-artifact veto.
2. **Roll2Film research challenger:** grouped-target explicit colour-map estimation, initially L0–L2, which must beat fixed-sample matched controls before it may claim that roll grouping supplies special information.
3. **FilmCase:** only opens after at least two non-duplicate operator modes and a legal-supervision Oracle gap exist. Otherwise ship one deterministic champion plus bounded strength control.
4. **Calibrated stock lane:** remains closed. FilmSet is Capture One recipe truth; BlueNeg is negative/archive truth. Neither is digital-to-film or stock/process/scanner calibration.

This is not “ML has failed.” It is a stricter answer: ML remains justified only where it predicts a demonstrably useful bounded operator, selector or support strength. A learned component is not entitled to exist merely because it is learnable.

## What the audit changes

| Previous working interpretation | Corrected interpretation | Consequence |
|---|---|---|
| E0 group-size improvement supports roll repeated-measure information | E0 varies total pixels with frame count; it currently proves estimator consistency under matched affine truth and detects a mixed-operator misspecification | Redo E0 at fixed total pixels, fixed colour coverage and controlled nuisance before any roll-information claim |
| 53/55/56 are several preferred experts | Same-input displacement diagnostics strongly indicate one near-collinear style/strength path | Cluster after best-basic and legal strength alignment; count them as one mode unless residual operator distance survives |
| 09 is the contained/safe anchor | The external full-resolution review reports red speckle on 09 as well as 53/55/56 for ID 11 | Freeze 09 and 56 as worst-case evidence; do not define safety from low strength or gamut margin alone |
| FilmSet must be restored/downloaded after CT3 | The complete decompressed archive is already local | Freeze its manifest now; no new download approval is needed for local internal research |
| FilmSet has 638 test inputs | The paper reports 638, but the distributed archive has 628 and the four-domain arithmetic closes only at 628 | Runtime/evaluator uses 628; provenance preserves `paper_reported_test_n=638` |
| Support should be a per-colour shrink mask | Per-cell blending can break invertibility and introduce banding | Start with a whole-image scalar along a family-valid champion→expert parameter/velocity path |
| Set encoder is the natural Roll2Film model | Amortisation before a known-truth MAP solver is valid can learn scanner, JPEG, metadata or scene shortcuts faster | MAP/known-truth gates first; set encoder only after the solver survives |
| Nonlinear/local capacity may be needed for stronger style | Global colour management and L0–L2 capacity are not yet exhausted | L3/L4 require stable residual evidence; L5 requires repeated local failure and is deleted on any severe local artifact |

## Evidence adjudication

### Adopted as verified fact

- The preserved response is byte-identical to the supplied file: 87,776 bytes and the SHA-256 above.
- The decompressed local FilmSet image tree contains 21,140 files and 11,262,805,356 bytes: four train branches of 4,657 and four test branches of 628.
- `data/processed/manifest.jsonl` exists, and the current Windows `data/film_domain` contains 4,212 JPEGs. Eligibility remains governed by the earlier fail-closed lineage audit; count does not imply usable training data.
- The FilmSet paper reports 5,285 originals but also 4,657 train plus 638 test, an arithmetic contradiction. The archive-observed 628 is operational truth; 638 remains publication provenance.
- The [ICLR 2024 identifiable UDT paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/8bb5a934785817f752e7f9322f9b4d54-Paper-Conference.pdf) relies on matching multiple pairs of corresponding cross-domain conditional distributions. Target-only roll groups do not directly satisfy that premise.
- The [BlueNeg paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.pdf), [dataset tree](https://huggingface.co/datasets/ttgroup/blueneg-release/tree/main) and [custom licence](https://huggingface.co/datasets/ttgroup/blueneg-release/blob/main/LICENSE) support a roll-metadata restoration/archive pilot and its attribution rule, not a digital-to-film target claim.
- [Emulating Emulsion](https://musicofmusix.github.io/siggraphposters25) is a critical nearest work: controlled RAW/chart pairs, an explicit capture/develop/scan model and approximately 30 parameters already occupy compact physical film-operator territory.

### Adopted as experimental requirements, not facts

- Fixed-total-sample E0, partition equivalence, independent-support versus repeated-support, nuisance-capacity, prior-swap and scanner/metadata shortcut controls.
- Best-basic adversaries that match EV, WB, tone, chroma and overall change magnitude before claiming non-basic style.
- Operator-space diversity after a legal strength-path alignment.
- Image-level support scalar before cell-wise support.
- FilmSet content-disjoint pair blindness with exact, perceptual and embedding leakage canaries; 4,657 internal development identities and one final 628 lockbox evaluation.
- Lexicographic promotion: severe veto, then style floor, then fidelity/appeal and product quality. No weighted score may compensate for a severe failure.

### Valuable but not yet project evidence

- The external reviewer’s verdict that ID 11 outputs 09/53/55/56 are severe failures. It is credible autonomous visual evidence and must seed the worst-case set, but it is not a completed internal three-pass blinded U4 adjudication.
- The cosine/RMS analysis showing 53/55/56 as one strength path. It is a strong diagnostic, not yet the frozen Hald/operator-grid diversity result.
- Draft numeric gates such as 10% group benefit, 20% paired-gap closure, 90%/80% style retention or 15% local gain. They must be estimated and frozen on pilot/dev data before confirmatory use.
- Licence interpretations beyond explicit text. Internal research access is not public-weight, example-image or commercial-release clearance.
- The broad novelty table. The decisive nearest works were checked; every 2025–2026 entry still needs source-level verification before a paper claim matrix is frozen.

## Why this is a better answer to the original ML question

The original failure mode was averaging many photographs into a bland global look. The audit’s product architecture avoids that collapse without permitting unconstrained RGB synthesis:

- a fixed bank preserves strong, visibly distinct transforms;
- a hard selector can choose a transform because it is safe/applicable to this image, but only if an Oracle proves per-image choice has value;
- a bounded scalar controls extrapolation without averaging incompatible looks;
- the final image is always rendered by an inspectable colour operator;
- ML is allowed to predict operator identity, parameters or support, but it must beat deterministic and no-neural baselines at matched style strength.

The phrase “this photo resembles a training case, so grade it that way” therefore becomes a falsifiable transform-applicability problem, not generic visual retrieval. Generic nearest-neighbour retrieval, photometric retrieval and no-neural rules are mandatory baselines. If they equal the learned ranker, the learned ranker is removed.

## Revised execution order

1. Freeze this response, attachment evidence and local dataset counts/hashes.
2. Finish the high-precision colour-state path; the legacy early sRGB8 adapter can create or hide exactly the artefacts under study.
3. Implement and validate canonical L0 exposure/WB, L1 orientation-preserving affine and L2 affine-plus-monotone-spline operators, including inverse/Jacobian and 33³/65³ bake parity. **L2 core passed; explicit L0/gauge/shaper closure remains.**
4. Redo E0 with a fixed sample budget and the complete falsification matrix. **Affine and L2 method controls passed without establishing real-roll information.**
5. Freeze FilmSet manifest and pair blindness. **Complete: 4,657 identities split 2,096/2,096/465; all 628 official test identities remain inaccessible until policy freeze.**
6. Build strength-matched deterministic/classical/learned baselines and test whether at least two substantive expert modes exist.
7. Run the Oracle gate. No Oracle gap means no FilmCase. A rules-based selector victory is an acceptable final result.
8. Pilot image-level support strength against simple global-strength and chroma/gamut margins on ID 11 and the wider stress set.
9. Only after E0 survives, request approval for the approximately 0.956GB BlueNeg preview/pseudo-GT pilot and resplit by whole roll.
10. Open L3/L4, an amortised set encoder or L5 only when the preceding level has a preregistered, repeated failure it cannot solve.

## Current claim ladder

| Evidence state | Maximum honest language |
|---|---|
| Current E0 | `known-operator grouped-target estimator unit test` |
| Fixed-N synthetic gates pass | `canonical explicit colour-map estimation from grouped target observations` |
| Correct real groups beat matched controls | `grouped-target unpaired explicit colour-operator identification under per-frame nuisance` |
| BlueNeg survives roll-disjoint shortcut controls | `archive/scanner-specific roll-look information` |
| FilmSet hidden transfer succeeds | `film-recipe transfer`; never real-film calibration |
| Controlled digital/film/process/scanner pairs succeed | only then a stock/process calibrated claim |

## Bottom line

The audit does not kill Roll2Film; it removes the circular reasoning that would have made a positive result untrustworthy. The strongest near-term route is a deterministic system that can win without ML, while Roll2Film earns the right to become the paper core only by proving that correct grouped observations add information beyond sample count, colour coverage, scanner and nuisance. That is a substantially stronger scientific and product position than committing to a model class in advance.
