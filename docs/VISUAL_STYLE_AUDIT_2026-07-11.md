# Velvia 50 Visual Style Audit — 2026-07-11

> Role: visual evidence note and next-experiment specification.
> Product standard: maximize strong, attractive stylization subject to a hard severe-glitch/artifact veto.
> This document records visual hypotheses, not calibrated-film truth.

---

## 1. Evidence inspected

### Numbered overview

- `outputs/contact_sheets/velvia50_all_schemes_numbered_20260616/MAIN_COMPARABLE_ORIGINAL_OVER_OUTPUT_NUMBERED.png`
- `outputs/contact_sheets/velvia50_all_schemes_numbered_20260616/NUMBER_MAP.csv`
- `outputs/contact_sheets/velvia50_union_all_models_20260616/ALL_56_SCHEMES_UNION40_ORIGINAL_OVER_OUTPUT_NUMBERED.png`

### User-preferred full schemes

| Number | Scheme | Original recipe evidence |
|---:|---|---|
| 01 | `baseline_current` | union run ID `s0p55_l0p35_g0p012`; aggressive legacy baseline |
| 09 | `baseline_noclip_s0p50` | `strength=0.50`, `luma_strength=0.25`, `grain=0`, source-gamut compression, output margin 4 |
| 53 | `velvia50_digital20_s0p72_gamutsafe` | original manifest `strength=0.72`, `luma_strength=0.42`, `grain=0`; different original source set |
| 55 | `velvia50_rawpixls20_s0p50_gamutsafe` | original manifest `strength=0.50`, `luma_strength=0.25`, `grain=0`, output margin 4 |
| 56 | `velvia50_rawpixls20_s0p58_gamutsafe` | original manifest `strength=0.58`, `luma_strength=0.38`, `grain=0` |

The smoke-only preferences `02/03/33` were not used for cross-scene conclusions.

### Technical comparators

- `baseline_saferich` (#10)
- `color_engine_challenge/local_maps_tuned_s1p2`
- `neural_film_lut_v2/scheme_b_nilut_distilled_v1_s1p0`
- `neural_film_lut_v2/scheme_c_context4d_distilled_v1_s1p0`

### Full-resolution stress inspection

The highest metric-clipping cases were inspected at original resolution for IDs 09, 11 and 29, comparing input, #09 and #56. This is still a small targeted audit, not completion of the U4 gold/stress review.

---

## 2. Visual signature of the preferred family

The following are hypotheses inferred from the inspected outputs. They require blinded confirmation from the user.

### High-confidence pattern

1. **Cool-shadow / warm-subject separation**
   - shadows, dark furniture, dark metal and some skies move toward cyan/blue;
   - red/orange/yellow subjects and highlights become more forceful;
   - the result reads as a palette decision, not merely a uniform saturation gain.

2. **Medium-to-strong transform strength**
   - #55 and #56 are directly comparable on the same raw.pixls set;
   - the user likes both `0.50` and the visibly stronger `0.58` result;
   - #53 indicates tolerance for an even stronger `0.72` transform on a different digital set.

3. **Luminance movement is part of the appeal**
   - preferred recipes use roughly `luma_strength=0.25–0.42`;
   - safe-rich uses `luma_strength=0.02` and visually stays much closer to the digital input;
   - aggressively locking luminance removes some of the perceived film styling.

4. **Distinct hue separation matters more than raw chroma amount**
   - flowers, foliage, red architecture and blue sky receive differentiated hue/tone treatment;
   - local maps mainly add mild richness and remain visually close to the input;
   - NILUT can have high aggregate chroma but still lacks the preferred stable palette and scene-to-scene consistency.

5. **Geometry and texture remain native**
   - the selected deterministic family retains edges, text layout and scene geometry;
   - this is a major reason it can accept stronger color than VAE/generative paths.

### Medium-confidence accents

- cyan/teal shadows are preferred over neutral or warm shadows;
- warm reds/oranges should stay luminous rather than becoming dark maroon;
- sky blue may move slightly toward cyan while clouds/highlights retain separation;
- green/yellow foliage should feel vivid, but neutral snow, walls and skin should not inherit a blanket cyan/magenta cast;
- mild toe/contrast is welcome; crushing dark regions is not necessary for the desired style.

---

## 3. Why technically strong candidates can look weak

### Safe-rich (#10)

Visual behavior:

- excellent containment and very high luminance similarity;
- only a modest cool/warm shift;
- often reads as careful correction rather than a committed film look.

Likely cause:

- profile defaults are conservative: `strength=0.35`, `luma_strength=0.02`, `preserve_luma_detail=0.90`, `neutral_protect=0.55`, `skin_protect=0.35` for Velvia;
- several simultaneous protections pull the transform back toward the input.

### Local bounded maps

Visual behavior:

- minimal local enrichment and no obvious severe geometry artifact;
- insufficient palette identity;
- matches the user's earlier conclusion that it looks mostly like extra saturation.

Likely cause:

- the old gate explicitly rewarded chroma gain;
- it did not require a recognizable cross-scene hue/tone signature.

### NILUT / context-4D variants

Visual behavior:

- larger tone/chroma movement than safe-rich;
- some scenes are darker or more contrast-heavy than neighboring scenes;
- the transform feels less coherent across snow, flowers, architecture and indoor light;
- stronger change alone does not reproduce the preferred cyan-shadow/warm-subject relationship.

Likely cause:

- the candidates were distilled from a pseudo-teacher rather than optimized for the user's visual preference;
- aggregate metrics do not encode palette coherence or scene-to-scene style identity.

---

## 4. Quantitative contradiction that supports visual-first selection

The existing union-40 metrics are useful diagnostics but are not a frozen gold benchmark.

| Scheme | Images with metric clipping | Mean L-SSIM | Mean chroma | Visual role |
|---:|---:|---:|---:|---|
| 01 | 40/40 | 0.9324 | 18.60 | preferred aggressive anchor |
| 09 | 0/40 | 0.9927 | 17.66 | preferred safe anchor |
| 53 | 40/40 | 0.9324 | 18.91 | preferred aggressive upper anchor |
| 55 | 40/40 | 0.9390 | 18.05 | preferred medium anchor |
| 56 | 40/40 | 0.9367 | 18.31 | preferred stronger medium anchor |
| 10 safe-rich | 0/40 | 0.9979 | 17.09 | technically safe, visually subdued comparator |
| local maps | 0/40 | 0.9979 | 17.68 | technically safe, weak style-identity comparator |
| NILUT-B | 0/40 | 0.8100 | 19.56 | strong movement, inconsistent/nonpreferred comparator |

Conclusions:

1. High L-SSIM and zero clipping do not imply strong style.
2. High mean chroma does not imply the desired film look.
3. Pixel clipping is a screening signal, not the same thing as a visually severe artifact.
4. #09 is the current engineering bridge: it is user-preferred, visually styled, and achieves zero new clipping in the recorded union metrics.
5. The union reruns for #53/#55/#56 do not reproduce the safety implied by their original names; rerun provenance and flags must be normalized before treating them as canonical anchors.

---

## 5. Newly observed artifact risk

Full-resolution inspection exposed a failure that aggregate clipping alone does not describe:

- in the red-lit bicycle/ColorChecker scene (union ID 11), both #09 and #56 amplify the saturated red highlight on metal into conspicuous neon-red speckling/quantized patches;
- #09 reports zero new clipping yet still shows this local chroma artifact;
- therefore `no clipping` is necessary but not sufficient for the product standard.

Add the following to the severe/moderate rubric:

- high-chroma highlight speckling;
- sensor-noise amplification into colored islands;
- hard chroma shelves around specular regions;
- false neon edges on metal, skin or text;
- smooth-wall/highlight gradients that become blotchy or banded.

The indoor lamp and bridge stress images otherwise preserve geometry and smooth large structures. The bridge confirms that #56 can increase style without obvious spatial glitch; the red-lit bicycle is the more discriminating color-artifact case.

---

## 6. Recommended style-safe bridge experiment

### Goal

Preserve the palette strength of #55/#56 while retaining the containment of #09 and reducing local chroma artifacts.

### Non-goals

- do not optimize stock authenticity in this experiment;
- do not add grain, halation or bloom before the color winner is selected;
- do not enable the full safe-rich guardrail bundle, which is already visually too conservative;
- do not compare different source sets or non-normalized recipe implementations.

### Required source set

Use one new frozen manifest derived from the existing union-40 only as a starting point, then add/verify dedicated stress cases for:

- face and varied skin;
- hand/fingers;
- small text and numbers;
- red-lit metal/ColorChecker (union ID 11);
- neutral snow/white wall;
- blue sky/cloud gradient;
- saturated red/yellow flowers;
- foliage;
- dark interior and specular highlight;
- architecture/bridge fine edges.

The current union-40 is provisional evidence, not yet U4 gold, because recipe provenance differs and some safety labels do not match the rerun metrics.

### Stage A — eight color-only variants

Freeze:

- `grain=0`;
- `output_margin=4`;
- no full `--use-guardrails`;
- identical input manifest, decode path, seed and output format.

Cross these four style/luma pairs with two gamut modes:

| Variant pair | Strength | Luma strength | Rationale |
|---|---:|---:|---|
| A | 0.50 | 0.25 | exact #09/#55 region |
| B | 0.55 | 0.30 | bridge between #09 and #56 |
| C | 0.58 | 0.35 | #56-like strength with normalized safety |
| D | 0.62 | 0.38 | controlled aggressive challenger below #53 |

Gamut modes:

1. `source` — existing #09 behavior;
2. `chroma` — per-pixel chroma compression, expected to reduce colored highlight artifacts.

This yields eight candidates. Add `tone_rolloff=0.04`, `shadow_floor_l=1`, `highlight_ceiling_l=99` only if applied identically to all eight; otherwise defer rolloff to Stage B.

### Stage B — selective protection on the top two

For the two visual winners, test only light protections:

- `neutral_protect ∈ {0.0, 0.15}`;
- `skin_protect ∈ {0.0, 0.15}`;
- `chroma_curve_strength ∈ {0.0, 0.15}` focused on already-saturated source regions;
- optional `dither=0.20` only if gradient inspection shows quantization.

Do not start from the current Velvia full guardrails (`neutral=0.55`, `skin=0.35`, chroma curve `0.45`), because the safe-rich visual already shows that this combination suppresses too much style.

### Stage C — full look

Only after a color-only winner:

- add grain, halation and bloom one at a time;
- visually score whether each effect increases overall appeal;
- reject any effect that hides color artifacts or creates new severe artifacts.

---

## 7. Visual scoring sheet

For every image/candidate, collect three independent judgments:

| Field | Scale | Rule |
|---|---|---|
| Severe artifact | yes/no + category | any confirmed `yes` blocks the candidate on gold |
| Style strength | 1–5 | how strongly stylized/film-inspired it appears |
| Overall appeal | 1–5 or pairwise choice | how much the user actually likes the result |

Also tag moderate issues without automatic rejection:

- skin cast;
- neutral contamination;
- crushed shadow/highlight;
- local chroma speckle;
- banding/posterization;
- halo/seam;
- excessive palette inconsistency.

Selection rule:

1. remove candidates with a confirmed severe gold-set failure;
2. form the Pareto frontier of style strength and appeal;
3. use moderate-issue count, determinism and performance as tie-breakers;
4. preserve #09 and #56 as named reference columns in every sheet.

---

## 8. Stop and branch rules

- If normalized #09 remains preferred, ship/improve it rather than adding a model.
- If `chroma` gamut mode removes red speckling without reducing style, promote it as the new safety mechanism.
- If all soft protections make the look bland, protect only the demonstrated failure regions instead of applying global guardrails.
- If none of the deterministic variants surpasses #56 without severe artifacts, then open the bounded LUT/grid challenge.
- If a neural candidate is stronger but inconsistent across scenes, keep the deterministic champion.
- Do not infer a universal Velvia preference from this one observer; this project is explicitly optimizing the owner's product taste first, with broader beta testing later.

---

## 9. Current conclusion

Vision materially changes the recommendation:

> The next target is not “make safe-rich stronger” and not “choose the best automatic metric.” It is **#56-like cool/warm palette strength with #09-like containment, plus a new defense against local high-chroma speckling.**

This is implementable with the existing deterministic Lab renderer before introducing any learned model.
