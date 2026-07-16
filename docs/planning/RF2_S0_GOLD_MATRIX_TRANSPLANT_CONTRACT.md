# RF2.S0 Gold archive-display matrix transplant contract

## Question and stop rule

RF1.4B1 established a repeatable BlueNeg archive preview-to-display mapping,
not a digital-to-film transform. RF2.S0 asks the smallest useful next question:
does the selected fixed 3x3 mapping remain visibly stylized, non-basic and
artifact-safe when transplanted to already frozen digital photographs?

Failure closes automatic transplant. It does not trigger a neural model, more
capacity, frame-random fitting or a weaker stock-authenticity definition.

## Frozen inputs

- all 47 official Gold pairs and six physical rolls from RF1.4B1;
- the existing provisional FilmCase set: 9 gold and 32 stress images, with
  every source hash already recorded;
- the existing same-input normalized `55`, `56` and safe-rich replays;
- no new image, owner vote, pair, label or download.

The provisional set supports this bounded falsification only. It is not silently
promoted into the final U4 population benchmark.

## Fixed operator and strength path

Fit one bounded ridge 3x3 affine on all six rolls using the exact RF1.4B1
sampling, roll/frame balancing, coefficient bounds, ridge and positive-
determinant rule. No digital pixel participates in this fit.

Test strengths `1.0`, `0.75`, `0.5` in that frozen strongest-first order using
identity interpolation. The first strength satisfying automatic safety, style
and residual gates becomes the sole visual candidate. This is a deterministic
selection rule, not post-hoc tuning.

## Anti-bland/basic-adjustment gate

Before candidate rendering, the same 4096-pixel diagnostic on the existing
anchors produced these gold-set image medians:

| Replay | style Delta E76 | residual after best EV/WB/contrast/saturation |
|---|---:|---:|
| anchor 55 | 7.1769 | 4.9711 |
| anchor 56 | 8.1337 | 6.2062 |
| safe-rich | 3.5788 | 2.1452 |

The candidate must reach style `>=7.0` and residual `>=4.9`. The matched basic
operator is fitted per image to the candidate output for diagnostic purposes
only; it is not fitted to a film target, anchor or held-out truth and cannot be
selected as the product result.

## Domain and safety gates

Archive input support is frozen without looking at digital distances: RGB
means/std plus luma 5/50/95% quantiles, standardized by archive median/MAD.
The threshold is the 95th percentile of archive leave-one-physical-roll-out
nearest-neighbour distances. At least 8/9 gold and 75% of stress images must be
eligible; other images fall back to safe-rich.

For the selected strength, every gold image has raw clip fraction `<=0.5%`, all
outputs are finite and the composite matrix determinant remains positive.
Only then generate three deterministic blind rounds comparing source,
safe-rich, 55, 56, matrix and matched-basic. All nine gold images require visual
review and any confirmed severe artifact rejects the transplant.

## Claim boundary

Even a complete pass yields only a fixed, film-inspired `Look Approximation`
challenger. It does not prove Kodak Gold identity, isolated emulsion response,
digital-to-film truth, calibration, authenticity, `S2`, or release readiness.

Machine contract: `configs/real_film_gold_matrix_transplant_v1.json`.
