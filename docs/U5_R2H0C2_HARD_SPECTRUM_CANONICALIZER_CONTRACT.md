# U5.R2H0C2 hard measured-spectrum canonicalizer contract

Date frozen: 2026-07-24
Parent: `U5.R2H0C1`
Status: **preregistered before canonicalizer-output evaluation**

## Question

For a CAVE cell whose measured spectrum is hidden, does hard Top-1 retrieval
of a D65-Lab-nearest spectrum from a different scene predict its frozen
overlap-witness output better than the H0A smooth bounded reconstruction?

This is a deterministic canonicalizer comparison, not film learning. The
target is the witness output of a known CAVE reflectance. It is neither a film
scan nor a digital-to-film pair.

## Fixed population and leakage boundary

Reuse the exact H0C1 source audit, 31 retained scenes, 400--700 nm overlap,
16x16 cell medians and D65 validity filter. `watercolors_ms` remains excluded.

Every query is evaluated leave-one-scene-out:

- the retrieval bank contains all representatives from the other 30 scenes;
- no representative from the query scene may influence its retrieved spectrum;
- scene is the bootstrap and leakage group;
- RGB previews, scene names, content embeddings and witness output are not
  retrieval features;
- candidate selection sees only query D65 Lab and bank D65 Lab.

## Fixed baselines and candidate

For every query, compute the known target by rendering its measured spectrum
through the immutable H0C1 overlap witness.

1. `identity`: input D65 XYZ, a no-look diagnostic.
2. `smooth`: H0A bounded smooth spectrum reconstructed from query D65 XYZ,
   then rendered through the overlap witness.
3. `hard_top1`: the single Lab-nearest different-scene measured spectrum,
   rendered unchanged through the overlap witness.
4. `hard_top1_fallback_T`: use hard Top-1 only when Lab distance is at most
   `T`; otherwise return the smooth output.

Thresholds are frozen at `0.5, 1.0, 2.0, 3.0` Delta E76. There is no soft
blend, Top-K average, learned embedding, learned metric or neural predictor.

## Metrics

Primary error is Delta E76 between predicted and known target witness output.
For every threshold report:

- selected-query count, coverage, scenes and maximum scene share;
- selected hard-vs-smooth win/tie/loss rate;
- selected smooth and hard median/p95 error;
- selected median relative error reduction;
- full-population fallback-policy median/p95/mean error;
- full-population p95 change versus smooth;
- scene-bootstrap 95% intervals for selected win rate and median relative
  reduction, using 1,000 resamples and seed `20260724`;
- raw linear-sRGB range/gamut excursions and exact replay hashes.

Report nearest-distance and known-target effect distributions. Strong style is
not a pass by itself; the question is whether the missing spectrum is predicted
more accurately without broadening the tail.

Win/loss uses error difference beyond `1e-12` Delta E76; absolute differences
at or below that tolerance are ties.

## Eligibility gates and fixed selection rule

A threshold is eligible only if all hold:

- selected queries `>=256`;
- coverage `>=10%`;
- selected scenes `>=16`;
- maximum selected scene share `<=15%`;
- selected hard win rate `>=60%` and scene-bootstrap 95% LCB `>=50%`;
- selected median relative error reduction `>=20%` and bootstrap 95% LCB
  `>=0%`;
- selected hard error median `<=5.0`, p95 `<=12.0`;
- full fallback-policy p95 error is no worse than smooth p95.

If several thresholds pass, select the highest-coverage threshold; ties choose
the smaller threshold. This rule is frozen before results. The no-threshold
hard retrieval is diagnostic only and can never be selected.

## Branches

- no eligible threshold: close hard measured-spectrum retrieval; do not add a
  larger model to rescue the same source result;
- an eligible threshold: retain one internal empirical canonicalizer candidate
  and open external measured-spectrum replication before any visual/product
  work;
- broad source-group bootstrap or tail failure: mark source-limited/unstable,
  even if point medians improve;
- severe numerical/gamut failure: do not silently clip or visually promote.

No result opens film-pixel fitting, RGB generation, stock authenticity,
production integration, real-photo rendering, LSM or preference claims.

## Claim ceiling

One-source, leave-one-scene-out prediction of a synthetic datasheet-prior
witness target from D65 Lab via an explicit hard empirical canonicalizer.
