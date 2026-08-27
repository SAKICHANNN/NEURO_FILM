# BW2.D0 — HP5 / Tri-X fixed-K1 baseline contract

Status: **frozen before render** (2026-08-27)

## Question

On the exact 16-source CC0 digital population already frozen for RF3.D0, do the
unchanged HP5 and Tri-X safe-Lab K=1 profiles produce deterministic, exactly
achromatic, boundary-clean, and mechanically distinguishable outputs?

This is the separately requested black-and-white branch. It does not repair the
missing controlled Velvia 50 / Portra 400 / Ektar 100 evidence and does not
count toward multi-stock completion.

## Fixed candidate and controls

- candidates: existing `hp5` and `tri_x_400` statistics and `safe-rich`
  profile parameters;
- one identical 16-image CC0 source population;
- grain and dither fixed to zero;
- no fitting, strength sweep, router, latent mode, learned RGB, new data, or
  post-result threshold change;
- canonical and reverse-order fresh-process executions.

## Gates

Every output must be finite, bounded, byte-exact on replay, exactly neutral at
RGB8 (`R == G == B`), and introduce no new 0/255 boundary components. The
population median of per-source median CIE76 distance between HP5 and Tri-X
must be at least `1.0`, and at least 12 of 16 sources must individually reach a
median CIE76 distance of `1.0`.

## Decisions

- Pass: retain two mechanically distinct B&W Look Approximation controls and
  open only a later independent real-stock evidence search.
- Fail: recommend one generic B&W Look Approximation instead of presenting the
  current two heuristics as stock-distinct; do not tune this cohort.

## Claim ceiling

At most, deterministic mechanical separation and safety of two existing
film-inspired B&W Look Approximation profiles. No HP5 or Tri-X authenticity,
closeness, developer/process response, calibrated stock response, preference,
product promotion, or multi-stock completion claim.
