# U5.R2L1 — Full-Frame Strength Preflight Contract

**Status:** frozen before implementation or full-frame metric inspection
**DRPT level:** L2
**Parent:** U5.R2L0 material Oracle gap and severe-veto pass
**Primary writer:** current Codex Goal session

## Question

Can the post-output R2L0 Oracle be converted into the simplest legal
inference-time deterministic policy without ML: trial-render the immutable
density-cyan operator at `s0.65`, measure the exact full-frame RGB8 new-endpoint
fraction, and hard-fallback to `s0.50` only when it exceeds `0.005`?

The policy predicts no pixels. It executes one explicit colour operator,
inspects its candidate output before committing it, and may execute the same
operator at a lower fixed strength. It never blends outputs or changes
strengths.

## Immutable inputs

- exact R2L0 decision and 41-row selected manifest;
- exact E1 frozen set and source hashes;
- exact E0 density operator config and
  `cyan_shadow_warm_highlight_like` witness;
- exact strengths `0.65` and `0.50`;
- epsilon `1/510` and threshold `0.005`;
- no fitting, learned predictor, labels, new pixels or threshold search.

## Exact inference policy

1. decode explicit RGB8 sRGB input;
2. convert to float64 linear sRGB using the frozen E1 transfer;
3. apply the fixed density operator at `0.65`;
4. encode and quantize once to RGB8 exactly as E1;
5. compute full-frame new-hard-clipping fraction against the source RGB8;
6. return the already-quantized `0.65` bytes when the fraction is at most
   `0.005`;
7. otherwise render, encode and quantize `0.50` and return those bytes.

The trial output is discarded only on fallback. There is no spatial
resampling, effect, post-render repair or extra output quantization.

## Frozen gates

- source, config, parent and archived-output hashes all verify;
- regenerated `s0.65` and `s0.50` RGB8 bytes match the immutable E1 outputs for
  all 41 images;
- full-frame preflight assignments match the R2L0 Oracle assignments on all
  41 images;
- returned policy bytes match the R2L0 selected-output bytes on all 41 images;
- two reports are byte-identical;
- explicit tests cover threshold equality, fallback, RGB/range failures,
  deterministic replay and output-byte reuse;
- full CPU suite passes.

Exact assignment agreement is deliberately strict. If full-frame versus
sampled endpoint statistics disagree on any image, this primary policy closes;
the threshold, epsilon, strengths and metric may not be retuned on the same 41
results.

## Visual inheritance

Exact selected-byte replay inherits only the completed R2L0 B0 visual evidence
on the exact 41 images. It does not create a new universal severe-artifact
guarantee. Any byte mismatch requires a new visual review.

## Branches

- **Exact pass:** retain a research-only deterministic hard preflight and
  measure full-resolution runtime/memory as a non-binding product diagnostic.
  Do not train a router merely to avoid one trial render.
- **Any assignment or byte mismatch:** close this rule without threshold or
  sampling rescue; fixed `s0.50` remains the fallback product-research
  candidate.
- **Any severe issue on changed bytes:** veto regardless of automatic gain.
- **Later OOD/full-product failure:** retain fixed `s0.50`; no dense strength
  mixing.

## Claim ceiling

B0 deterministic replay evidence for a full-frame clipping-limited hard
strength preflight over one fixed film-inspired density Look Approximation.
No universal safety, learned routing, preference, digital-to-film
identification, named-stock response, calibration, authenticity or production
claim.
