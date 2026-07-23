# U5.R2B fixed global operator frontier results

Date: 2026-07-23  
Node: `ULT > U5 > U5.R2 > U5.R2B`  
Decision: **B0 qualified challenger retained; no production promotion**

## Result

The fixed seven-policy global bank produces one qualified B0 development
challenger:

`anchor56_chroma_margin4_challenger`

It passes the frozen style, non-basic and clipping screen, remains visibly
stylized in three anonymous contact-sheet rounds, and has no confirmed severe
artifact in full-resolution inspection of all nine provisional gold images.
This is autonomous visual development evidence only. It is not a stock
response, calibrated digital-to-film operator, owner vote, cross-rater result,
hidden confirmation, population preference or production-default decision.

`anchor09_color_only` also passes the automatic screen, but full-resolution
ID11 inspection confirms the frozen neon-red speckle/posterization regression
across the bicycle-stem highlight. The hard severe-artifact veto rejects it.

## Reproducibility

- frozen set: 9 provisional gold + 32 stress inputs;
- seven primary candidates, 287 deterministic PNG outputs per pass;
- render manifest SHA-256, identical on two complete renders:
  `e0410f9ab8e5efe76a866fe8478148b0bb270651d560225d63eeb7faabe4f769`;
- automatic report SHA-256, identical on two complete evaluations:
  `bf5cfc7ae20057be965cc72524aa59cee72d342f6ad34389de942c07dd5c2b1b`;
- config SHA-256:
  `209d9a0676f4ce887bee463f1de26eff52eaba727df8015764d0755429231ad6`;
- blind pre-unblind review SHA-256:
  `331db3c97faf228dbfdae8fd88e72117d23501df3efe50a98db90b2ec1a1ea8c`;
- full-resolution review SHA-256:
  `2a20ac95f1d1f69d27f5f23c0ebf6c7284212dddf3a8a70ea14839061e7d29c5`;
- evaluator software commit: `5cff939046b136bf229280cca087d63c963176e3`;
- implementation verification: 9 focused evaluator/operator tests and 710
  complete CPU tests pass; both repeat-evaluation stderr logs are empty.

The inherited non-basic diagnostic is accurately labelled as the frozen joint
EV/white-balance/contrast/saturation affine fit. It does not contain a
separate global-luma curve.

## Automatic screen

| Candidate | Gold style ΔE76 | Gold non-basic ΔE76 | Worst gold new clipping | Worst stress new clipping | Automatic |
|---|---:|---:|---:|---:|---|
| bland safe-rich | 3.5855 | 2.1060 | 0.0000% | 0.0000% | fail style/residual |
| anchor 01 | 8.6200 | 6.6672 | 9.5167% | 5.9230% | fail clipping |
| anchor 09 | 7.1809 | 4.9333 | 0.0000% | 0.0000% | pass |
| anchor 53 | 9.7602 | 7.1636 | 10.4935% | 7.2361% | fail clipping |
| anchor 55 | 7.2389 | 4.9173 | 8.9499% | 5.5363% | fail clipping |
| anchor 56 | 8.2445 | 6.2086 | 9.5767% | 6.1289% | fail clipping |
| anchor 56 + chroma margin 4 | 8.3052 | 5.5166 | 0.0000% | 0.0000% | pass |

The result explains why the user's preferred raw anchors cannot simply become
the automatic product frontier: 01/53/55/56 retain strong style but create
roughly 9–10.5% worst-case new endpoint clipping. The margin-4 policy preserves
strong non-basic style without that failure.

## Autonomous visual gate

The three contact-sheet rounds were reviewed before the private mapping was
read. Safe-rich was correctly identified as materially bland in every round.
Among the two stylized survivors:

- round 1 preferred anchor 09;
- rounds 2 and 3 preferred the margin-4 anchor 56 challenger;
- contact sheets alone confirmed no severe artifact and correctly escalated
  saturated-red ID11 transitions to full resolution.

Every survivor was then inspected at full resolution on all nine gold images.
Anchor 09 fails decisively on ID11. The margin-4 challenger preserves faces,
text, fine texture, smooth walls/water/snow, highlights, shadows and geometry
without a confirmed severe glitch. Its strong cyan/blue shadow versus
orange/warm-highlight separation is a visible style, not merely a small
saturation increase.

## Decision and next gate

Retain `anchor56_chroma_margin4_challenger` as the strongest currently
qualified **B0 global Look Approximation challenger**. Do not change the
production default. Do not infer stock authenticity from its Velvia-derived
historical naming or from the A0/B0 preference anchors.

U5.R2C may now freeze the complete fixed-bank B1 empirical-ceiling policy and
annotation-budget feasibility. Any actual cross-rater recruitment or external
annotation remains separately authority-gated. In parallel, algorithm
research may test explicit statistics-to-LUT and hard-retrieval candidates
only on rights-cleared controls and synthetic known-operator witnesses; those
tests cannot bypass the closed real-film stock-identifiability, operator
fitting, training or LSM gates.

The Ultimate Goal remains ACTIVE.
